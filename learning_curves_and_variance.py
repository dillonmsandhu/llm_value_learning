from __future__ import annotations
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, List

from envs.tree_mrp import TreeMRPExactValue, TreeMRP
from envs.tabular_matrix_env import TabularParams
from vine import estimate_vine_all_states
import mc_value
import vine_target_value
import fbi_value


def compute_sample_costs(
    num_envs: int,
    num_steps: int,
    depth: int,
    k_vine: int | str = 0,
) -> int:
    """
    Calculates the exact number of environment transitions per update step.
    
    For Standard MC:
      num_envs primary trajectories of length num_steps = num_envs * num_steps transitions.
      
    For Vine-Target:
      num_envs * num_steps primary transitions.
      PLUS: for each visited state at depth d (0 <= d < num_steps),
            we fork K rollouts of remaining length (depth + 1 - d).
            
    For FBI:
      num_envs * depth base transitions + num_envs * depth * K 1-step branching transitions.
    """
    if isinstance(k_vine, str) and k_vine.startswith("fbi"):
        parts = k_vine.split("_")
        k_b = int(parts[1])
        return fbi_value.compute_sample_costs(num_envs, depth, k_b)

    primary_transitions = num_envs * num_steps
    if isinstance(k_vine, int) and k_vine <= 0:
        return primary_transitions

    k_int = int(k_vine)
    vine_transitions_per_env = sum(k_int * (depth + 1 - d) for d in range(num_steps))
    total_vine_transitions = num_envs * vine_transitions_per_env
    return primary_transitions + total_vine_transitions


def run_experiment_with_transitions(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    k_vine_list: Tuple[int, ...] = (4,),
    include_fbi: bool = True,
    fbi_lambdas: Tuple[float, ...] = (0.0, 0.5),
    total_transitions_budget: int = 150000,
    n_seeds: int = 10,
    seed: int = 42,
    save_dir: str = "results/figures",
):
    """
    Runs multi-seed training where the horizontal axis is strictly the
    TOTAL NUMBER OF SAMPLED TRANSITIONS.
    
    Compares:
      1. Standard MC Critic (K=1)
      2. Vine-Target Critic (K=4) + Online Precision-Weighted Interpolation Audit
      3. FBI(lambda=0.0) (Pure 1-Step FBI)
      4. FBI(lambda=0.5) (Hybrid 1-Step FBI + Monte Carlo)
      
    Computes:
      - Learning curves (MSE vs transitions, Base transitions, Updates)
      - Exact Bias-Variance Decomposition across seeds:
          MSE(s) = Bias^2(s) + Variance(s)
      - Internal Decision Node MSE and Sibling Advantage Ranking Accuracy
    """
    print("=" * 85)
    print(f"SAMPLE-EFFICIENT VALUE LEARNING & BIAS-VARIANCE EXPERIMENT")
    print(f"Budget: {total_transitions_budget} total sampled transitions across {n_seeds} seeds")
    print(f"Tree MRP: Depth={depth}, p={p}, ClusterSize={cluster_size}")
    print("=" * 85)

    evaluator = TreeMRPExactValue(depth=depth, p=p, cluster_size=cluster_size)
    v_true = evaluator.solve_backward_induction()[:evaluator.num_states]
    num_states = evaluator.num_states
    num_internal = evaluator.leaf_offset
    num_envs = 64
    num_steps = depth + 1

    rng = jax.random.PRNGKey(seed)

    methods = [("Standard MC (K=1)", 0)] + [(f"Vine-Target (K={k})", k) for k in k_vine_list]
    if include_fbi:
        for lam in fbi_lambdas:
            label = f"FBI (K=4)" if lam == 0.0 and len(fbi_lambdas) == 1 else f"FBI (K=4, lam={lam})"
            methods.append((label, f"fbi_4_{lam}"))
    results = {}

    for name, k_val in methods:
        transitions_per_update = compute_sample_costs(num_envs, num_steps, depth, k_vine=k_val)
        num_updates = max(10, total_transitions_budget // transitions_per_update)
        actual_total_transitions = num_updates * transitions_per_update
        total_timesteps_for_cfg = num_updates * (num_envs * num_steps)

        print(f"\n--- Method: {name} ---")
        print(f"  Transitions per update step: {transitions_per_update} (Primary: {num_envs * num_steps})")
        print(f"  Number of updates: {num_updates} | Total transitions: {actual_total_transitions}")

        rng, method_rng = jax.random.split(rng)
        seed_keys = jax.random.split(method_rng, n_seeds)

        if k_val == 0:
            # Standard MC
            cfg = mc_value.default_config.copy()
            cfg.update({
                "DEPTH": depth,
                "P_SUCCESS": p,
                "CLUSTER_SIZE": cluster_size,
                "NUM_ENVS": num_envs,
                "NUM_STEPS": num_steps,
                "TOTAL_TIMESTEPS": total_timesteps_for_cfg,
                "N_SEEDS": n_seeds,
            })
            train_fn = jax.jit(jax.vmap(mc_value.make_train(cfg)))
            out = train_fn(seed_keys)
        elif isinstance(k_val, str) and k_val.startswith("fbi"):
            # FBI(lambda): 1-Step Fitted Backward Induction
            parts = k_val.split("_")
            k_b = int(parts[1])
            lam_val = float(parts[2]) if len(parts) > 2 else 0.0
            cfg = fbi_value.default_config.copy()
            cfg.update({
                "DEPTH": depth,
                "P_SUCCESS": p,
                "CLUSTER_SIZE": cluster_size,
                "K_BRANCHES": k_b,
                "LMBDA": lam_val,
                "NUM_ENVS": num_envs,
                "TOTAL_TIMESTEPS": actual_total_transitions,
                "N_SEEDS": n_seeds,
            })
            train_fn = jax.jit(jax.vmap(fbi_value.make_train(cfg)[0]))
            out = train_fn(seed_keys)
        else:
            # Vine-Target
            cfg = vine_target_value.default_config.copy()
            cfg.update({
                "DEPTH": depth,
                "P_SUCCESS": p,
                "CLUSTER_SIZE": cluster_size,
                "K_VINE": k_val,
                "NUM_ENVS": num_envs,
                "NUM_STEPS": num_steps,
                "TOTAL_TIMESTEPS": total_timesteps_for_cfg,
                "N_SEEDS": n_seeds,
            })
            train_fn = jax.jit(jax.vmap(vine_target_value.make_train(cfg)))
            out = train_fn(seed_keys)

        metrics = out["metrics"]
        x_transitions = (jnp.arange(1, num_updates + 1) * transitions_per_update)

        # Evaluate final predictions of each seed across all states
        final_params = out["runner_state"][0].params
        model = mc_value.TransformerCritic(
            d_model=cfg["D_MODEL"],
            n_heads=cfg["N_HEADS"],
            n_layers=cfg["N_LAYERS"],
            final_hidden_dim=cfg["FINAL_HIDDEN_DIM"],
            max_len=depth + 4,
        )
        all_obs = evaluator.obs_stack[:num_states]

        eval_fn = jax.vmap(lambda p: model.apply(p, all_obs, method=model.value))
        v_preds_all_seeds = eval_fn(final_params)  # (n_seeds, num_states)

        # Compute sibling ranking accuracy and internal node MSE across seeds
        rank_acc_seeds = jax.vmap(
            lambda preds: fbi_value.compute_sibling_ranking_accuracy(preds, v_true, num_internal)
        )(v_preds_all_seeds)
        mean_rank_acc = float(jnp.mean(rank_acc_seeds))
        internal_mse = float(jnp.mean(jnp.square(v_preds_all_seeds[:, :num_internal] - v_true[None, :num_internal])))

        # ---------------------------------------------------------------------
        # BIAS-VARIANCE DECOMPOSITION:
        # ---------------------------------------------------------------------
        mean_pred = jnp.mean(v_preds_all_seeds, axis=0)       # (num_states,)
        variance_s = jnp.var(v_preds_all_seeds, axis=0)       # (num_states,)
        bias_sq_s = jnp.square(mean_pred - v_true)           # (num_states,)
        mse_s = jnp.mean(jnp.square(v_preds_all_seeds - v_true[None, :]), axis=0)

        mean_bias_sq = float(jnp.mean(bias_sq_s))
        mean_var = float(jnp.mean(variance_s))
        mean_mse = float(jnp.mean(mse_s))

        print(f"  Final Across {n_seeds} Seeds:")
        print(f"    Mean MSE (All):      {mean_mse:.6f}")
        print(f"    Internal Node MSE:   {internal_mse:.6f}")
        print(f"    Bias^2:              {mean_bias_sq:.6f} ({mean_bias_sq / (mean_mse + 1e-9) * 100:.1f}%)")
        print(f"    Variance:            {mean_var:.6f} ({mean_var / (mean_mse + 1e-9) * 100:.1f}%)")
        print(f"    Sibling Rank Acc:    {mean_rank_acc * 100:.1f}%")

        if "interp_batch_mse" in metrics:
            mc_b_mse = float(jnp.mean(metrics["mc_batch_mse"][:, -1]))
            cr_b_mse = float(jnp.mean(metrics["critic_batch_mse"][:, -1]))
            in_b_mse = float(jnp.mean(metrics["interp_batch_mse"][:, -1]))
            alpha_end = float(jnp.mean(metrics["mean_alpha"][:, -1]))
            print(f"    [Variant 2 Shrinkage Audit @ Final Batch]:")
            print(f"      Raw V_MC Batch MSE: {mc_b_mse:.6f} | V_phi Batch MSE: {cr_b_mse:.6f} | V_interp Batch MSE: {in_b_mse:.6f} (alpha={alpha_end:.2f})")

        base_trans_per_update = num_envs * (depth if "FBI" in name else num_steps)
        x_base_transitions = (jnp.arange(1, num_updates + 1) * base_trans_per_update)

        results[name] = {
            "k_val": k_val,
            "x_transitions": np.array(x_transitions),
            "x_base_transitions": np.array(x_base_transitions),
            "value_error": np.array(metrics["value_error"]),         # (n_seeds, num_updates)
            "root_abs_error": np.array(metrics["root_abs_error"]),   # (n_seeds, num_updates)
            "advantage_cosine_sim": np.array(metrics["advantage_cosine_sim"]) if "advantage_cosine_sim" in metrics else None,
            "sibling_ranking_acc": np.array(metrics["sibling_ranking_acc"]) if "sibling_ranking_acc" in metrics else None,
            "v_preds_seeds": np.array(v_preds_all_seeds),
            "mean_pred": np.array(mean_pred),
            "bias_sq": mean_bias_sq,
            "variance": mean_var,
            "mse": mean_mse,
            "internal_mse": internal_mse,
            "rank_acc": mean_rank_acc,
            "bias_sq_s": np.array(bias_sq_s),
            "variance_s": np.array(variance_s),
            "mse_s": np.array(mse_s),
        }

    # Save complete benchmark results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "benchmark_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # =========================================================================
    # PLOT 1: Learning Curves: Total Transitions | Base Transitions | Updates
    # =========================================================================
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    colors = {
        "Standard MC (K=1)": "#d62728",
        "Vine-Target (K=4)": "#1f77b4",
        "Vine-Target (K=8)": "#2ca02c",
        "FBI (K=4)": "#9467bd",
        "FBI (K=4, lam=0.0)": "#9467bd",
        "FBI (K=4, lam=0.5)": "#ff7f0e",
        "FBI (K=4, lam=1.0)": "#8c564b",
    }

    # Compute global y-bounds across all methods so all subplots share a consistent scale
    all_min_y, all_max_y = [], []
    for data in results.values():
        if "value_error" not in data:
            continue
        y_seeds = data["value_error"]
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        all_min_y.append(np.min(mean_y))
        all_max_y.append(np.max(mean_y + sem_y))

    global_min_y = float(min(all_min_y)) if all_min_y else 1e-5
    global_max_y = float(max(all_max_y)) if all_max_y else 1.0
    y_bottom = max(1e-7, global_min_y * 0.5)
    y_top = global_max_y * 2.0

    # Subplot 0: Global Value Error vs. Total Sampled Transitions
    ax0 = axes[0]
    for name, data in results.items():
        if "value_error" not in data:
            continue
        x = data["x_transitions"]
        y_seeds = data["value_error"]
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")

        lower_bound = np.clip(np.maximum(mean_y * 0.5, mean_y - sem_y), y_bottom, None)
        upper_bound = np.clip(mean_y + sem_y, None, y_top)

        ax0.plot(x, mean_y, label=name, color=c, linewidth=2.2)
        ax0.fill_between(x, lower_bound, upper_bound, color=c, alpha=0.18)

    ax0.set_title("Total Environment Transitions\n(Base Rollouts + Auxiliary Lookaheads)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Total Environment Transitions", fontsize=10)
    ax0.set_ylabel("Mean Squared Value Error (MSE vs. $V^*$)", fontsize=10)
    ax0.set_yscale("log")
    ax0.set_ylim(bottom=y_bottom, top=y_top)
    ax0.grid(True, which="both", linestyle="--", alpha=0.5)
    ax0.legend(fontsize=9)

    # Subplot 1: Global Value Error vs. Base Rollout Transitions (Excluding Auxiliary)
    ax1 = axes[1]
    for name, data in results.items():
        if "value_error" not in data:
            continue
        x_base = data["x_base_transitions"]
        y_seeds = data["value_error"]
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")

        lower_bound = np.clip(np.maximum(mean_y * 0.5, mean_y - sem_y), y_bottom, None)
        upper_bound = np.clip(mean_y + sem_y, None, y_top)

        ax1.plot(x_base, mean_y, label=name, color=c, linewidth=2.2)
        ax1.fill_between(x_base, lower_bound, upper_bound, color=c, alpha=0.18)

    ax1.set_title("Base Rollout Transitions\n(On-Policy PPO Data, Excl. Auxiliary)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Base Rollout Transitions (Used to Train PPO)", fontsize=10)
    ax1.set_ylabel("Mean Squared Value Error (MSE vs. $V^*$)", fontsize=10)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=y_bottom, top=y_top)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Subplot 2: Global Value Error vs. Parameter Updates
    ax2 = axes[2]
    for name, data in results.items():
        if "value_error" not in data:
            continue
        y_seeds = data["value_error"]
        mean_y = np.mean(y_seeds, axis=0)
        x_updates = np.arange(1, len(mean_y) + 1)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")

        lower_bound = np.clip(np.maximum(mean_y * 0.5, mean_y - sem_y), y_bottom, None)
        upper_bound = np.clip(mean_y + sem_y, None, y_top)

        ax2.plot(x_updates, mean_y, label=name, color=c, linewidth=2.2)
        ax2.fill_between(x_updates, lower_bound, upper_bound, color=c, alpha=0.18)

    ax2.set_title("Parameter Updates\n(Per-Update Learning Speed)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Gradient Updates / Parameter Steps", fontsize=10)
    ax2.set_ylabel("Mean Squared Value Error (MSE vs. $V^*$)", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=y_bottom, top=y_top)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    lc_path = os.path.join(save_dir, "learning_curves_by_transitions.png")
    plt.savefig(lc_path, dpi=200)
    plt.close()
    print(f"\n3-panel learning curves plot saved to {lc_path}")

    # Dedicated Plot: Global Value Error vs. Base Rollout Transitions
    fig_base, ax_base = plt.subplots(figsize=(7.5, 4.8))
    for name, data in results.items():
        if "value_error" not in data:
            continue
        x_base = data["x_base_transitions"]
        y_seeds = data["value_error"]
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")

        lower_bound = np.clip(np.maximum(mean_y * 0.5, mean_y - sem_y), y_bottom, None)
        upper_bound = np.clip(mean_y + sem_y, None, y_top)

        ax_base.plot(x_base, mean_y, label=name, color=c, linewidth=2.2)
        ax_base.fill_between(x_base, lower_bound, upper_bound, color=c, alpha=0.18)

    ax_base.set_title("Global Value Error vs. Base Rollout Transitions\n(On-Policy PPO Data, Excluding Auxiliary Rollouts)", fontsize=11, fontweight="bold")
    ax_base.set_xlabel("Base Rollout Transitions (Used to Train PPO)", fontsize=10)
    ax_base.set_ylabel("Mean Squared Value Error (MSE vs. $V^*$)", fontsize=10)
    ax_base.set_yscale("log")
    ax_base.set_ylim(bottom=y_bottom, top=y_top)
    ax_base.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_base.legend(fontsize=10)
    plt.tight_layout()
    base_lc_path = os.path.join(save_dir, "learning_curves_by_base_transitions.png")
    plt.savefig(base_lc_path, dpi=200)
    plt.close()
    print(f"Base rollouts learning curve plot saved to {base_lc_path}")

    # =========================================================================
    # PLOT: Advantage Cosine Similarity Learning Curves (3 Panels)
    # =========================================================================
    fig_adv, axes_adv = plt.subplots(1, 3, figsize=(19, 5.2))

    # Subplot 0: Cosine Similarity vs. Total Transitions
    ax0 = axes_adv[0]
    for name, data in results.items():
        if data.get("advantage_cosine_sim") is None:
            continue
        x = data["x_transitions"]
        y_seeds = data["advantage_cosine_sim"]
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")
        ax0.plot(x, mean_y, label=name, color=c, linewidth=2.2)
        ax0.fill_between(x, mean_y - sem_y, np.minimum(1.0, mean_y + sem_y), color=c, alpha=0.18)

    ax0.set_title("Total Environment Transitions\n(Base Rollouts + Auxiliary Lookaheads)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Total Environment Transitions", fontsize=10)
    ax0.set_ylabel("Advantage Cosine Similarity ($\cos(\hat{A}, A^*)$)", fontsize=10)
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(fontsize=9, loc="lower right")

    # Subplot 1: Cosine Similarity vs. Base Rollout Transitions
    ax1 = axes_adv[1]
    for name, data in results.items():
        if data.get("advantage_cosine_sim") is None:
            continue
        x_base = data["x_base_transitions"]
        y_seeds = data["advantage_cosine_sim"]
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")
        ax1.plot(x_base, mean_y, label=name, color=c, linewidth=2.2)
        ax1.fill_between(x_base, mean_y - sem_y, np.minimum(1.0, mean_y + sem_y), color=c, alpha=0.18)

    ax1.set_title("Base Rollout Transitions\n(On-Policy PPO Data, Excl. Auxiliary)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Base Rollout Transitions", fontsize=10)
    ax1.set_ylabel("Advantage Cosine Similarity ($\cos(\hat{A}, A^*)$)", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="lower right")

    # Subplot 2: Cosine Similarity vs. Parameter Updates
    ax2 = axes_adv[2]
    for name, data in results.items():
        if data.get("advantage_cosine_sim") is None:
            continue
        y_seeds = data["advantage_cosine_sim"]
        mean_y = np.mean(y_seeds, axis=0)
        x_updates = np.arange(1, len(mean_y) + 1)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")
        ax2.plot(x_updates, mean_y, label=name, color=c, linewidth=2.2)
        ax2.fill_between(x_updates, mean_y - sem_y, np.minimum(1.0, mean_y + sem_y), color=c, alpha=0.18)

    ax2.set_title("Parameter Updates\n(Per-Update Learning Speed)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Gradient Updates / Parameter Steps", fontsize=10)
    ax2.set_ylabel("Advantage Cosine Similarity ($\cos(\hat{A}, A^*)$)", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    adv_lc_path = os.path.join(save_dir, "learning_curves_advantage_alignment.png")
    plt.savefig(adv_lc_path, dpi=200)
    plt.close()
    print(f"Advantage alignment learning curve plot saved to {adv_lc_path}")

    # =========================================================================
    # PLOT: Sibling Ranking Accuracy Learning Curves (3 Panels)
    # =========================================================================
    fig_rank, axes_rank = plt.subplots(1, 3, figsize=(19, 5.2))

    # Subplot 0: Ranking Accuracy vs. Total Transitions
    ax0 = axes_rank[0]
    for name, data in results.items():
        if data.get("sibling_ranking_acc") is None:
            continue
        x = data["x_transitions"]
        y_seeds = data["sibling_ranking_acc"] * 100.0
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")
        ax0.plot(x, mean_y, label=name, color=c, linewidth=2.2)
        ax0.fill_between(x, np.maximum(0.0, mean_y - sem_y), np.minimum(100.0, mean_y + sem_y), color=c, alpha=0.18)

    ax0.set_title("Total Environment Transitions\n(Base Rollouts + Auxiliary Lookaheads)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Total Environment Transitions", fontsize=10)
    ax0.set_ylabel("Sibling Advantage Ranking Accuracy (%)", fontsize=10)
    ax0.set_ylim(45, 105)
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(fontsize=9, loc="lower right")

    # Subplot 1: Ranking Accuracy vs. Base Rollout Transitions
    ax1 = axes_rank[1]
    for name, data in results.items():
        if data.get("sibling_ranking_acc") is None:
            continue
        x_base = data["x_base_transitions"]
        y_seeds = data["sibling_ranking_acc"] * 100.0
        mean_y = np.mean(y_seeds, axis=0)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")
        ax1.plot(x_base, mean_y, label=name, color=c, linewidth=2.2)
        ax1.fill_between(x_base, np.maximum(0.0, mean_y - sem_y), np.minimum(100.0, mean_y + sem_y), color=c, alpha=0.18)

    ax1.set_title("Base Rollout Transitions\n(On-Policy PPO Data, Excl. Auxiliary)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Base Rollout Transitions", fontsize=10)
    ax1.set_ylabel("Sibling Advantage Ranking Accuracy (%)", fontsize=10)
    ax1.set_ylim(45, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="lower right")

    # Subplot 2: Ranking Accuracy vs. Parameter Updates
    ax2 = axes_rank[2]
    for name, data in results.items():
        if data.get("sibling_ranking_acc") is None:
            continue
        y_seeds = data["sibling_ranking_acc"] * 100.0
        mean_y = np.mean(y_seeds, axis=0)
        x_updates = np.arange(1, len(mean_y) + 1)
        sem_y = np.std(y_seeds, axis=0) / np.sqrt(y_seeds.shape[0])
        c = colors.get(name, "#333333")
        ax2.plot(x_updates, mean_y, label=name, color=c, linewidth=2.2)
        ax2.fill_between(x_updates, np.maximum(0.0, mean_y - sem_y), np.minimum(100.0, mean_y + sem_y), color=c, alpha=0.18)

    ax2.set_title("Parameter Updates\n(Per-Update Learning Speed)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Gradient Updates / Parameter Steps", fontsize=10)
    ax2.set_ylabel("Sibling Advantage Ranking Accuracy (%)", fontsize=10)
    ax2.set_ylim(45, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    rank_lc_path = os.path.join(save_dir, "learning_curves_ranking_accuracy.png")
    plt.savefig(rank_lc_path, dpi=200)
    plt.close()
    print(f"Sibling ranking accuracy learning curve plot saved to {rank_lc_path}")

    # =========================================================================
    # PLOT 2 & BIAS-VARIANCE DECOMPOSITION ACROSS SEEDS:
    # =========================================================================
    results = bias_variance_decomposition_across_seeds(
        results=results,
        evaluator=evaluator,
        save_dir=save_dir,
        n_seeds=n_seeds,
        seed=seed,
        colors=colors,
    )

    # Re-save complete benchmark results including Vine MC
    with open(os.path.join(save_dir, "benchmark_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results


def bias_variance_decomposition_across_seeds(
    results: Dict[str, Any],
    evaluator: TreeMRPExactValue,
    save_dir: str = "results/figures",
    n_seeds: int = 8,
    seed: int = 42,
    colors: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Computes exact empirical bias-variance decomposition across seeds for all value estimation methods,
    including non-parametric Vine MC (K=16) where Bias^2 ≈ 0 and MSE is entirely sampling variance.
    Generates a 3-panel figure:
      1. Absolute Error Decomposition (All Methods including Vine MC K=16)
      2. Zoom-in on Neural Function Approximators (10^-5 scale)
      3. Estimator Variance by State (Log Scale)
    Prints an empirical summary table.
    """
    if colors is None:
        colors = {
            "Standard MC (K=1)": "#d62728",
            "Vine-Target (K=4)": "#1f77b4",
            "Vine-Target (K=8)": "#2ca02c",
            "FBI (K=4)": "#9467bd",
            "FBI (K=4, lam=0.0)": "#9467bd",
            "FBI (K=4, lam=0.5)": "#ff7f0e",
            "FBI (K=4, lam=1.0)": "#8c564b",
            "Vine MC (K=16)": "#2ca02c",
        }

    depth = evaluator.depth
    cluster_size = len(evaluator.correct_leaf_indices)
    env = TreeMRP(depth=evaluator.depth, p=evaluator.p, cluster_size=cluster_size)
    env_params = TabularParams(max_steps_in_episode=evaluator.depth + 2)
    v_true = evaluator.solve_backward_induction()[:evaluator.num_states]
    num_states = evaluator.num_states
    num_internal = evaluator.leaf_offset

    # -------------------------------------------------------------------------
    # Non-Parametric Online Estimator: Vine MC (K=16)
    # -------------------------------------------------------------------------
    if "Vine MC (K=16)" not in results:
        print(f"\n--- Method: Vine MC (K=16) (Non-Parametric Online Monte Carlo) ---")
        max_steps = depth + 2
        rng = jax.random.PRNGKey(seed + 999)
        seed_keys = jax.random.split(rng, n_seeds)
        vine_fn = jax.jit(jax.vmap(lambda r: estimate_vine_all_states(env, env_params, 16, r, max_steps)[0]))
        v_preds_vine = np.array(vine_fn(seed_keys))  # (n_seeds, num_states)

        rank_acc_seeds_vine = jax.vmap(
            lambda preds: fbi_value.compute_sibling_ranking_accuracy(preds, v_true, num_internal)
        )(v_preds_vine)
        mean_rank_acc_vine = float(jnp.mean(rank_acc_seeds_vine))
        adv_cos_seeds_vine = jax.vmap(
            lambda preds: fbi_value.compute_advantage_cosine_similarity(preds, v_true, num_internal)
        )(v_preds_vine)
        mean_adv_cos_vine = float(jnp.mean(adv_cos_seeds_vine))
        internal_mse_vine = float(jnp.mean(jnp.square(v_preds_vine[:, :num_internal] - v_true[None, :num_internal])))

        mean_pred_vine = jnp.mean(v_preds_vine, axis=0)
        variance_s_vine = jnp.var(v_preds_vine, axis=0)
        bias_sq_s_vine = jnp.square(mean_pred_vine - v_true)
        mse_s_vine = jnp.mean(jnp.square(v_preds_vine - v_true[None, :]), axis=0)

        mean_bias_sq_vine = float(jnp.mean(bias_sq_s_vine))
        mean_var_vine = float(jnp.mean(variance_s_vine))
        mean_mse_vine = float(jnp.mean(mse_s_vine))

        print(f"  Vine MC (K=16) Across {n_seeds} Seeds:")
        print(f"    Mean MSE (All):      {mean_mse_vine:.6f}")
        print(f"    Internal Node MSE:   {internal_mse_vine:.6f}")
        print(f"    Bias^2:              {mean_bias_sq_vine:.6f} ({mean_bias_sq_vine / (mean_mse_vine + 1e-9) * 100:.1f}%)")
        print(f"    Variance:            {mean_var_vine:.6f} ({mean_var_vine / (mean_mse_vine + 1e-9) * 100:.1f}%)")
        print(f"    Sibling Rank Acc:    {mean_rank_acc_vine * 100:.1f}%")
        print(f"    Adv Cosine Sim:      {mean_adv_cos_vine:.4f}")

        results["Vine MC (K=16)"] = {
            "k_val": 16,
            "v_preds_seeds": v_preds_vine,
            "mean_pred": np.array(mean_pred_vine),
            "bias_sq": mean_bias_sq_vine,
            "variance": mean_var_vine,
            "mse": mean_mse_vine,
            "internal_mse": internal_mse_vine,
            "rank_acc": mean_rank_acc_vine,
            "advantage_cosine_sim": np.repeat(adv_cos_seeds_vine[:, None], 1, axis=1),
            "bias_sq_s": np.array(bias_sq_s_vine),
            "variance_s": np.array(variance_s_vine),
            "mse_s": np.array(mse_s_vine),
        }

    # =========================================================================
    # PLOT 2: Direct Bias-Variance Tradeoff Decomposition (3 Panels)
    # =========================================================================
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # Panel 0: Absolute Error Decomposition (All Methods including Vine MC K=16)
    ax_all = axes[0]
    all_names = list(results.keys())
    bias_sq_all = [results[n]["bias_sq"] for n in all_names]
    var_all = [results[n]["variance"] for n in all_names]
    x_pos_all = np.arange(len(all_names))
    width = 0.55

    ax_all.bar(x_pos_all, bias_sq_all, width, label="$\mathrm{Bias}^2$ (Approximation Error)", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax_all.bar(x_pos_all, var_all, width, bottom=bias_sq_all, label="$\mathrm{Variance}$ (Sampling Fluctuations)", color="#1f77b4", alpha=0.85, edgecolor="black")

    ax_all.set_xticks(x_pos_all)
    ax_all.set_xticklabels(all_names, fontsize=10, rotation=15, ha="right")
    ax_all.set_ylabel("Total Error Decomposition ($\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Variance}$)", fontsize=10)
    ax_all.set_title("Absolute Error Decomposition\n(All Methods: Online MC vs. Neural Critics)", fontsize=11, fontweight="bold")
    ax_all.legend(fontsize=9, loc="upper right")
    ax_all.grid(True, axis="y", linestyle="--", alpha=0.5)

    max_tot_all = max(b + v for b, v in zip(bias_sq_all, var_all))
    ax_all.set_ylim(bottom=0, top=max_tot_all * 1.22)

    for i, (b_sq, v) in enumerate(zip(bias_sq_all, var_all)):
        tot = b_sq + v
        pct_v = v / (tot + 1e-9) * 100
        ax_all.text(i, tot + max_tot_all * 0.02, f"MSE: {tot:.5f}\n({pct_v:.0f}% Var)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 1: Zoom-in on Neural Function Approximators (10^-5 scale)
    ax_nn = axes[1]
    nn_names = [n for n in all_names if n != "Vine MC (K=16)"]
    bias_sq_nn = [results[n]["bias_sq"] for n in nn_names]
    var_nn = [results[n]["variance"] for n in nn_names]
    x_pos_nn = np.arange(len(nn_names))

    ax_nn.bar(x_pos_nn, bias_sq_nn, width, label="$\mathrm{Bias}^2$", color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax_nn.bar(x_pos_nn, var_nn, width, bottom=bias_sq_nn, label="$\mathrm{Variance}$", color="#1f77b4", alpha=0.85, edgecolor="black")

    ax_nn.set_xticks(x_pos_nn)
    ax_nn.set_xticklabels(nn_names, fontsize=10, rotation=15, ha="right")
    ax_nn.set_ylabel("Total Error Decomposition ($\mathrm{MSE}$)", fontsize=10)
    ax_nn.set_title("Zoom-in: Neural Critics ($10^{-5}$ Scale)\n(Function Approximation Benefit)", fontsize=11, fontweight="bold")
    ax_nn.legend(fontsize=9, loc="upper right")
    ax_nn.grid(True, axis="y", linestyle="--", alpha=0.5)

    max_tot_nn = max(b + v for b, v in zip(bias_sq_nn, var_nn))
    ax_nn.set_ylim(bottom=0, top=max_tot_nn * 1.25)

    for i, (b_sq, v) in enumerate(zip(bias_sq_nn, var_nn)):
        tot = b_sq + v
        pct_v = v / (tot + 1e-9) * 100
        ax_nn.text(i, tot + max_tot_nn * 0.02, f"MSE: {tot:.6f}\n({pct_v:.0f}% Var)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel 2: Estimator Variance by State Across Seeds
    ax_v = axes[2]
    for name in all_names:
        c = colors.get(name, "#333333")
        var_s = results[name]["variance_s"]
        ls = "--" if "Vine MC" in name else "-"
        ax_v.plot(range(num_states), np.maximum(var_s, 1e-9), marker="o", markersize=3.5, label=name, color=c, alpha=0.85, linestyle=ls)

    ax_v.set_xlabel(f"State Index (0=Root, 1..{num_internal-1}=Internal, {num_internal}..{num_states-1}=Leaves)", fontsize=10)
    ax_v.set_ylabel("Estimator Variance $\mathrm{Var}(\hat{V}(s))$ Across Seeds (Log Scale)", fontsize=10)
    ax_v.set_yscale("log")
    ax_v.set_title("Estimator Variance by State\n(Log-Scale: Sensitivity to Sample Noise)", fontsize=11, fontweight="bold")
    ax_v.legend(fontsize=8.5, loc="lower left")
    ax_v.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    bv_path = os.path.join(save_dir, "bias_variance_decomposition.png")
    plt.savefig(bv_path, dpi=200)
    plt.close()
    print(f"Bias-variance decomposition plot saved to {bv_path}")

    # =========================================================================
    # Print Final Summary Table
    # =========================================================================
    print("\n" + "=" * 140)
    print("EMPIRICAL BIAS-VARIANCE, ADVANTAGE ALIGNMENT & RANKING SUMMARY TABLE")
    print("=" * 140)
    tbl_header = (
        f"{'Method':<22} | {'Transitions':<14} | {'Total MSE':<11} | "
        f"{'Internal MSE':<12} | {'Bias^2':<11} | {'Variance':<11} | {'Var %':<7} | {'Rank Acc':<8} | {'Adv Cos Sim':<11}"
    )
    print(tbl_header)
    print("-" * len(tbl_header))
    for name in all_names:
        res = results[name]
        tot = res["mse"]
        if "internal_mse" in res:
            int_mse = res["internal_mse"]
        elif "v_preds_seeds" in res:
            int_mse = float(jnp.mean(jnp.square(res["v_preds_seeds"][:, :num_internal] - v_true[None, :num_internal])))
        else:
            int_mse = tot
        b_sq = res["bias_sq"]
        v = res["variance"]
        pct = (v / (tot + 1e-9)) * 100
        r_acc = res.get("rank_acc", 1.0) * 100
        adv_cos_str = f"{float(np.mean(res['advantage_cosine_sim'][:, -1])):.4f}" if res.get("advantage_cosine_sim") is not None else "N/A"
        n_trans_str = f"{int(res['x_transitions'][-1]):d}" if "x_transitions" in res else "N/A (online)"
        print(
            f"{name:<22} | {n_trans_str:<14} | {tot:<11.6f} | "
            f"{int_mse:<12.6f} | {b_sq:<11.6f} | {v:<11.6f} | {pct:<6.1f}% | {r_acc:<7.1f}% | {adv_cos_str:<11}"
        )
    print("=" * 140)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--cluster-size", type=int, default=2)
    parser.add_argument("--transitions", type=int, default=120000)
    parser.add_argument("--no-fbi", action="store_true", help="Exclude FBI from experiment")
    parser.add_argument("--fbi-lambdas", type=float, nargs="+", default=[0.0], help="Lambda values for FBI(lambda)")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/figures")
    parser.add_argument("--use-cached", action="store_true", help="Use cached benchmark_results.pkl if available")
    args = parser.parse_args()

    cache_path = os.path.join(args.out_dir, "benchmark_results.pkl")
    if args.use_cached and os.path.exists(cache_path):
        print(f"Loading cached benchmark results from {cache_path}...")
        with open(cache_path, "rb") as f:
            cached_results = pickle.load(f)
        evaluator = TreeMRPExactValue(depth=args.depth, p=args.p, cluster_size=args.cluster_size)
        bias_variance_decomposition_across_seeds(
            results=cached_results,
            evaluator=evaluator,
            save_dir=args.out_dir,
            n_seeds=args.seeds,
            seed=args.seed,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(cached_results, f)
    else:
        run_experiment_with_transitions(
            depth=args.depth,
            p=args.p,
            cluster_size=args.cluster_size,
            k_vine_list=(4,),
            include_fbi=not args.no_fbi,
            fbi_lambdas=tuple(args.fbi_lambdas),
            total_transitions_budget=args.transitions,
            n_seeds=args.seeds,
            seed=args.seed,
            save_dir=args.out_dir,
        )
