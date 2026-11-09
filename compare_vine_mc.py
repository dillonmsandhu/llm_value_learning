from __future__ import annotations
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any

from envs.tree_mrp import TreeMRPExactValue, TreeMRP
from envs.tabular_matrix_env import TabularParams
from transformer import TransformerCritic
from vine import estimate_vine_all_states
import mc_value
import vine_target_value


def generate_figure3_full_comparison(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    k_list: Tuple[int, ...] = (1, 4, 16, 64),
    timesteps: int = 48000,
    seed: int = 42,
    save_path: str = "results/figures/figure3_vine_comparison.png",
):
    """
    Generates the comprehensive Figure 3 comparison:
      - Row 1: Vine Online Estimators (sweeping K) + Summary MSE bar chart
      - Row 2: Vine-Target Value Learning (sweeping K) + Standard MC Value Learning
      
    For each scatter plot:
      X-axis: Ground truth backward induction value V*(s)
      Y-axis: Estimated / predicted value V_hat(s)
      Dots:   All active states in the tree, color-coded by tree depth
    """
    print(f"==========================================================================")
    print(f"COMPREHENSIVE VALUE ESTIMATION BENCHMARK (Tree MRP Depth={depth}, p={p})")
    print(f"Comparing: 1) Vine Estimators, 2) Vine-Target Critic, 3) Standard MC Critic")
    print(f"==========================================================================\n")

    evaluator = TreeMRPExactValue(depth=depth, p=p, cluster_size=cluster_size)
    env = TreeMRP(depth=depth, p=p, cluster_size=cluster_size)
    env_params = TabularParams(max_steps_in_episode=depth + 2)
    v_true = evaluator.solve_backward_induction()[:evaluator.num_states]
    num_states = evaluator.num_states
    max_steps = depth + 1

    # State depths for coloring (0 = root, ..., depth = leaves)
    state_depths = jnp.array([len(evaluator.state_strings[s]) for s in range(num_states)])
    all_obs = evaluator.obs_stack[:num_states]

    rng = jax.random.PRNGKey(seed)
    rng, vine_rng, mc_rng, vt_rng = jax.random.split(rng, 4)

    # -------------------------------------------------------------------------
    # 1. Evaluate Vine Online Estimators across K (Row 1)
    # -------------------------------------------------------------------------
    print("--- 1. Evaluating Vine Online Estimators (No Training) ---")
    vine_estimates = {}
    vine_mses = {}
    for k in k_list:
        vine_rng, k_rng = jax.random.split(vine_rng)
        v_vine_fn = jax.jit(lambda r: estimate_vine_all_states(env, env_params, k, r, max_steps))
        v_k, _ = v_vine_fn(k_rng)
        vine_estimates[k] = v_k
        vine_mses[k] = float(jnp.mean(jnp.square(v_k - v_true)))
        print(f"  Vine (K={k:2d}): MSE = {vine_mses[k]:.6f} | Root Est = {float(v_k[0]):.4f}")

    # -------------------------------------------------------------------------
    # 2. Train Standard MC Value Learning Critic (PPO-style root rollouts)
    # -------------------------------------------------------------------------
    print(f"\n--- 2. Training Standard MC Critic ({timesteps} timesteps) ---")
    mc_cfg = mc_value.default_config.copy()
    mc_cfg.update({
        "DEPTH": depth,
        "P_SUCCESS": p,
        "CLUSTER_SIZE": cluster_size,
        "TOTAL_TIMESTEPS": timesteps,
        "SEED": seed,
    })
    mc_train_fn = jax.jit(mc_value.make_train(mc_cfg))
    mc_out = mc_train_fn(mc_rng)
    mc_trained_params = mc_out["runner_state"][0].params

    mc_model = TransformerCritic(
        d_model=mc_cfg["D_MODEL"],
        n_heads=mc_cfg["N_HEADS"],
        n_layers=mc_cfg["N_LAYERS"],
        final_hidden_dim=mc_cfg["FINAL_HIDDEN_DIM"],
        max_len=depth + 4,
    )
    v_mc_critic = mc_model.apply(mc_trained_params, all_obs, method=mc_model.value)
    mc_mse = float(jnp.mean(jnp.square(v_mc_critic - v_true)))
    print(f"  Standard MC Critic: MSE = {mc_mse:.6f} | Root Pred = {float(v_mc_critic[0]):.4f}")

    # -------------------------------------------------------------------------
    # 3. Train Vine-Target Value Learning Critics across K (Row 2)
    # -------------------------------------------------------------------------
    print(f"\n--- 3. Training Vine-Target Critics across K ({timesteps} timesteps each) ---")
    vine_target_estimates = {}
    vine_target_mses = {}

    for k in k_list:
        vt_rng, k_rng = jax.random.split(vt_rng)
        vt_cfg = vine_target_value.default_config.copy()
        vt_cfg.update({
            "DEPTH": depth,
            "P_SUCCESS": p,
            "CLUSTER_SIZE": cluster_size,
            "K_VINE": k,
            "TOTAL_TIMESTEPS": timesteps,
            "SEED": seed,
        })
        vt_train_fn = jax.jit(vine_target_value.make_train(vt_cfg))
        vt_out = vt_train_fn(k_rng)
        vt_params = vt_out["runner_state"][0].params

        vt_model = TransformerCritic(
            d_model=vt_cfg["D_MODEL"],
            n_heads=vt_cfg["N_HEADS"],
            n_layers=vt_cfg["N_LAYERS"],
            final_hidden_dim=vt_cfg["FINAL_HIDDEN_DIM"],
            max_len=depth + 4,
        )
        v_vt = vt_model.apply(vt_params, all_obs, method=vt_model.value)
        vine_target_estimates[k] = v_vt
        vine_target_mses[k] = float(jnp.mean(jnp.square(v_vt - v_true)))
        print(f"  Vine-Target Critic (K={k:2d}): MSE = {vine_target_mses[k]:.6f} | Root Pred = {float(v_vt[0]):.4f}")

    # -------------------------------------------------------------------------
    # 4. Print Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"FIGURE 3 COMPLETE COMPARISON (Ground Truth V*(root) = {float(v_true[0]):.4f})")
    print("=" * 92)
    header = f"{'Method':<28} | {'MSE vs V*':<12} | {'Max Abs Err':<12} | {'Mean Bias':<12} | {'V(root)':<10}"
    print(header)
    print("-" * len(header))

    for k in k_list:
        v_est = vine_estimates[k]
        mse = float(jnp.mean(jnp.square(v_est - v_true)))
        max_err = float(jnp.max(jnp.abs(v_est - v_true)))
        bias = float(jnp.mean(v_est - v_true))
        print(f"{f'Vine Estimator (K={k})':<28} | {mse:<12.6f} | {max_err:<12.4f} | {bias:<12.6f} | {float(v_est[0]):<10.4f}")

    print("-" * len(header))
    for k in k_list:
        v_est = vine_target_estimates[k]
        mse = float(jnp.mean(jnp.square(v_est - v_true)))
        max_err = float(jnp.max(jnp.abs(v_est - v_true)))
        bias = float(jnp.mean(v_est - v_true))
        print(f"{f'Vine-Target Critic (K={k})':<28} | {mse:<12.6f} | {max_err:<12.4f} | {bias:<12.6f} | {float(v_est[0]):<10.4f}")

    print("-" * len(header))
    mse_mc = float(jnp.mean(jnp.square(v_mc_critic - v_true)))
    max_err_mc = float(jnp.max(jnp.abs(v_mc_critic - v_true)))
    bias_mc = float(jnp.mean(v_mc_critic - v_true))
    print(f"{'Standard MC Critic (PPO)':<28} | {mse_mc:<12.6f} | {max_err_mc:<12.4f} | {bias_mc:<12.6f} | {float(v_mc_critic[0]):<10.4f}")
    print("=" * 92)

    # -------------------------------------------------------------------------
    # 5. Render 2-Row Comparison Figure
    # -------------------------------------------------------------------------
    # 2 rows x 5 columns:
    # Row 1: Vine (K=1), Vine (K=4), Vine (K=16), Vine (K=64), MSE Bar Chart Summary
    # Row 2: Vine-Target (K=1), Vine-Target (K=4), Vine-Target (K=16), Vine-Target (K=64), Standard MC Critic
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))

    scatter_handle = None

    # ROW 1: Vine Online Estimators
    for col, k in enumerate(k_list):
        ax = axes[0, col]
        v_est = vine_estimates[k]
        mse = vine_mses[k]

        scatter = ax.scatter(
            v_true,
            v_est,
            c=state_depths,
            cmap="viridis",
            s=40,
            alpha=0.85,
            edgecolors="none",
        )
        scatter_handle = scatter
        ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5, alpha=0.8)
        ax.set_title(f"Vine Online (K = {k})\nMSE = {mse:.5f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("True Value $V^*(s)$", fontsize=10)
        ax.set_ylabel("Vine Estimate $\\hat{V}(s)$", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    # ROW 1, COL 5: MSE Summary Bar Chart
    ax_bar = axes[0, 4]
    labels = [f"V(K={k})" for k in k_list] + [f"VT(K={k})" for k in k_list] + ["MC"]
    all_mses = [vine_mses[k] for k in k_list] + [vine_target_mses[k] for k in k_list] + [mc_mse]
    colors = ["#1f77b4"] * 4 + ["#2ca02c"] * 4 + ["#d62728"]
    bars = ax_bar.bar(range(len(labels)), all_mses, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_bar.set_ylabel("MSE vs Ground Truth $V^*$", fontsize=10)
    ax_bar.set_title("Overall Value MSE Summary\n(Lower is Better)", fontsize=11, fontweight="bold")
    ax_bar.set_yscale("log")
    ax_bar.grid(True, which="both", linestyle="--", alpha=0.5)

    # ROW 2: Vine-Target Value Learning (sweeping K)
    for col, k in enumerate(k_list):
        ax = axes[1, col]
        v_est = vine_target_estimates[k]
        mse = vine_target_mses[k]

        ax.scatter(
            v_true,
            v_est,
            c=state_depths,
            cmap="viridis",
            s=40,
            alpha=0.85,
            edgecolors="none",
        )
        ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5, alpha=0.8)
        ax.set_title(f"Vine-Target Critic (K = {k})\nMSE = {mse:.6f}", fontsize=11, fontweight="bold", color="#1b7837")
        ax.set_xlabel("True Value $V^*(s)$", fontsize=10)
        ax.set_ylabel("Learned Critic $V_\\theta(s)$", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    # ROW 2, COL 5: Standard MC Value Learning Critic
    ax_mc = axes[1, 4]
    ax_mc.scatter(
        v_true,
        v_mc_critic,
        c=state_depths,
        cmap="viridis",
        s=40,
        alpha=0.85,
        edgecolors="none",
    )
    ax_mc.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5, alpha=0.8, label="Ideal (y=x)")
    ax_mc.set_title(f"Standard MC Critic (PPO)\nMSE = {mc_mse:.6f}", fontsize=11, fontweight="bold", color="#b2182b")
    ax_mc.set_xlabel("True Value $V^*(s)$", fontsize=10)
    ax_mc.set_ylabel("Learned Critic $V_\\theta(s)$", fontsize=10)
    ax_mc.set_xlim(-0.05, 1.05)
    ax_mc.set_ylim(-0.05, 1.05)
    ax_mc.grid(True, linestyle="--", alpha=0.5)

    # Add overarching Row labels
    fig.text(0.01, 0.72, "Row 1: Vine Online Estimators (No Training)", fontsize=13, fontweight="bold", rotation=90, va="center")
    fig.text(0.01, 0.28, "Row 2: Trained Transformer Value Critics", fontsize=13, fontweight="bold", rotation=90, va="center")

    # Add horizontal colorbar for state depth
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
    cbar = fig.colorbar(scatter_handle, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Tree Depth of State (0 = Root, 4 = Leaves)", fontsize=11)

    plt.tight_layout(rect=[0.02, 0.06, 1.0, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Full Figure 3 comparison plot saved to {save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--cluster-size", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=48000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/figures/figure3_vine_comparison.png")
    args = parser.parse_args()

    generate_figure3_full_comparison(
        depth=args.depth,
        p=args.p,
        cluster_size=args.cluster_size,
        k_list=(1, 4, 16, 64),
        timesteps=args.timesteps,
        seed=args.seed,
        save_path=args.out,
    )
