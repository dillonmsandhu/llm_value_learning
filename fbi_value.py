from __future__ import annotations
import os
import time
import pickle
import argparse
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import numpy as np

from envs.tree_mrp import TreeMRPExactValue, TreeMRP
from envs.tabular_matrix_env import TabularParams, TabularState
from transformer import TransformerCritic

SAVE_DIR = "fbi_value"

default_config = {
    "N_SEEDS": 1,
    "ENV_NAME": "TreeMRP",
    "DEPTH": 4,
    "P_SUCCESS": 0.8,
    "CLUSTER_SIZE": 2,
    "GAMMA": 1.0,
    "K_BRANCHES": 4,
    "LMBDA": 0.0,          # FBI(lambda): 0.0 = pure 1-step FBI, 1.0 = penultimate-anchored MC, (0,1) = hybrid
    "STAGE_STEPS": 4,      # Inner gradient steps per backward depth stage so V_phi fits t+1 before t queries it
    "MINIBATCH_SIZE": 64,  # Active-pool minibatch size per stage gradient step (avoids masked full-batch wasted FLOPs)
    "NUM_ENVS": 64,
    "TOTAL_TIMESTEPS": 120000,
    "LR": 1e-3,
    "WARMUP_RATIO": 0.1,
    "MAX_GRAD_NORM": 1.0,
    "WEIGHT_DECAY": 0.01,
    "D_MODEL": 64,
    "N_HEADS": 4,
    "N_LAYERS": 2,
    "FINAL_HIDDEN_DIM": 32,
}


def compute_sample_costs(num_envs: int, depth: int, k_branches: int) -> int:
    """
    Computes total environment transitions per FBI update step:
      1. Base trajectories: num_envs * depth steps from root to leaf
      2. 1-Step branching lookaheads: num_envs * depth * k_branches steps
    Total = num_envs * depth * (1 + k_branches).
    """
    base_cost = num_envs * depth
    branch_cost = num_envs * depth * k_branches
    return base_cost + branch_cost


def compute_explained_variance(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Explained variance EV = 1 - Var(y_true - y_pred) / Var(y_true)."""
    var_y = jnp.var(y_true)
    var_diff = jnp.var(y_true - y_pred)
    return jnp.where(var_y > 1e-8, 1.0 - (var_diff / var_y), 0.0)


def compute_sibling_ranking_accuracy(all_preds: jnp.ndarray, v_true_all: jnp.ndarray, num_internal: int) -> jnp.ndarray:
    """
    Computes the fraction of internal parent nodes (0 .. num_internal - 1) with V*(left) != V*(right)
    where the critic correctly orders the two sibling children:
      sign(V_phi(right) - V_phi(left)) == sign(V*(right) - V*(left)).
    Directly measures step-level PPO advantage sign accuracy across all decision branches.
    """
    parents = jnp.arange(num_internal, dtype=jnp.int32)
    left_children = 2 * parents + 1
    right_children = 2 * parents + 2

    true_diff = v_true_all[right_children] - v_true_all[left_children]
    pred_diff = all_preds[right_children] - all_preds[left_children]

    non_tied_mask = jnp.abs(true_diff) > 1e-6
    correct_order = (true_diff * pred_diff) > 0.0
    return jnp.sum(correct_order * non_tied_mask) / jnp.maximum(1.0, jnp.sum(non_tied_mask))


def compute_advantage_cosine_similarity(
    all_preds: jnp.ndarray,
    v_true_all: jnp.ndarray,
    num_internal: int,
    eps: float = 1e-8,
) -> jnp.ndarray:
    parents = jnp.arange(num_internal, dtype=jnp.int32)
    left_children = 2 * parents + 1
    right_children = 2 * parents + 2

    true_v_parent = v_true_all[parents]
    pred_v_parent = all_preds[parents]

    true_adv_left = v_true_all[left_children] - true_v_parent
    true_adv_right = v_true_all[right_children] - true_v_parent
    true_adv = jnp.concatenate([true_adv_left, true_adv_right], axis=0)

    pred_adv_left = all_preds[left_children] - pred_v_parent
    pred_adv_right = all_preds[right_children] - pred_v_parent
    pred_adv = jnp.concatenate([pred_adv_left, pred_adv_right], axis=0)

    dot = jnp.sum(true_adv * pred_adv)
    norm_true = jnp.linalg.norm(true_adv)
    norm_pred = jnp.linalg.norm(pred_adv)

    return dot / (norm_true * norm_pred + eps)


def make_train(config: dict):
    """
    Factory creating a 100% JIT-compilable pure-JAX training function that trains
    the Transformer Critic using 1-Step Fitted Backward Induction with trace parameter lambda (FBI(lambda)).

    Algorithm:
      1. Generate B base trajectories from root s_0 to leaf s_D.
      2. At penultimate level (t = D - 1): sample K completions to termination (leaves)
         giving unbiased Monte Carlo anchor targets: y(s_{D-1}) = (1/K) sum R^{(k)}.
      3. At earlier levels (t < D - 1): sample K single-step continuations s_{t+1}^{(1..K)}.
      4. Interleaved Backward Fitting with Replay Anchoring:
         - Fit critic on leaf ground truths (both base leaves and penultimate branched leaves)
           and penultimate anchors in D_fit.
         - Moving backward t = D - 2 down to 0:
             V_branch(s_t) = (1/K) sum_{k=1}^K sg(V_phi(s_{t+1}^{(k)}))
             y(s_t) = (1 - lambda) * V_branch(s_t) + lambda * gamma * y(s_{t+1})
             D_fit <- D_fit U {(s_t, y(s_t))}
             Update phi on D_fit for `stage_steps` gradient steps.
    """
    depth = int(config.get("DEPTH", 4))
    p_success = float(config.get("P_SUCCESS", 0.8))
    cluster_size = int(config.get("CLUSTER_SIZE", 2))
    gamma = float(config.get("GAMMA", 1.0))
    k_branches = int(config.get("K_BRANCHES", 4))
    lmbda = float(config.get("LMBDA", 0.0))
    stage_steps = int(config.get("STAGE_STEPS", 4))
    minibatch_size = int(config.get("MINIBATCH_SIZE", 64))
    num_envs = int(config["NUM_ENVS"])

    # 1. Environment & Evaluator
    evaluator = TreeMRPExactValue(depth=depth, p=p_success, cluster_size=cluster_size, gamma=gamma)
    env = TreeMRP(depth=depth, p=p_success, cluster_size=cluster_size, gamma=gamma)
    env_params = TabularParams(max_steps_in_episode=depth + 2)

    v_true_all = evaluator.solve_backward_induction()[:evaluator.num_states]
    all_state_obs = evaluator.obs_stack[:evaluator.num_states]
    num_internal = evaluator.leaf_offset  # 2^depth - 1 internal nodes
    root_obs = all_state_obs[0]
    root_v_true = v_true_all[0]
    term_idx = evaluator.terminal_idx

    # Pre-extract leaf reward table: shape (evaluator.num_total_states,)
    leaf_reward_table = jnp.asarray(env.R[:, 0, term_idx], dtype=jnp.float32)

    # Compute transition accounting
    transitions_per_update = compute_sample_costs(num_envs, depth, k_branches)
    total_timesteps = int(config["TOTAL_TIMESTEPS"])
    num_updates = total_timesteps // transitions_per_update

    def train(rng: jax.random.PRNGKey):
        # 2. Transformer Critic Initialization
        network = TransformerCritic(
            d_model=int(config.get("D_MODEL", 64)),
            n_heads=int(config.get("N_HEADS", 4)),
            n_layers=int(config.get("N_LAYERS", 2)),
            final_hidden_dim=int(config.get("FINAL_HIDDEN_DIM", 32)),
            max_len=depth + 4,
        )

        total_grad_steps = num_updates * depth * stage_steps
        warmup_steps = int(config.get("WARMUP_RATIO", 0.1) * total_grad_steps)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=float(config["LR"]),
            warmup_steps=warmup_steps,
            decay_steps=total_grad_steps,
            end_value=1e-5,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.get("MAX_GRAD_NORM", 1.0))),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=float(config.get("WEIGHT_DECAY", 0.01)),
                b1=0.9,
                b2=0.95,
            ),
        )

        rng, init_rng = jax.random.split(rng)
        network_params = network.init(init_rng, root_obs)
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=optimizer)

        # 3. Main Update Step
        def _update_step(runner_state, unused):
            train_state, rng, update_idx = runner_state

            # --- STEP A: COLLECT BASE TRAJECTORIES FROM ROOT ---
            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            init_obs, init_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)

            def _rollout_base_step(curr, _):
                curr_s, k = curr
                k, step_k = jax.random.split(k)
                step_keys = jax.random.split(step_k, num_envs)
                actions = jnp.zeros(num_envs, dtype=jnp.int32)
                next_obs, next_state, rewards, dones, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, curr_s, actions, env_params)
                return (next_state, k), (curr_s.s_idx, next_obs, next_state.s_idx)

            # Rollout depth steps: from root (depth 0) to leaf (depth `depth`)
            (leaf_env_state, rng), (step_s_idx, step_next_obs, step_next_s_idx) = jax.lax.scan(
                _rollout_base_step, (init_state, rng), None, length=depth
            )

            # Assemble base trajectories:
            # base_states: shape (depth + 1, num_envs)
            # base_obs:    shape (depth + 1, num_envs, obs_dim)
            base_states = jnp.concatenate([init_state.s_idx[None, :], step_next_s_idx], axis=0)
            base_obs = jnp.concatenate([init_obs[None, :, :], step_next_obs], axis=0)

            # Leaves are at index `depth`:
            leaf_states = base_states[depth]
            leaf_rewards = leaf_reward_table[leaf_states]  # (num_envs,)

            # --- STEP B: SAMPLE K 1-STEP BRANCHES FOR EACH VISITED STATE (VMAPPED) ---
            rng, branch_rng = jax.random.split(rng)
            level_keys = jax.random.split(branch_rng, depth)

            def _branch_level(d, level_key):
                curr_s_level = base_states[d]  # (num_envs,)
                branch_s_idx = jnp.repeat(curr_s_level, k_branches)
                branch_time = jnp.full(num_envs * k_branches, d, dtype=jnp.int32)
                branch_in_state = TabularState(s_idx=branch_s_idx, time=branch_time)

                step_keys = jax.random.split(level_key, num_envs * k_branches)
                actions = jnp.zeros(num_envs * k_branches, dtype=jnp.int32)
                b_next_obs, b_next_s, _, _, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, branch_in_state, actions, env_params)

                b_next_obs = b_next_obs.reshape(num_envs, k_branches, -1)
                b_next_s_idx = b_next_s.s_idx.reshape(num_envs, k_branches)
                return b_next_obs, b_next_s_idx

            # Parallelize 1-step branching across all depths 0 .. depth - 1:
            # all_branch_obs:   (depth, num_envs, k_branches, obs_dim)
            # all_branch_s_idx: (depth, num_envs, k_branches)
            all_branch_obs, all_branch_s_idx = jax.vmap(_branch_level)(
                jnp.arange(depth, dtype=jnp.int32), level_keys
            )

            # --- STEP C: INTERLEAVED BACKWARD FITTING WITH REPLAY ANCHORING (XLA SCAN) ---
            targets = jnp.zeros((depth + 1, num_envs), dtype=jnp.float32)

            # 1. Terminal Boundary: leaf states (d = depth)
            targets = targets.at[depth].set(leaf_rewards)

            # 2. Penultimate Anchors (d = depth - 1) & Branched Leaf Endpoints:
            penultimate_leaf_branches = all_branch_s_idx[depth - 1]  # (num_envs, k_branches)
            penultimate_leaf_obs = all_branch_obs[depth - 1].reshape(-1, base_obs.shape[-1])  # (num_envs * k_branches, obs_dim)
            penultimate_rewards = leaf_reward_table[penultimate_leaf_branches]  # (num_envs, k_branches)
            flat_penult_leaf_rewards = penultimate_rewards.reshape(-1)
            num_penult_leaves = flat_penult_leaf_rewards.shape[0]
            y_penultimate = jnp.mean(penultimate_rewards, axis=-1)  # (num_envs,)
            targets = targets.at[depth - 1].set(y_penultimate)

            # Query baseline prediction at penultimate level for GAE residual delta_{depth-1}
            v_penult = jax.lax.stop_gradient(
                network.apply(train_state.params, base_obs[depth - 1], method=network.value)
            )
            delta_penult = y_penultimate - v_penult
            gae_adv = delta_penult  # (num_envs,) GAE accumulator initialized at penultimate step

            # Fast XLA active-pool minibatch fitting stage:
            # Samples static-shape minibatches strictly from already-resolved levels [min_level .. depth]
            # (zero wasted FLOPs on unresolved masked levels).
            def _fit_stage(curr_state, min_level, curr_targets, stage_rng):
                def _single_grad_step(carry, _):
                    st, k = carry
                    k, k_lvl, k_env, k_leaf = jax.random.split(k, 4)

                    # Sample minibatch strictly from active levels [min_level, depth]
                    mb_levels = jax.random.randint(k_lvl, (minibatch_size,), minval=min_level, maxval=depth + 1)
                    mb_envs = jax.random.randint(k_env, (minibatch_size,), minval=0, maxval=num_envs)
                    mb_trunk_obs = base_obs[mb_levels, mb_envs]
                    mb_trunk_targets = curr_targets[mb_levels, mb_envs]

                    # Sample minibatch from terminal leaf branch endpoints
                    mb_leaf_idx = jax.random.randint(k_leaf, (minibatch_size,), minval=0, maxval=num_penult_leaves)
                    mb_leaf_obs = penultimate_leaf_obs[mb_leaf_idx]
                    mb_leaf_targets = flat_penult_leaf_rewards[mb_leaf_idx]

                    # Unified single forward pass over trunk + leaf anchor minibatches
                    combined_obs = jnp.concatenate([mb_trunk_obs, mb_leaf_obs], axis=0)

                    def loss_fn(p):
                        combined_preds = network.apply(p, combined_obs, method=network.value)
                        trunk_preds = combined_preds[:minibatch_size]
                        leaf_preds = combined_preds[minibatch_size:]

                        trunk_loss = 0.5 * jnp.mean(jnp.square(trunk_preds - mb_trunk_targets))
                        leaf_branch_loss = 0.5 * jnp.mean(jnp.square(leaf_preds - mb_leaf_targets))
                        return trunk_loss + 0.25 * leaf_branch_loss

                    loss_val, grads = jax.value_and_grad(loss_fn)(st.params)
                    new_st = st.apply_gradients(grads=grads)
                    return (new_st, k), loss_val

                (new_state, _), losses = jax.lax.scan(_single_grad_step, (curr_state, stage_rng), None, length=stage_steps)
                return new_state, losses[-1]

            # Initial cumulative buffer D_fit: terminal leaves (depth) and penultimate anchors (depth - 1)
            rng, penult_fit_rng, scan_fit_rng = jax.random.split(rng, 3)
            train_state, loss_penult = _fit_stage(train_state, jnp.int32(depth - 1), targets, penult_fit_rng)

            # 3. Inductive Lookaheads backward via jax.lax.scan (t = depth - 2 down to 0):
            def _backward_stage_step(carry, t):
                curr_train_state, curr_targets, curr_gae_adv, curr_rng = carry
                curr_rng, stage_k = jax.random.split(curr_rng)

                next_branch_obs = all_branch_obs[t]  # (num_envs, k_branches, obs_dim)
                flat_next_branch_obs = next_branch_obs.reshape(-1, next_branch_obs.shape[-1])
                # Single fused forward pass for both current trunk state and K 1-step children
                eval_obs_t = jnp.concatenate([base_obs[t], flat_next_branch_obs], axis=0)
                eval_v_t = jax.lax.stop_gradient(
                    network.apply(curr_train_state.params, eval_obs_t, method=network.value)
                )
                v_curr_t = eval_v_t[:num_envs]
                v_next = eval_v_t[num_envs:].reshape(num_envs, k_branches)
                v_branch_t = jnp.mean(v_next, axis=-1)  # (num_envs,)

                # Branched Bellman residual: delta_t = E_{a~pi}[V(s')] - V(s)
                delta_t = v_branch_t - v_curr_t

                # GAE accumulator: A_t^(lambda) = delta_t + gamma * lambda * A_{t+1}^(lambda)
                next_gae_adv = delta_t + (gamma * lmbda) * curr_gae_adv

                # Target: y(s_t) = V(s_t) + A_t^(lambda)
                y_t = v_curr_t + next_gae_adv
                curr_targets = curr_targets.at[t].set(y_t)

                # Cumulative Replay Anchoring: active levels are now [t .. depth]
                curr_train_state, stage_loss = _fit_stage(curr_train_state, t, curr_targets, stage_k)
                return (curr_train_state, curr_targets, next_gae_adv, curr_rng), stage_loss

            backward_levels = jnp.arange(depth - 2, -1, -1, dtype=jnp.int32)
            (train_state, targets, gae_adv, _), stage_losses = jax.lax.scan(
                _backward_stage_step,
                (train_state, targets, gae_adv, scan_fit_rng),
                backward_levels,
            )
            loss_t = stage_losses[-1] if depth >= 2 else loss_penult

            # --- STEP D: EVALUATION METRICS ---
            all_preds = network.apply(train_state.params, all_state_obs, method=network.value)
            val_mse = jnp.mean(jnp.square(all_preds - v_true_all))
            internal_mse = jnp.mean(jnp.square(all_preds[:num_internal] - v_true_all[:num_internal]))
            leaf_mse = jnp.mean(jnp.square(all_preds[num_internal:] - v_true_all[num_internal:]))
            root_abs_err = jnp.abs(all_preds[0] - root_v_true)
            max_abs_err = jnp.max(jnp.abs(all_preds - v_true_all))
            ranking_acc = compute_sibling_ranking_accuracy(all_preds, v_true_all, num_internal)
            adv_cos_sim = compute_advantage_cosine_similarity(all_preds, v_true_all, num_internal)

            metrics = {
                "value_error": val_mse,
                "internal_mse": internal_mse,
                "leaf_mse": leaf_mse,
                "root_abs_error": root_abs_err,
                "max_abs_error": max_abs_err,
                "sibling_ranking_acc": ranking_acc,
                "advantage_cosine_sim": adv_cos_sim,
                "loss": loss_t,
                "target_mean": jnp.mean(targets),
            }

            return (train_state, branch_rng, update_idx + 1), metrics

        # Run all updates via jax.lax.scan
        initial_runner_state = (train_state, rng, 0)
        final_runner_state, update_metrics = jax.lax.scan(
            _update_step, initial_runner_state, None, length=num_updates
        )

        return {"runner_state": final_runner_state, "metrics": update_metrics}

    return train, transitions_per_update


def run_fbi_experiment(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    gamma: float = 1.0,
    k_branches: int = 4,
    lmbda: float = 0.0,
    stage_steps: int = 4,
    num_envs: int = 64,
    total_timesteps: int = 120000,
    lr: float = 1e-3,
    seed: int = 42,
    save_dir: str = "results/fbi_value",
) -> Dict[str, Any]:
    """Runs a single training run of 1-Step Fitted Backward Induction (FBI(lambda))."""
    cfg = {
        "DEPTH": depth,
        "P_SUCCESS": p,
        "CLUSTER_SIZE": cluster_size,
        "GAMMA": gamma,
        "K_BRANCHES": k_branches,
        "LMBDA": lmbda,
        "STAGE_STEPS": stage_steps,
        "NUM_ENVS": num_envs,
        "TOTAL_TIMESTEPS": total_timesteps,
        "LR": lr,
        "D_MODEL": 64,
        "N_HEADS": 4,
        "N_LAYERS": 2,
        "FINAL_HIDDEN_DIM": 32,
        "WEIGHT_DECAY": 0.01,
        "MAX_GRAD_NORM": 1.0,
        "WARMUP_RATIO": 0.1,
    }

    train_fn, trans_per_up = make_train(cfg)
    jitted_train = jax.jit(train_fn)

    cost = compute_sample_costs(num_envs, depth, k_branches)
    num_updates = total_timesteps // cost

    print(f"\n{'='*75}")
    print(f"FITTED BACKWARD INDUCTION FBI(lambda={lmbda}) TRAINING")
    print(f"Tree Depth={depth}, p={p}, Branches K={k_branches}, lambda={lmbda}, Envs={num_envs}")
    print(f"Transitions per Update: {trans_per_up:,d} (Base={num_envs*depth}, Branch={num_envs*depth*k_branches})")
    print(f"Total Updates: {num_updates:,d} | Total Budget: {total_timesteps:,d} transitions")
    print(f"{'='*75}")

    t0 = time.time()
    rng = jax.random.PRNGKey(seed)
    out = jitted_train(rng)
    jax.block_until_ready(out)
    elapsed = time.time() - t0

    metrics = out["metrics"]
    final_mse = float(metrics["value_error"][-1])
    final_int_mse = float(metrics["internal_mse"][-1])
    final_root_err = float(metrics["root_abs_error"][-1])
    final_max_err = float(metrics["max_abs_error"][-1])
    final_rank_acc = float(metrics["sibling_ranking_acc"][-1])

    print(f"\nTraining completed in {elapsed:.2f}s ({num_updates / elapsed:.1f} updates/s)")
    print(f"  Final Global Value MSE:     {final_mse:.6f}")
    print(f"  Final Internal Node MSE:    {final_int_mse:.6f}")
    print(f"  Final Root Abs Error:       {final_root_err:.6f}")
    print(f"  Final Max Abs Error:        {final_max_err:.6f}")
    print(f"  Sibling Ranking Accuracy:   {final_rank_acc*100:.1f}%")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"fbi_d{depth}_p{p}_k{k_branches}_lam{lmbda}_seed{seed}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"config": cfg, "metrics": jax.device_get(metrics)}, f)
    print(f"Saved run data to {out_path}")

    # Plot single run learning curve
    x_trans = np.arange(1, num_updates + 1) * trans_per_up
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: Value Error vs Transitions
    axes[0].plot(x_trans, metrics["value_error"], color="#9467bd", linewidth=2.0, label="All States MSE")
    axes[0].plot(x_trans, metrics["internal_mse"], color="#e377c2", linewidth=1.8, linestyle="--", label="Internal Nodes MSE")
    axes[0].set_title(f"FBI (K={k_branches}, $\\lambda$={lmbda}) Value Error vs. Transitions", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Total Transitions", fontsize=10)
    axes[0].set_ylabel("MSE vs. $V^*$", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right: Root State Error vs Transitions
    axes[1].plot(x_trans, metrics["root_abs_error"], color="#9467bd", linewidth=2.0)
    axes[1].set_title(f"FBI (K={k_branches}, $\\lambda$={lmbda}) Root State Error vs. Transitions", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Total Transitions", fontsize=10)
    axes[1].set_ylabel("$|V(root) - V^*(root)|$", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"fbi_curve_k{k_branches}_lam{lmbda}.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(f"Learning curve saved to {fig_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Step Fitted Backward Induction (FBI(lambda)) Value Learning")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--cluster-size", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--k-branches", type=int, default=4)
    parser.add_argument("--lmbda", type=float, default=0.0, help="FBI(lambda) interpolation parameter in [0, 1]")
    parser.add_argument("--stage-steps", type=int, default=4, help="Gradient steps per backward stage")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/fbi_value")
    args = parser.parse_args()

    run_fbi_experiment(
        depth=args.depth,
        p=args.p,
        cluster_size=args.cluster_size,
        gamma=args.gamma,
        k_branches=args.k_branches,
        lmbda=args.lmbda,
        stage_steps=args.stage_steps,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.out_dir,
    )
