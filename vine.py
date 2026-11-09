from __future__ import annotations
import os
import argparse
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional

from envs.tree_mrp import TreeMRPExactValue, TreeMRP
from envs.tabular_matrix_env import TabularParams, TabularState


def estimate_vine_single_state(
    env: TreeMRP,
    params: TabularParams,
    s_idx: int | jnp.ndarray,
    k_rollouts: int,
    rng: jax.random.PRNGKey,
    max_steps: int,
    gamma: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forks K independent Monte Carlo rollouts directly from state s_idx until termination.
    Returns:
      v_hat: estimated scalar value (mean return across K rollouts)
      returns: individual Monte Carlo returns of shape (K,)
    """
    # Initialize K copies of the environment at state s_idx
    init_state = TabularState(
        s_idx=jnp.full((k_rollouts,), s_idx, dtype=jnp.int32),
        time=jnp.zeros(k_rollouts, dtype=jnp.int32),
    )

    def _step_fn(carry, _):
        curr_state, k = carry
        k, step_k = jax.random.split(k)
        step_keys = jax.random.split(step_k, k_rollouts)
        actions = jnp.zeros(k_rollouts, dtype=jnp.int32)

        next_obs, next_state, rewards, dones, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, curr_state, actions, params)

        next_carry = (next_state, k)
        return next_carry, (rewards, dones)

    _, (rewards_seq, dones_seq) = jax.lax.scan(
        _step_fn, (init_state, rng), None, length=max_steps
    )
    # rewards_seq: (max_steps, k_rollouts)
    # dones_seq: (max_steps, k_rollouts)

    # Compute discounted cumulative return backwards
    def _backward_scan(next_g, step_data):
        r, d = step_data
        g = r + gamma * (1.0 - d.astype(jnp.float32)) * next_g
        return g, g

    _, returns_seq = jax.lax.scan(
        _backward_scan, jnp.zeros(k_rollouts, dtype=jnp.float32), (rewards_seq, dones_seq), reverse=True
    )
    # Return at t=0 (the state s_idx)
    g_0 = returns_seq[0]
    v_hat = jnp.mean(g_0)
    return v_hat, g_0


def estimate_vine_all_states(
    env: TreeMRP,
    params: TabularParams,
    k_rollouts: int,
    rng: jax.random.PRNGKey,
    max_steps: int,
    gamma: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimates the Vine value for ALL active tree states simultaneously in parallel using jax.vmap.
    Returns:
      v_vine: (num_states,) array of estimated values
      returns_all: (num_states, k_rollouts) array of sampled returns
    """
    num_states = env.num_states
    all_indices = jnp.arange(num_states, dtype=jnp.int32)
    state_keys = jax.random.split(rng, num_states)

    def _eval_state(s, k_rng):
        return estimate_vine_single_state(
            env=env,
            params=params,
            s_idx=s,
            k_rollouts=k_rollouts,
            rng=k_rng,
            max_steps=max_steps,
            gamma=gamma,
        )

    v_vine, returns_all = jax.vmap(_eval_state)(all_indices, state_keys)
    return v_vine, returns_all


def run_vine_sweep(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    k_values: Tuple[int, ...] = (1, 4, 16, 64),
    seed: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluates Vine estimation error and variance across multiple rollout budgets K.
    """
    evaluator = TreeMRPExactValue(depth=depth, p=p, cluster_size=cluster_size)
    env = TreeMRP(depth=depth, p=p, cluster_size=cluster_size)
    env_params = TabularParams(max_steps_in_episode=depth + 2)
    v_true = evaluator.solve_backward_induction()[:evaluator.num_states]

    rng = jax.random.PRNGKey(seed)
    max_steps = depth + 1

    results = {}
    print(f"=== Vine Online Value Estimator Sweep (Depth={depth}, p={p}) ===")
    print(f"Total states: {evaluator.num_states} | Exact V*(root) = {float(v_true[0]):.4f}\n")
    print(f"{'K (Rollouts)':>12} | {'MSE vs V*':>12} | {'Max Abs Err':>12} | {'Empirical Var':>14} | {'Theory Var':>12} | {'V_hat(root)':>12}")
    print("-" * 85)

    for k in k_values:
        rng, step_rng = jax.random.split(rng)
        # JIT-compile the full state space evaluation for this K
        v_vine_fn = jax.jit(lambda r: estimate_vine_all_states(env, env_params, k, r, max_steps))
        v_vine, returns_all = v_vine_fn(step_rng)

        mse = float(jnp.mean(jnp.square(v_vine - v_true)))
        max_err = float(jnp.max(jnp.abs(v_vine - v_true)))
        emp_var = float(jnp.mean(jnp.var(returns_all, axis=-1)))
        theory_var = float(jnp.mean((v_true * (1.0 - v_true)) / k))
        root_est = float(v_vine[0])

        results[k] = {
            "v_vine": v_vine,
            "v_true": v_true,
            "mse": mse,
            "max_err": max_err,
            "emp_var": emp_var,
            "theory_var": theory_var,
            "root_est": root_est,
        }

        print(
            f"{k:12d} | "
            f"{mse:12.6f} | "
            f"{max_err:12.4f} | "
            f"{emp_var:14.6f} | "
            f"{theory_var:12.6f} | "
            f"{root_est:12.4f}"
        )

    return results


def plot_vine_figure3(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    k_list: Tuple[int, ...] = (1, 4, 16, 64),
    n_seeds: int = 15,
    seed: int = 42,
    save_path: str = "results/figures/vine_figure3.png",
):
    """
    Reproduces Vine Figure 3: Scatter plot of True Value vs Predicted Value for all states across multiple seeds.
    Horizontal axis = Ground truth value due to backward induction V*(s).
    Vertical axis = Vine Monte Carlo estimate V_hat(s) with K rollouts.
    All outcomes across seeds are plotted simultaneously to visualize estimator variance.
    """
    evaluator = TreeMRPExactValue(depth=depth, p=p, cluster_size=cluster_size)
    env = TreeMRP(depth=depth, p=p, cluster_size=cluster_size)
    env_params = TabularParams(max_steps_in_episode=depth + 2)
    v_true = evaluator.solve_backward_induction()[:evaluator.num_states]

    # Node depths for coloring
    depths = jnp.array([len(evaluator.state_strings[s]) for s in range(evaluator.num_states)])

    rng = jax.random.PRNGKey(seed)
    max_steps = depth + 1

    n_cols = len(k_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.2), sharey=True, sharex=True)
    if n_cols == 1:
        axes = [axes]

    for ax, k in zip(axes, k_list):
        rng, step_rng = jax.random.split(rng)
        seed_keys = jax.random.split(step_rng, n_seeds)

        # Vectorized evaluation over all seeds in parallel via vmap
        v_vine_fn = jax.jit(jax.vmap(lambda r: estimate_vine_all_states(env, env_params, k, r, max_steps)[0]))
        v_vine_seeds = v_vine_fn(seed_keys)  # (n_seeds, num_states)

        mean_v_vine = jnp.mean(v_vine_seeds, axis=0)
        mean_var = float(jnp.mean(jnp.var(v_vine_seeds, axis=0)))
        mse = float(jnp.mean(jnp.square(v_vine_seeds - v_true[None, :])))

        # Plot individual outcomes for all seeds with alpha so density and spread are clear
        for s_idx in range(n_seeds):
            scatter = ax.scatter(
                v_true,
                v_vine_seeds[s_idx],
                c=depths,
                cmap="viridis",
                s=32,
                alpha=0.35,
                edgecolors="none",
            )

        # Overlay mean across seeds with dark border to show calibration / unbiasedness
        ax.scatter(
            v_true,
            mean_v_vine,
            c=depths,
            cmap="viridis",
            s=50,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.7,
            label="Mean across seeds" if ax == axes[0] else None,
        )

        # Diagonal line y = x (perfect calibration)
        ax.plot([0.0, 1.0], [0.0, 1.0], "r--", linewidth=1.5, alpha=0.8, label="Ideal ($y = x$)" if ax == axes[0] else None)
        ax.set_title(f"Vine (K = {k})\nMSE = {mse:.4f} | Var = {mean_var:.4f}", fontsize=11)
        ax.set_xlabel("True Value $V^*(s)$", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Vine Estimate $\\hat{V}_{\\text{vine}}(s)$", fontsize=11)
    cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.05, pad=0.2)
    cbar.set_label("Tree Depth of State", fontsize=10)
    fig.suptitle(f"Vine Online Estimator: True vs. Estimated Values across {n_seeds} Seeds", fontsize=12, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--cluster-size", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=15, help="Number of seeds to plot in Figure 3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_vine_sweep(depth=args.depth, p=args.p, cluster_size=args.cluster_size, seed=args.seed)
    plot_vine_figure3(
        depth=args.depth,
        p=args.p,
        cluster_size=args.cluster_size,
        n_seeds=args.seeds,
        seed=args.seed,
        save_path="results/figures/vine_figure3.png",
    )
