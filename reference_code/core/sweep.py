import os
import time
import json
import itertools
import datetime
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.utils import save_results, merge_hparams



def plot_sweep_curves(
    curves,
    steps_per_pi,
    metric_key,
    env_name,
    out_dir,
    best_label=None,
    log_scale=True,
    title=None,
):
    """Plots mean learning curves for all hyperparameter configurations with clean styling."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort curves by final value (lowest first for error metrics)
    sorted_items = sorted(curves.items(), key=lambda item: float(item[1][-1]))
    
    colormap = plt.cm.turbo(np.linspace(0.05, 0.95, len(sorted_items)))
    
    for idx, (label, curve) in enumerate(sorted_items):
        y = np.asarray(curve)
        x = list(range(len(y)))
        is_best = (label == best_label) or (idx == 0)
        
        final_val = float(y[-1])
        display_label = f"{label} (final={final_val:.2e})"
        if is_best:
            display_label = f"★ [BEST] {display_label}"
            ax.plot(
                x, y,
                label=display_label,
                color="black" if len(sorted_items) > 1 else colormap[0],
                linewidth=2.8,
                linestyle="-",
                zorder=10,
                alpha=0.95,
            )
        else:
            ax.plot(
                x, y,
                label=display_label,
                color=colormap[idx],
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
            )

    if log_scale:
        ax.set_yscale("log")
    xlabel_str = f"Update Steps ({steps_per_pi} env steps/update)" if steps_per_pi > 1 else "Update Steps"
    ax.set_xlabel(xlabel_str, fontsize=12)
    ax.set_ylabel(metric_key, fontsize=12)
    plot_title = title or f"Hyperparameter Sweep: {metric_key} ({env_name})"
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Place legend outside if many curves, or upper right
    if len(curves) > 8:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, frameon=True)
    else:
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"hyperparameter_sweep_{metric_key}.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Sweep plot saved to {plot_path}")
    return plot_path


def plot_best_seeds(
    seed_trajectories,
    best_hparams,
    steps_per_pi,
    metric_key,
    env_name,
    out_dir,
    log_scale=True,
):
    """Plots individual seed trajectories for the best hyperparameter configuration."""
    arr = np.asarray(seed_trajectories)
    if arr.ndim != 2:
        return None
    
    n_seeds, time_steps = arr.shape
    x = list(range(time_steps))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    hparams_str = ", ".join([f"{k}={v}" for k, v in best_hparams.items()])
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_seeds, 10)))
    for s in range(n_seeds):
        ax.plot(
            x, arr[s],
            label=f"Seed {s} (final={arr[s, -1]:.2e})",
            color=colors[s % 10],
            linewidth=1.5,
            alpha=0.75,
        )
    
    mean_curve = arr.mean(axis=0)
    ax.plot(
        x, mean_curve,
        label=f"Mean across {n_seeds} seeds",
        color="black",
        linewidth=2.5,
        linestyle="--",
        zorder=10,
    )
    
    if log_scale:
        ax.set_yscale("log")
    xlabel_str = f"Update Steps ({steps_per_pi} env steps/update)" if steps_per_pi > 1 else "Update Steps"
    ax.set_xlabel(xlabel_str, fontsize=12)
    ax.set_ylabel(metric_key, fontsize=12)
    ax.set_title(f"Best Config Seeds ({hparams_str}) - {env_name}", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "best_config_seeds.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Best config seeds plot saved to {plot_path}")
    return plot_path


def plot_seeds_grid(
    metric_tensor,
    combinations,
    keys,
    steps_per_pi,
    metric_key,
    env_name,
    out_dir,
    log_scale=True,
):
    """Plots a multi-panel subplot grid showing individual seeds for each configuration."""
    arr = np.asarray(metric_tensor)  # (n_combos, n_seeds, time_steps)
    n_combos, n_seeds, time_steps = arr.shape
    if n_combos > 16 or n_seeds <= 1:
        return None
    
    if n_combos == 6:
        rows, cols = 2, 3
    elif n_combos <= 3:
        rows, cols = 1, n_combos
    elif n_combos == 4:
        rows, cols = 2, 2
    else:
        cols = min(4, n_combos)
        rows = (n_combos + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    x = list(range(time_steps))
    xlabel_str = f"Update Steps ({steps_per_pi} env steps/update)" if steps_per_pi > 1 else "Update Steps"
    
    for idx, combo in enumerate(combinations):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        current_params = dict(zip(keys, combo))
        label_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
        
        for s in range(n_seeds):
            ax.plot(x, arr[idx, s], alpha=0.6, linewidth=1.2)
        
        mean_y = arr[idx].mean(axis=0)
        ax.plot(x, mean_y, color="black", linewidth=2.0, linestyle="--")
        
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"#{idx}: {label_str}\n(final={mean_y[-1]:.2e})", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        if r == rows - 1:
            ax.set_xlabel(xlabel_str, fontsize=9)
        if c == 0:
            ax.set_ylabel(metric_key, fontsize=9)
            
    # Hide unused subplots
    for idx in range(n_combos, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis("off")
        
    fig.suptitle(f"All Configurations Seed Trajectories - {env_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "all_configs_seeds_grid.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"All configs seed grid saved to {plot_path}")
    return plot_path


def tune(
    make_train,
    base_config,
    param_grid,
    metric_key="nn_weighted_VE",
    rank_by="auc",
    rank_order="lower",
    window_size=20,
    save_dir="results/tuning",
    rng_seed=42,
    log_scale=True,
    save_checkpoint=False,
    save_metrics=True,
):
    """
    Parallel hyperparameter tuning using nested jax.vmap.
    
    Args:
        make_train: Factory function returning train(rng, hparams).
        base_config: Dictionary containing experiment configuration.
        param_grid: Dictionary mapping hyperparameter names to lists/tuples of values.
        metric_key: The metric to optimize / rank by (default: "nn_weighted_VE").
        rank_by: Metric ranking criterion: "auc" (default), "final_window", "final_step", "min", "max".
        rank_order: "lower" (lower is better, default) or "higher" (higher is better).
        window_size: Number of final steps to average when using "final_window" (default: 20).
        save_dir: Base directory to save tuning outputs.
        rng_seed: Base seed for PRNG.
        log_scale: Whether to plot on log y-scale.
        save_checkpoint: Whether to serialize full runner state in out.pkl.
        save_metrics: Whether to serialize metrics in out.pkl.
        
    Returns:
        result: Dictionary containing summary_df, best_config, best_hparams, best_idx,
                curves, metrics, and out_dir.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], (list, tuple)) else [param_grid[k]] for k in keys]
    combinations = list(itertools.product(*values))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = base_config.get("ENV_NAME", "unnamed_env")
    out_dir = os.path.join(save_dir, f"{timestamp}/{env_name}")
    os.makedirs(out_dir, exist_ok=True)

    n_seeds = base_config.get("N_SEEDS", 1)
    rng = jax.random.PRNGKey(base_config.get("SEED", rng_seed))
    rngs = jax.random.split(rng, n_seeds)

    train_fn = make_train(base_config)
    
    # vmap over seeds (axis 0 of rngs, None for hparams)
    vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

    print(f"\n{'='*60}")
    print(f"RUNNING SWEEP: {len(combinations)} configs x {n_seeds} seeds = {len(combinations)*n_seeds} runs")
    print(f"Environment: {env_name} | Primary Metric: {metric_key}")
    print(f"Hyperparameters: {keys}")
    print(f"Ranking By: {rank_by} ({'lower' if rank_order in ['lower', 'min', 'asc', 'ascending'] else 'higher'} is better)")
    print(f"{'='*60}\n")
    
    metrics_list = []
    runner_state_list = []
    for c_idx, combo in enumerate(combinations):
        current_hparams = {}
        for i, k in enumerate(keys):
            val = combo[i]
            if isinstance(val, (float, np.floating)):
                current_hparams[k] = jnp.array(val, dtype=jnp.float32)
            else:
                current_hparams[k] = jnp.array(val)

        param_desc = ", ".join([f"{k}={combo[i]}" for i, k in enumerate(keys)])
        print(f"[{c_idx+1}/{len(combinations)}] Evaluating: {param_desc} ...", end="", flush=True)
        t0 = time.time()
        out_c = vmapped_train(rngs, current_hparams)
        # Block until metric array is ready to accurately measure timing and ensure progress reporting
        if metric_key in out_c["metrics"]:
            _ = out_c["metrics"][metric_key].block_until_ready()
        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)", flush=True)
        metrics_list.append(out_c["metrics"])
        if save_checkpoint and "runner_state" in out_c:
            runner_state_list.append(out_c["runner_state"])

    # Stack metrics across configs so shape is (n_combos, n_seeds, time_steps)
    metrics = jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *metrics_list)
    out = {"metrics": metrics}
    if save_checkpoint and runner_state_list:
        out["runner_state"] = jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *runner_state_list)


    if metric_key not in metrics:
        raise KeyError(f"Metric '{metric_key}' not found. Available keys: {list(metrics.keys())}")

    # metrics[metric_key] shape: (n_combos, n_seeds, time_steps)
    metric_tensor = np.asarray(metrics[metric_key])
    mean_trajectories = metric_tensor.mean(axis=1)    # shape: (n_combos, time_steps)
    std_trajectories = metric_tensor.std(axis=1)      # shape: (n_combos, time_steps)

    curves = {}
    table_rows = []

    for idx, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        label = ", ".join([f"{k}={v}" for k, v in current_params.items()])

        curve = mean_trajectories[idx]
        curves[label] = curve
        time_steps = len(curve)
        win = max(1, min(time_steps, window_size))

        auc_mean = float(curve.mean())
        auc_std = float(metric_tensor[idx].mean(axis=-1).std()) if n_seeds > 1 else 0.0

        window_mean = float(curve[-win:].mean())
        window_std = float(metric_tensor[idx, :, -win:].mean(axis=-1).std()) if n_seeds > 1 else 0.0

        final_mean = float(curve[-1])
        final_std = float(std_trajectories[idx, -1]) if n_seeds > 1 else 0.0
        min_val = float(curve.min())
        max_val = float(curve.max())

        row = {
            "config_idx": idx,
            **current_params,
            f"auc_{metric_key}": auc_mean,
            f"auc_{metric_key}_std": auc_std,
            f"final_window_{metric_key}": window_mean,
            f"final_window_{metric_key}_std": window_std,
            f"final_{metric_key}_mean": final_mean,
            f"final_{metric_key}_std": final_std,
            f"min_{metric_key}": min_val,
            f"max_{metric_key}": max_val,
            # Backward-compatible column names:
            f"mean_{metric_key}": auc_mean,
            f"final_{metric_key}": final_mean,
        }
        table_rows.append(row)

    summary_df = pd.DataFrame(table_rows)
    if not summary_df.empty:
        # Determine sort column
        rank_by_lower = rank_by.lower()
        if rank_by_lower in ["auc", "mean", "time_mean", "auc_mean"]:
            sort_col = f"auc_{metric_key}"
        elif rank_by_lower in ["final_window", "window", "final_window_mean"]:
            sort_col = f"final_window_{metric_key}"
        elif rank_by_lower in ["final", "final_step", "final_mean", "last"]:
            sort_col = f"final_{metric_key}_mean"
        elif rank_by_lower in ["min", "minimum"]:
            sort_col = f"min_{metric_key}"
        elif rank_by_lower in ["max", "maximum"]:
            sort_col = f"max_{metric_key}"
        else:
            sort_col = f"auc_{metric_key}" if f"auc_{metric_key}" in summary_df.columns else f"final_{metric_key}_mean"

        is_ascending = rank_order.lower() in ["lower", "min", "asc", "ascending"]
        summary_df = summary_df.sort_values(by=sort_col, ascending=is_ascending).reset_index(drop=True)
        summary_df["rank"] = summary_df.index + 1

        # Reorder columns with rank first
        cols = ["rank", "config_idx"] + keys + [
            f"auc_{metric_key}",
            f"final_window_{metric_key}",
            f"final_{metric_key}_mean",
            f"final_{metric_key}_std",
            f"min_{metric_key}",
            f"max_{metric_key}",
            f"mean_{metric_key}",
        ]
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]

    # Determine best configuration
    best_row = summary_df.iloc[0]
    best_idx = int(best_row["config_idx"])
    best_hparams = {k: best_row[k] for k in keys}
    best_label = ", ".join([f"{k}={v}" for k, v in best_hparams.items()])
    best_full_config = merge_hparams(base_config, best_hparams)

    # Save summary CSV
    csv_path = os.path.join(out_dir, "tuning_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nTuning summary saved to {csv_path}")
    
    # Save summary JSON
    json_path = os.path.join(out_dir, "tuning_summary.json")
    summary_df.to_json(json_path, orient="records", indent=4)

    # Save best config JSON
    best_config_meta = {
        "best_hyperparameters": best_hparams,
        "best_config_idx": best_idx,
        "rank": 1,
        "primary_metric": metric_key,
        "rank_by": rank_by,
        "rank_order": rank_order,
        "window_size": win,
        f"auc_{metric_key}": float(best_row[f"auc_{metric_key}"]),
        f"final_window_{metric_key}": float(best_row[f"final_window_{metric_key}"]),
        f"final_{metric_key}_mean": float(best_row[f"final_{metric_key}_mean"]),
        f"final_{metric_key}_std": float(best_row[f"final_{metric_key}_std"]),
        f"min_{metric_key}": float(best_row[f"min_{metric_key}"]),
        f"max_{metric_key}": float(best_row[f"max_{metric_key}"]),
        f"mean_{metric_key}_over_time": float(best_row[f"auc_{metric_key}"]),
        "timestamp": timestamp,
        "env_name": env_name,
        "n_seeds": n_seeds,
        "config": best_full_config,
    }
    best_config_path = os.path.join(out_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config_meta, f, indent=4)
    print(f"Best configuration saved to {best_config_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"★ TOP CONFIGURATION: {best_label} (Ranked by {rank_by} [{'lower' if is_ascending else 'higher'} is better])")
    print(f"  AUC {metric_key}:          {best_row[f'auc_{metric_key}']:.4e}")
    print(f"  Final Window ({win} steps): {best_row[f'final_window_{metric_key}']:.4e}")
    print(f"  Final {metric_key}:         {best_row[f'final_{metric_key}_mean']:.4e} (±{best_row[f'final_{metric_key}_std']:.4e})")
    print(f"  Min {metric_key}:           {best_row[f'min_{metric_key}']:.4e}")
    print("=" * 60)
    print("\nTop 5 Configurations:")
    print(summary_df.head(5).to_string(index=False))
    print("=" * 60 + "\n")

    # Save raw outputs/checkpoints
    if save_checkpoint:
        save_results(out, base_config, env_name, out_dir)
    elif save_metrics:
        save_results(metrics, base_config, env_name, out_dir)
    else:
        save_results(base_config, base_config, env_name, out_dir)

    # Generate Plots
    steps_per_pi = base_config.get("NUM_ENVS", 1) * base_config.get("NUM_STEPS", 1)
    
    # 1. All curves plot
    plot_sweep_curves(
        curves=curves,
        steps_per_pi=steps_per_pi,
        metric_key=metric_key,
        env_name=env_name,
        out_dir=out_dir,
        best_label=best_label,
        log_scale=log_scale,
    )

    # 2. Best config seed trajectories
    if n_seeds > 1:
        plot_best_seeds(
            seed_trajectories=metric_tensor[best_idx],
            best_hparams=best_hparams,
            steps_per_pi=steps_per_pi,
            metric_key=metric_key,
            env_name=env_name,
            out_dir=out_dir,
            log_scale=log_scale,
        )

        # 3. Multi-grid plot if <= 16 configs
        plot_seeds_grid(
            metric_tensor=metric_tensor,
            combinations=combinations,
            keys=keys,
            steps_per_pi=steps_per_pi,
            metric_key=metric_key,
            env_name=env_name,
            out_dir=out_dir,
            log_scale=log_scale,
        )

    return {
        "summary_df": summary_df,
        "best_config": best_full_config,
        "best_hparams": best_hparams,
        "best_idx": best_idx,
        "curves": curves,
        "metrics": metrics,
        "out_dir": out_dir,
        "timestamp": timestamp,
        "env_name": env_name,
    }

