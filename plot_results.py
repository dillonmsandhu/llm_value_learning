from __future__ import annotations
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Publication-quality styling: Vector fonts TrueType 42
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
})

METHOD_COLORS = {
    "Standard MC": "#1f77b4",          # Blue
    "Vine-Target (K=2)": "#ff7f0e",    # Orange
    "Vine-Target (K=4)": "#d62728",    # Red
    "FBI (lam=0.0)": "#2ca02c",        # Green
    "FBI (lam=0.5)": "#9467bd",        # Purple
    "Standard MC (K=1)": "#1f77b4",
    "FBI (K=4, lam=0.0)": "#2ca02c",
    "FBI (K=4, lam=0.5)": "#9467bd",
}

DEFAULT_PALETTE = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def get_color(method_name: str, idx: int = 0) -> str:
    for key, c in METHOD_COLORS.items():
        if key in method_name or method_name in key:
            return c
    return DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]


def compute_normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Normalized Area Under the Curve: (1 / (X_max - X_min)) * int_Xmin^Xmax y dx."""
    dx = x[-1] - x[0]
    if dx <= 1e-8:
        return float(np.mean(y))
    # Support numpy 2.0+ and <2.0
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = trapz_fn(y, x)
    return float(area / dx)


def plot_comparison_curves(
    results: Dict[str, Any],
    save_dir: str,
    prefix: str = "benchmark",
) -> None:
    """
    Produces a publication-quality 2x3 grid:
      Row 1: MSVE vs Gradient Step | MSVE vs Sample Budget | MSVE vs Total Compute (Wall-Clock Seconds)
      Row 2: Advantage Cosine Sim vs Gradient Step | Adv Cos Sim vs Sample Budget | Adv Cos Sim vs Total Compute
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=300)
    fig.suptitle(f"Value Learning & Advantage Alignment Benchmark ({results.get('env_title', 'Reasoning Environment')})", y=0.98)

    method_names = list(results["methods"].keys())

    # Row 1: MSVE
    ax_msve_step = axes[0, 0]
    ax_msve_sample = axes[0, 1]
    ax_msve_time = axes[0, 2]

    # Row 2: Advantage Cosine Similarity
    ax_adv_step = axes[1, 0]
    ax_adv_sample = axes[1, 1]
    ax_adv_time = axes[1, 2]

    for idx, name in enumerate(method_names):
        data = results["methods"][name]
        color = get_color(name, idx)

        # X-axes:
        x_steps = np.asarray(data["x_updates"])
        x_samples = np.asarray(data["x_samples"])
        x_times = np.asarray(data["x_wallclock"])  # shape: (n_seeds, num_evals) or (num_evals,)
        if x_times.ndim == 2:
            mean_time = np.mean(x_times, axis=0)
        else:
            mean_time = x_times

        # Metrics (shape: (n_seeds, num_evals))
        msve = np.asarray(data["msve"])
        adv_cos = np.asarray(data["advantage_cosine_sim"])

        mean_msve = np.mean(msve, axis=0)
        sem_msve = np.std(msve, axis=0) / np.sqrt(msve.shape[0])

        mean_adv = np.mean(adv_cos, axis=0)
        sem_adv = np.std(adv_cos, axis=0) / np.sqrt(adv_cos.shape[0])

        # --- Panel 1: MSVE vs Updates ---
        ax_msve_step.plot(x_steps, mean_msve, label=name, color=color, lw=2.0)
        ax_msve_step.fill_between(x_steps, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.18)

        # --- Panel 2: MSVE vs Samples ---
        ax_msve_sample.plot(x_samples, mean_msve, label=name, color=color, lw=2.0)
        ax_msve_sample.fill_between(x_samples, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.18)

        # --- Panel 3: MSVE vs Wall-Clock Time ---
        ax_msve_time.plot(mean_time, mean_msve, label=name, color=color, lw=2.0)
        ax_msve_time.fill_between(mean_time, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.18)

        # --- Panel 4: Adv Cos Sim vs Updates ---
        ax_adv_step.plot(x_steps, mean_adv, label=name, color=color, lw=2.0)
        ax_adv_step.fill_between(x_steps, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.18)

        # --- Panel 5: Adv Cos Sim vs Samples ---
        ax_adv_sample.plot(x_samples, mean_adv, label=name, color=color, lw=2.0)
        ax_adv_sample.fill_between(x_samples, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.18)

        # --- Panel 6: Adv Cos Sim vs Wall-Clock Time ---
        ax_adv_time.plot(mean_time, mean_adv, label=name, color=color, lw=2.0)
        ax_adv_time.fill_between(mean_time, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.18)

    # Subplot formatting
    for ax in [ax_msve_step, ax_msve_sample, ax_msve_time]:
        ax.set_yscale("log")
        ax.set_ylabel("Mean Squared Value Error (MSVE)")
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)

    for ax in [ax_adv_step, ax_adv_sample, ax_adv_time]:
        ax.set_ylabel(r"Advantage Cosine Similarity $\cos(\hat{\mathbf{A}}, \mathbf{A}^*)$")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)

    ax_msve_step.set_title("MSVE vs. Gradient Updates")
    ax_msve_step.set_xlabel("Parameter Updates")
    ax_msve_sample.set_title("MSVE vs. Sample Transitions")
    ax_msve_sample.set_xlabel("Total Environment Transitions")
    ax_msve_time.set_title("MSVE vs. Total Compute (Seconds)")
    ax_msve_time.set_xlabel("Elapsed Wall-Clock Time (s)")

    ax_adv_step.set_title("Advantage Alignment vs. Updates")
    ax_adv_step.set_xlabel("Parameter Updates")
    ax_adv_sample.set_title("Advantage Alignment vs. Samples")
    ax_adv_sample.set_xlabel("Total Environment Transitions")
    ax_adv_time.set_title("Advantage Alignment vs. Total Compute")
    ax_adv_time.set_xlabel("Elapsed Wall-Clock Time (s)")

    ax_msve_step.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_path = os.path.join(save_dir, f"{prefix}_comparison_curves.pdf")
    png_path = os.path.join(save_dir, f"{prefix}_comparison_curves.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plotting] Saved comparison curves to:\n  - {pdf_path}\n  - {png_path}")


def plot_scatter_true_vs_estimated(
    results: Dict[str, Any],
    save_dir: str,
    prefix: str = "benchmark",
) -> None:
    """
    Produces a 2-row grid of scatter plots showing True Value V*(s) vs Estimated Value V_hat(s):
      - Row 1: Snapshot at 1/5 (20%) of learning
      - Row 2: Snapshot at 4/5 (80%) of learning
      - Columns: Each evaluated algorithm
      - Each point is a (state, seed) pair.
    """
    methods = list(results["methods"].keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, n_methods, figsize=(3.8 * n_methods, 7.5), dpi=300, squeeze=False)
    fig.suptitle(f"Critic Calibration: True Value $V^*(s)$ vs. Predicted $\hat{{V}}_\phi(s)$", y=0.98)

    true_values = np.asarray(results["eval_true_values"])  # (num_eval_states,)

    for col_idx, name in enumerate(methods):
        data = results["methods"][name]
        color = get_color(name, col_idx)

        # Snapshot 1: 1/5 learning (shape: (n_seeds, num_eval_states))
        preds_1_5 = np.asarray(data["scatter_snapshots"]["1_5"])
        # Snapshot 2: 4/5 learning (shape: (n_seeds, num_eval_states))
        preds_4_5 = np.asarray(data["scatter_snapshots"]["4_5"])

        for row_idx, (preds, stage_label) in enumerate([(preds_1_5, "1/5 Learning (20% Budget)"), (preds_4_5, "4/5 Learning (80% Budget)")]):
            ax = axes[row_idx, col_idx]

            # Flatten across seeds and states
            n_seeds, n_states = preds.shape
            x_pts = np.tile(true_values, n_seeds)
            y_pts = preds.flatten()

            # Scatter points
            ax.scatter(x_pts, y_pts, alpha=0.35, color=color, s=14, edgecolors="none")

            # Reference diagonal y = x
            min_val = min(0.0, float(np.min(true_values)), float(np.min(preds)))
            max_val = max(1.0, float(np.max(true_values)), float(np.max(preds)))
            pad = 0.05 * (max_val - min_val + 1e-4)
            line_range = [min_val - pad, max_val + pad]
            ax.plot(line_range, line_range, "k--", lw=1.2, alpha=0.7, label=r"Ideal $y = x$")

            # Summary metrics on scatter
            mse = float(np.mean((y_pts - x_pts) ** 2))
            corr = float(np.corrcoef(x_pts, y_pts)[0, 1]) if np.std(y_pts) > 1e-6 else 0.0

            ax.text(
                0.05, 0.88,
                f"MSVE: {mse:.4f}\nPearson $r$: {corr:.3f}",
                transform=ax.transAxes,
                fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )

            ax.set_xlim(line_range)
            ax.set_ylim(line_range)
            ax.grid(True, ls="--", lw=0.5, alpha=0.5)

            if row_idx == 0:
                ax.set_title(f"{name}\n({stage_label})", fontsize=10.5)
            else:
                ax.set_title(f"({stage_label})", fontsize=9.5)

            if col_idx == 0:
                ax.set_ylabel(r"Predicted Value $\hat{V}_\phi(s)$")
            if row_idx == 1:
                ax.set_xlabel(r"Ground-Truth $V^*(s)$")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_path = os.path.join(save_dir, f"{prefix}_scatter_true_vs_estimated.pdf")
    png_path = os.path.join(save_dir, f"{prefix}_scatter_true_vs_estimated.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plotting] Saved scatter calibration grid to:\n  - {pdf_path}\n  - {png_path}")


def compute_and_save_auc_summary(
    results: Dict[str, Any],
    save_dir: str,
    prefix: str = "benchmark",
) -> str:
    """
    Computes cross-algorithm aggregate AUC metrics for:
      - Normalized AUC(MSVE) vs Samples, Compute (time), Updates
      - Normalized AUC(Adv Cos Sim) vs Samples, Compute (time), Updates
      - Final Window MSVE and Adv Cos Sim
    Exports to summary_auc.csv and returns an aligned ASCII table.
    """
    rows = []
    headers = [
        "Algorithm",
        "Final MSVE",
        "Final AdvCos",
        "AUC_Sample(MSVE)",
        "AUC_Compute(MSVE)",
        "AUC_Update(MSVE)",
        "AUC_Sample(Adv)",
        "AUC_Compute(Adv)",
        "AUC_Update(Adv)",
    ]

    for name, data in results["methods"].items():
        x_steps = np.asarray(data["x_updates"])
        x_samples = np.asarray(data["x_samples"])
        x_times = np.asarray(data["x_wallclock"])
        mean_time = np.mean(x_times, axis=0) if x_times.ndim == 2 else x_times

        msve = np.asarray(data["msve"])
        adv_cos = np.asarray(data["advantage_cosine_sim"])

        mean_msve = np.mean(msve, axis=0)
        sem_msve = np.std(msve, axis=0) / np.sqrt(msve.shape[0])
        mean_adv = np.mean(adv_cos, axis=0)
        sem_adv = np.std(adv_cos, axis=0) / np.sqrt(adv_cos.shape[0])

        final_msve_str = f"{mean_msve[-1]:.2e} ± {sem_msve[-1]:.1e}"
        final_adv_str = f"{mean_adv[-1]:.4f} ± {sem_adv[-1]:.3f}"

        # Normalized AUCs:
        auc_s_msve = compute_normalized_auc(x_samples, mean_msve)
        auc_c_msve = compute_normalized_auc(mean_time, mean_msve)
        auc_u_msve = compute_normalized_auc(x_steps, mean_msve)

        auc_s_adv = compute_normalized_auc(x_samples, mean_adv)
        auc_c_adv = compute_normalized_auc(mean_time, mean_adv)
        auc_u_adv = compute_normalized_auc(x_steps, mean_adv)

        rows.append({
            "Algorithm": name,
            "Final MSVE": final_msve_str,
            "Final AdvCos": final_adv_str,
            "AUC_Sample(MSVE)": f"{auc_s_msve:.4e}",
            "AUC_Compute(MSVE)": f"{auc_c_msve:.4e}",
            "AUC_Update(MSVE)": f"{auc_u_msve:.4e}",
            "AUC_Sample(Adv)": f"{auc_s_adv:.4f}",
            "AUC_Compute(Adv)": f"{auc_c_adv:.4f}",
            "AUC_Update(Adv)": f"{auc_u_adv:.4f}",
        })

    # Save to CSV
    csv_path = os.path.join(save_dir, f"{prefix}_summary_auc.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            line = ",".join([str(r[h]) for h in headers])
            f.write(line + "\n")

    # Format ASCII table
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep_line = "+" + "+".join(["-" * (col_widths[h] + 2) for h in headers]) + "+"
    header_line = "| " + " | ".join([h.ljust(col_widths[h]) for h in headers]) + " |"

    table_lines = [sep_line, header_line, sep_line]
    for r in rows:
        row_line = "| " + " | ".join([str(r[h]).ljust(col_widths[h]) for h in headers]) + " |"
        table_lines.append(row_line)
    table_lines.append(sep_line)
    table_str = "\n".join(table_lines)

    print(f"\n[Summary] Cross-Algorithm Aggregate AUC Summary:\n{table_str}")
    print(f"[Summary] Saved AUC metrics table to: {csv_path}\n")
    return table_str


def main():
    parser = argparse.ArgumentParser(description="Consolidated Plotting for Value Benchmark Suite")
    parser.add_argument("--data_path", type=str, required=True, help="Path to benchmark_results.pkl")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save figures and CSVs")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Filename prefix")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(args.data_path) or "results"
    os.makedirs(save_dir, exist_ok=True)

    with open(args.data_path, "rb") as f:
        results = pickle.load(f)

    plot_comparison_curves(results, save_dir, prefix=args.prefix)
    plot_scatter_true_vs_estimated(results, save_dir, prefix=args.prefix)
    compute_and_save_auc_summary(results, save_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
