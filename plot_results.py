from __future__ import annotations
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

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
    "Dirichlet (double-sampled)": "#e377c2",
    "Standard MC (Compute-Matched)": "#0b3c5d",  # Dark Navy (dashed)
    "Standard MC": "#1f77b4",                    # Blue
    "TD (lam=0.0)": "#6b7280",                   # Slate Gray
    "TD (lam=0.8)": "#6366f1",                   # Indigo
    "Vine-Target (K=2)": "#ff7f0e",              # Orange
    "Vine-Target (K=4)": "#d62728",              # Red
    "Vine MC Estimator (K=2)": "#b45309",        # Dark Amber (horizontal ref)
    "Vine MC Estimator (K=4)": "#7f1d1d",        # Dark Maroon (horizontal ref)
    "Batched FBI (lam=0.0)": "#2ca02c",          # Green
    "Batched FBI (lam=0.5)": "#9467bd",          # Purple
    "Batched FBI (lam=0.8)": "#8c564b",          # Brown
    "Staged FBI (lam=0.0)": "#17becf",           # Cyan
    "Staged FBI (lam=0.5)": "#e377c2",           # Magenta
    "Staged FBI (lam=0.8)": "#bcbd22",           # Olive-Gold
    "FBI (lam=0.0)": "#2ca02c",
    "FBI (lam=0.5)": "#9467bd",
    "FBI (lam=0.8)": "#8c564b",
    "Standard MC (K=1)": "#1f77b4",
    "FBI (K=4, lam=0.0)": "#2ca02c",
    "FBI (K=4, lam=0.5)": "#9467bd",
    "FBI (K=4, lam=0.8)": "#8c564b",
}

DEFAULT_PALETTE = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def get_color(method_name: str, idx: int = 0) -> str:
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
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
      Row 1: MSVE vs Gradient Steps | MSVE vs Active Rollout Transitions (Log) | MSVE vs Estimated LLM FLOPs (Log)
      Row 2: Advantage Cosine Sim across the same three X-axes
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), dpi=300)
    fig.suptitle(f"Value Learning & Advantage Alignment Benchmark ({results.get('env_title', 'Reasoning Environment')})", y=0.98)

    method_names = list(results["methods"].keys())
    has_llm_flops = all("x_llm_flops" in results["methods"][m] for m in method_names)
    base_total_grad_steps = results.get("base_total_grad_steps", None)
    has_compute_matched = any(results["methods"][m].get("is_compute_matched_mc", False) for m in method_names)

    # Row 1: MSVE
    ax_msve_step = axes[0, 0]
    ax_msve_sample = axes[0, 1]
    ax_msve_compute = axes[0, 2]

    # Row 2: Advantage Cosine Similarity
    ax_adv_step = axes[1, 0]
    ax_adv_sample = axes[1, 1]
    ax_adv_compute = axes[1, 2]

    for idx, name in enumerate(method_names):
        data = results["methods"][name]
        color = get_color(name, idx)
        is_matched_mc = data.get("is_compute_matched_mc", False)
        linestyle = "--" if is_matched_mc else "-"
        lw = 1.8 if is_matched_mc else 2.0

        # X-axes:
        x_steps = np.asarray(data["x_updates"])
        x_samples = np.asarray(data["x_samples"])
        if has_llm_flops:
            x_compute = np.asarray(data["x_llm_flops"])
        else:
            x_times = np.asarray(data["x_wallclock"])
            x_compute = np.mean(x_times, axis=0) if x_times.ndim == 2 else x_times

        # Metrics (shape: (n_seeds, num_evals))
        msve = np.asarray(data["msve"])
        adv_cos = np.asarray(data["advantage_cosine_sim"])

        mean_msve = np.mean(msve, axis=0)
        sem_msve = np.std(msve, axis=0) / np.sqrt(msve.shape[0])

        mean_adv = np.mean(adv_cos, axis=0)
        sem_adv = np.std(adv_cos, axis=0) / np.sqrt(adv_cos.shape[0])

        # --- Panel 1: MSVE vs Gradient Steps ---
        ax_msve_step.plot(x_steps, mean_msve, label=name, color=color, lw=lw, ls=linestyle)
        ax_msve_step.fill_between(x_steps, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.14 if is_matched_mc else 0.18)

        # --- Panel 2: MSVE vs Active Rollout Transitions ---
        ax_msve_sample.plot(x_samples, mean_msve, label=name, color=color, lw=lw, ls=linestyle)
        ax_msve_sample.fill_between(x_samples, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.14 if is_matched_mc else 0.18)

        # --- Panel 3: MSVE vs Estimated LLM FLOPs (or Wall-Clock Time) ---
        ax_msve_compute.plot(x_compute, mean_msve, label=name, color=color, lw=lw, ls=linestyle)
        ax_msve_compute.fill_between(x_compute, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.14 if is_matched_mc else 0.18)

        # --- Panel 4: Adv Cos Sim vs Gradient Steps ---
        ax_adv_step.plot(x_steps, mean_adv, label=name, color=color, lw=lw, ls=linestyle)
        ax_adv_step.fill_between(x_steps, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.14 if is_matched_mc else 0.18)

        # --- Panel 5: Adv Cos Sim vs Active Rollout Transitions ---
        ax_adv_sample.plot(x_samples, mean_adv, label=name, color=color, lw=lw, ls=linestyle)
        ax_adv_sample.fill_between(x_samples, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.14 if is_matched_mc else 0.18)

        # --- Panel 6: Adv Cos Sim vs Estimated LLM FLOPs (or Wall-Clock Time) ---
        ax_adv_compute.plot(x_compute, mean_adv, label=name, color=color, lw=lw, ls=linestyle)
        ax_adv_compute.fill_between(x_compute, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.14 if is_matched_mc else 0.18)

    # Non-parametric reference baselines (e.g. Pure Vine MC Estimator with no network/backprop)
    ref_baselines = results.get("reference_baselines", {})
    for ref_idx, (ref_name, ref_data) in enumerate(ref_baselines.items()):
        color = get_color(ref_name, len(method_names) + ref_idx)
        ls_ref = ":" if "K=2" in ref_name else "-."
        r_msve = np.asarray(ref_data["msve"])
        r_adv = np.asarray(ref_data["advantage_cosine_sim"])
        m_msve, s_msve = float(np.mean(r_msve)), float(np.std(r_msve) / np.sqrt(len(r_msve)))
        m_adv, s_adv = float(np.mean(r_adv)), float(np.std(r_adv) / np.sqrt(len(r_adv)))

        # Direct estimation has no learning curve: show one use, never repeated spending.
        ax_msve_step.errorbar(0, m_msve, yerr=s_msve, fmt="D", color=color,
                             capsize=3, label=ref_name + " (one batch)")
        ax_adv_step.errorbar(0, m_adv, yerr=s_adv, fmt="D", color=color, capsize=3)

        def _plot_budgeted_ref(ax, x_key, mean_val, sem_val):
            # First entry also supports older files containing cumulative budgets.
            if x_key in ref_data:
                rx = np.asarray(ref_data[x_key], dtype=np.float64).reshape(-1)
                if rx.size:
                    ax.errorbar(rx[0], mean_val, yerr=sem_val, fmt="D",
                                color=color, capsize=3,
                                label=ref_name + " (one batch; generation cost)")

        _plot_budgeted_ref(ax_msve_sample, "x_samples", m_msve, s_msve)
        _plot_budgeted_ref(ax_adv_sample, "x_samples", m_adv, s_adv)
        _plot_budgeted_ref(ax_msve_compute, "x_llm_flops", m_msve, s_msve)
        _plot_budgeted_ref(ax_adv_compute, "x_llm_flops", m_adv, s_adv)

    # Subplot formatting
    for ax in [ax_msve_step, ax_msve_sample, ax_msve_compute]:
        ax.set_yscale("log")
        ax.set_ylabel("Mean Squared Value Error (MSVE)")
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)

    for ax in [ax_adv_step, ax_adv_sample, ax_adv_compute]:
        ax.set_ylabel(r"Advantage Cosine Similarity $\cos(\hat{\mathbf{A}}, \mathbf{A}^*)$")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)

    ax_msve_sample.set_xscale("log")
    ax_adv_sample.set_xscale("log")
    if has_llm_flops:
        ax_msve_compute.set_xscale("log")
        ax_adv_compute.set_xscale("log")

    if has_compute_matched and base_total_grad_steps is not None:
        ax_msve_step.set_xlim(0, base_total_grad_steps * 1.03)
        ax_adv_step.set_xlim(0, base_total_grad_steps * 1.03)

    step_label = "Cumulative Gradient Steps"
    sample_label = "Active Rollout Transitions (Log Scale)"
    compute_label = "Estimated LLM FLOPs (Units of $N$ Params, Log Scale)" if has_llm_flops else "Elapsed Wall-Clock Time (s)"
    compute_title_suffix = "Estimated LLM Compute" if has_llm_flops else "Total Compute (Seconds)"

    ax_msve_step.set_title("MSVE vs. Gradient Steps")
    ax_msve_step.set_xlabel(step_label)
    ax_msve_sample.set_title("MSVE vs. Sample Transitions")
    ax_msve_sample.set_xlabel(sample_label)
    ax_msve_compute.set_title(f"MSVE vs. {compute_title_suffix}")
    ax_msve_compute.set_xlabel(compute_label)

    ax_adv_step.set_title("Advantage Alignment vs. Gradient Steps")
    ax_adv_step.set_xlabel(step_label)
    ax_adv_sample.set_title("Advantage Alignment vs. Samples")
    ax_adv_sample.set_xlabel(sample_label)
    ax_adv_compute.set_title(f"Advantage Alignment vs. {compute_title_suffix}")
    ax_adv_compute.set_xlabel(compute_label)

    ax_msve_step.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    flop_assumptions = results.get("flop_assumptions", None)
    if has_llm_flops and flop_assumptions:
        fig.text(
            0.5, 0.012,
            flop_assumptions,
            ha="center", va="bottom", fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95),
        )
        plt.tight_layout(rect=[0, 0.055, 1, 0.95])
    else:
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
    ref_baselines = results.get("reference_baselines", {})
    ref_names = list(ref_baselines.keys())
    n_cols = len(methods) + len(ref_names)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 7.5), dpi=300, squeeze=False)
    fig.suptitle(f"Critic Calibration: True Value $V^*(s)$ vs. Predicted $\hat{{V}}_\phi(s)$", y=0.98)

    true_values = np.asarray(results["eval_true_values"])  # (num_eval_states,)

    def _draw_scatter_cell(ax, preds, color, title_str, add_ylabel, add_xlabel):
        n_seeds, _ = preds.shape
        x_pts = np.tile(true_values, n_seeds)
        y_pts = preds.flatten()

        ax.scatter(x_pts, y_pts, alpha=0.30, color=color, s=14, edgecolors="none")

        min_val = min(0.0, float(np.min(true_values)), float(np.min(preds)))
        max_val = max(1.0, float(np.max(true_values)), float(np.max(preds)))
        pad = 0.05 * (max_val - min_val + 1e-4)
        line_range = [min_val - pad, max_val + pad]
        ax.plot(line_range, line_range, "k--", lw=1.2, alpha=0.7, label=r"Ideal $y = x$")

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
        ax.set_title(title_str, fontsize=10.0)
        if add_ylabel:
            ax.set_ylabel(r"Predicted Value $\hat{V}(s)$")
        if add_xlabel:
            ax.set_xlabel(r"Ground-Truth $V^*(s)$")

    for col_idx, name in enumerate(methods):
        data = results["methods"][name]
        color = get_color(name, col_idx)
        preds_1_5 = np.asarray(data["scatter_snapshots"]["1_5"])
        preds_4_5 = np.asarray(data["scatter_snapshots"]["4_5"])

        _draw_scatter_cell(
            axes[0, col_idx], preds_1_5, color,
            f"{name}\n(1/5 Learning (20% Budget))",
            add_ylabel=(col_idx == 0), add_xlabel=False,
        )
        _draw_scatter_cell(
            axes[1, col_idx], preds_4_5, color,
            "(4/5 Learning (80% Budget))",
            add_ylabel=(col_idx == 0), add_xlabel=True,
        )

    for r_offset, ref_name in enumerate(ref_names):
        col_idx = len(methods) + r_offset
        ref_data = ref_baselines[ref_name]
        color = get_color(ref_name, col_idx)
        v_preds = np.asarray(ref_data["v_preds"])

        ax_top = axes[0, col_idx]
        ax_top.axis("off")
        ax_top.text(
            0.5, 0.5,
            f"{ref_name}\n\nNon-Parametric MC Estimator\n(No Neural Net / No Backprop)\nEvaluated Across All Eval States",
            ha="center", va="center", fontsize=9.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor=color, lw=1.4),
        )
        _draw_scatter_cell(
            axes[1, col_idx], v_preds, color,
            f"{ref_name}\n(Static Estimator)",
            add_ylabel=(col_idx == 0), add_xlabel=True,
        )

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
        if "x_llm_flops" in data:
            x_compute = np.asarray(data["x_llm_flops"])
        else:
            x_times = np.asarray(data["x_wallclock"])
            x_compute = np.mean(x_times, axis=0) if x_times.ndim == 2 else x_times

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
        auc_c_msve = compute_normalized_auc(x_compute, mean_msve)
        auc_u_msve = compute_normalized_auc(x_steps, mean_msve)

        auc_s_adv = compute_normalized_auc(x_samples, mean_adv)
        auc_c_adv = compute_normalized_auc(x_compute, mean_adv)
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

    for ref_name, ref_data in results.get("reference_baselines", {}).items():
        r_msve = np.asarray(ref_data["msve"])
        r_adv = np.asarray(ref_data["advantage_cosine_sim"])
        m_msve = float(np.mean(r_msve))
        s_msve = float(np.std(r_msve) / np.sqrt(len(r_msve)))
        m_adv = float(np.mean(r_adv))
        s_adv = float(np.std(r_adv) / np.sqrt(len(r_adv)))
        rows.append({
            "Algorithm": ref_name,
            "Final MSVE": f"{m_msve:.2e} ± {s_msve:.1e}",
            "Final AdvCos": f"{m_adv:.4f} ± {s_adv:.3f}",
            "AUC_Sample(MSVE)": "---",
            "AUC_Compute(MSVE)": "---",
            "AUC_Update(MSVE)": "---",
            "AUC_Sample(Adv)": "---",
            "AUC_Compute(Adv)": "---",
            "AUC_Update(Adv)": "---",
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


def plot_difficulty_breakdown(
    results: Dict[str, Any],
    save_dir: str,
    prefix: str = "benchmark",
) -> None:
    """
    Plots Value Error (MSVE) and Advantage Cosine Similarity as a function of
    Problem Difficulty theta in [0, 1] (or Problem Index 0 .. P-1).
    Demonstrates where Standard MC degrades on hard/needle-in-haystack problems,
    and how Vine-Target and FBI maintain value accuracy.
    """
    if "thetas" not in results or "methods" not in results:
        return
    thetas = np.asarray(results["thetas"])
    if len(thetas) <= 1:
        # Single problem, difficulty curve not applicable
        return

    num_problems = len(thetas)
    prob_indices = np.arange(num_problems)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fig.suptitle("Value Accuracy & Advantage Alignment vs. Problem Difficulty Index", y=0.98)

    ax_msve = axes[0]
    ax_adv = axes[1]

    methods = list(results["methods"].keys())
    breakdown_rows = []

    for idx, name in enumerate(methods):
        data = results["methods"][name]
        color = get_color(name, idx)

        if "msve_per_problem" not in data or "adv_cos_per_problem" not in data:
            continue

        msve_p = np.asarray(data["msve_per_problem"])        # (n_seeds, num_problems)
        adv_p = np.asarray(data["adv_cos_per_problem"])       # (n_seeds, num_problems)

        mean_msve = np.mean(msve_p, axis=0)
        sem_msve = np.std(msve_p, axis=0) / np.sqrt(msve_p.shape[0])

        mean_adv = np.mean(adv_p, axis=0)
        sem_adv = np.std(adv_p, axis=0) / np.sqrt(adv_p.shape[0])

        # Plot MSVE vs Problem Index / theta
        ax_msve.plot(prob_indices, mean_msve, "o-", label=name, color=color, lw=2.0, ms=5)
        ax_msve.fill_between(prob_indices, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.18)

        # Plot Adv Cos vs Problem Index / theta
        ax_adv.plot(prob_indices, mean_adv, "o-", label=name, color=color, lw=2.0, ms=5)
        ax_adv.fill_between(prob_indices, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.18)

        for p in range(num_problems):
            breakdown_rows.append({
                "Algorithm": name,
                "Problem_Index": p,
                "Difficulty_Theta": f"{thetas[p]:.3f}",
                "MSVE_Mean": f"{mean_msve[p]:.4e}",
                "MSVE_SEM": f"{sem_msve[p]:.4e}",
                "AdvCos_Mean": f"{mean_adv[p]:.4f}",
                "AdvCos_SEM": f"{sem_adv[p]:.4f}",
            })

    # Non-parametric reference baselines across problems
    for ref_idx, (ref_name, ref_data) in enumerate(results.get("reference_baselines", {}).items()):
        if "msve_per_problem" not in ref_data or "adv_cos_per_problem" not in ref_data:
            continue
        color = get_color(ref_name, len(methods) + ref_idx)
        ls_ref = "--" if "K=2" in ref_name else ":"
        msve_p = np.asarray(ref_data["msve_per_problem"])
        adv_p = np.asarray(ref_data["adv_cos_per_problem"])
        mean_msve = np.mean(msve_p, axis=0)
        sem_msve = np.std(msve_p, axis=0) / np.sqrt(msve_p.shape[0])
        mean_adv = np.mean(adv_p, axis=0)
        sem_adv = np.std(adv_p, axis=0) / np.sqrt(adv_p.shape[0])

        ax_msve.plot(prob_indices, mean_msve, ls=ls_ref, marker="s", label=ref_name, color=color, lw=1.8, ms=4)
        ax_msve.fill_between(prob_indices, np.maximum(1e-7, mean_msve - sem_msve), mean_msve + sem_msve, color=color, alpha=0.10)
        ax_adv.plot(prob_indices, mean_adv, ls=ls_ref, marker="s", label=ref_name, color=color, lw=1.8, ms=4)
        ax_adv.fill_between(prob_indices, mean_adv - sem_adv, mean_adv + sem_adv, color=color, alpha=0.10)

        for p in range(num_problems):
            breakdown_rows.append({
                "Algorithm": ref_name,
                "Problem_Index": p,
                "Difficulty_Theta": f"{thetas[p]:.3f}",
                "MSVE_Mean": f"{mean_msve[p]:.4e}",
                "MSVE_SEM": f"{sem_msve[p]:.4e}",
                "AdvCos_Mean": f"{mean_adv[p]:.4f}",
                "AdvCos_SEM": f"{sem_adv[p]:.4f}",
            })

    # Formatting
    ax_msve.set_yscale("log")
    ax_msve.set_xlabel("Problem Index (Increasing Difficulty $\\theta: 0 \\to 1$)")
    ax_msve.set_ylabel("Final Mean Squared Value Error (MSVE)")
    ax_msve.set_title("MSVE vs. Problem Difficulty")
    ax_msve.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax_msve.legend(loc="upper left", framealpha=0.9, fontsize=8.0)

    ax_adv.set_xlabel("Problem Index (Increasing Difficulty $\\theta: 0 \\to 1$)")
    ax_adv.set_ylabel(r"Advantage Cosine Similarity $\cos(\hat{\mathbf{A}}, \mathbf{A}^*)$")
    ax_adv.set_title("Advantage Alignment vs. Problem Difficulty")
    ax_adv.set_ylim(-0.2, 1.05)
    ax_adv.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax_adv.grid(True, ls="--", lw=0.5, alpha=0.6)

    # Add secondary x-axis with theta values (stride ticks when P > 12 for readability)
    tick_stride = 2 if num_problems > 12 else 1
    tick_idx = prob_indices[::tick_stride]
    tick_lbls = [f"$\\theta$={thetas[i]:.2f}" for i in tick_idx]

    ax_msve_top = ax_msve.secondary_xaxis("top")
    ax_msve_top.set_xticks(tick_idx)
    ax_msve_top.set_xticklabels(tick_lbls, fontsize=8, rotation=35)

    ax_adv_top = ax_adv.secondary_xaxis("top")
    ax_adv_top.set_xticks(tick_idx)
    ax_adv_top.set_xticklabels(tick_lbls, fontsize=8, rotation=35)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, f"{prefix}_difficulty_breakdown.pdf")
    png_path = os.path.join(save_dir, f"{prefix}_difficulty_breakdown.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plotting] Saved difficulty breakdown plots to:\n  - {pdf_path}\n  - {png_path}")

    # Save CSV
    if breakdown_rows:
        csv_path = os.path.join(save_dir, f"{prefix}_difficulty_breakdown.csv")
        headers = list(breakdown_rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for r in breakdown_rows:
                f.write(",".join([str(r[h]) for h in headers]) + "\n")
        print(f"[Summary] Saved difficulty breakdown metrics table to: {csv_path}")


def filter_methods(
    results: Dict[str, Any],
    include_methods: Optional[List[str]] = None,
    exclude_compute_matched: bool = True,
) -> Dict[str, Any]:
    """
    Filters results dictionary to include only designated methods.
    Excludes compute-matched MC by default.
    """
    filtered = dict(results)
    orig_methods = results.get("methods", {})
    new_methods = {}
    for name, data in orig_methods.items():
        if exclude_compute_matched and (data.get("is_compute_matched_mc", False) or "Compute-Matched" in name):
            continue
        if include_methods is not None and name not in include_methods:
            continue
        new_methods[name] = data

    if include_methods is not None:
        ordered = {}
        for m in include_methods:
            if m in new_methods:
                ordered[m] = new_methods[m]
        new_methods = ordered

    filtered["methods"] = new_methods
    return filtered


def plot_pareto_frontier(
    results: Dict[str, Any],
    save_dir: str,
    prefix: str = "benchmark",
) -> None:
    """
    Plots the Empirical Pareto Frontier:
      Panel 1: Final Advantage Cosine Sim vs. Active Rollout Transitions (Log Scale)
      Panel 2: Final Advantage Cosine Sim vs. Estimated LLM FLOPs (Log Scale)
    Highlights which algorithms dominate in sample and compute efficiency.
    """
    os.makedirs(save_dir, exist_ok=True)
    methods = list(results["methods"].keys())
    if not methods:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=300)
    env_title = results.get("env_title", "Reasoning Benchmark")
    fig.suptitle(f"Advantage Alignment vs. LLM Compute & Sample Efficiency: Pareto Frontier ({env_title})", y=0.98)

    ax_sample = axes[0]
    ax_compute = axes[1]

    has_llm_flops = all("x_llm_flops" in results["methods"][m] for m in methods)

    sample_pts = []
    compute_pts = []

    for idx, name in enumerate(methods):
        data = results["methods"][name]
        color = get_color(name, idx)

        final_samples = float(data["x_samples"][-1])
        final_compute = float(data["x_llm_flops"][-1]) if has_llm_flops else float(np.mean(data["x_wallclock"]))

        adv_arr = np.asarray(data["advantage_cosine_sim"])
        mean_adv = float(np.mean(adv_arr[:, -1]))
        sem_adv = float(np.std(adv_arr[:, -1]) / np.sqrt(adv_arr.shape[0]))

        # Panel 1: Adv vs Active Rollout Transitions
        ax_sample.errorbar(final_samples, mean_adv, yerr=sem_adv, fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=4, ms=9, label=name)
        sample_pts.append((final_samples, mean_adv, name, color))

        # Panel 2: Adv vs FLOPs / Compute
        ax_compute.errorbar(final_compute, mean_adv, yerr=sem_adv, fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=4, ms=9, label=name)
        compute_pts.append((final_compute, mean_adv, name, color))

    # Add non-parametric Vine MC reference trajectories (u=1 -> u=U) and Pareto points
    ref_baselines = results.get("reference_baselines", {})
    for pts, ax, x_key in [(sample_pts, ax_sample, "x_samples"), (compute_pts, ax_compute, "x_llm_flops")]:
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.set_ylabel(r"Final Advantage Cosine Similarity $\cos(\hat{\mathbf{A}}, \mathbf{A}^*)$")
        ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)

        all_x_vals = [p[0] for p in pts]
        all_pts_to_annotate = list(pts)

        for r_idx, (ref_name, ref_data) in enumerate(ref_baselines.items()):
            r_adv = np.asarray(ref_data["advantage_cosine_sim"])
            m_adv = float(np.mean(r_adv))
            s_adv = float(np.std(r_adv) / np.sqrt(len(r_adv)))
            c_ref = get_color(ref_name, len(methods) + r_idx)
            ls_ref = ":" if "K=2" in ref_name else "-."

            if x_key in ref_data and (x_key != "x_llm_flops" or has_llm_flops):
                rx = np.asarray(ref_data[x_key], dtype=np.float64)
                all_x_vals.extend([float(rx[0]), float(rx[-1])])
                ax.plot([rx[0], rx[-1]], [m_adv, m_adv], color=c_ref, ls=ls_ref, lw=1.6, alpha=0.8)
                ax.plot(rx[0], m_adv, marker="D", ms=6.0, mfc="white", mec=c_ref, mew=1.6, zorder=5)
                ax.errorbar(
                    rx[-1], m_adv, yerr=s_adv, fmt="D", color=c_ref, ecolor=c_ref,
                    elinewidth=1.6, capsize=4, ms=8.5, zorder=6,
                    label=f"{ref_name} ({m_adv:.4f})",
                )
                all_pts_to_annotate.append((float(rx[-1]), m_adv, ref_name, c_ref))
            else:
                ax.axhline(m_adv, color=c_ref, ls=ls_ref, lw=1.6, alpha=0.85, label=f"{ref_name} ({m_adv:.4f})")

        if all_x_vals:
            ax.set_xlim(min(all_x_vals) / 2.2, max(all_x_vals) * 2.4)

        # Smart non-overlapping annotation offsets sorted descending by y, keeping high labels above and low labels below
        sorted_ann = sorted(all_pts_to_annotate, key=lambda t: (-t[1], t[0]))
        placed_boxes = []
        y_mid = float(np.median([t[1] for t in sorted_ann])) if sorted_ann else 0.5
        for x_val, y_val, name, c in sorted_ann:
            if "Vine MC" in name:
                candidates = (
                    [(0, -16, "top", "center"), (-10, -16, "top", "right"), (0, 14, "bottom", "center")]
                    if "K=2" in name
                    else [(0, 14, "bottom", "center"), (10, 14, "bottom", "left"), (0, -16, "top", "center")]
                )
            elif y_val >= y_mid:
                candidates = [
                    (0, 14, "bottom", "center"),
                    (-14, 12, "bottom", "right"),
                    (14, 12, "bottom", "left"),
                    (0, 34, "bottom", "center"),
                    (0, -16, "top", "center"),
                    (0, -34, "top", "center"),
                ]
            else:
                candidates = [
                    (0, -16, "top", "center"),
                    (-14, -14, "top", "right"),
                    (14, -14, "top", "left"),
                    (0, -34, "top", "center"),
                    (0, 14, "bottom", "center"),
                    (0, 34, "bottom", "center"),
                ]

            lx = np.log10(max(1.0, x_val))
            half_w = max(0.24, 0.018 * len(name))
            best = candidates[0]
            for cand in candidates:
                dx_p, dy_p, va_s, ha_s = cand
                ha_shift = {"center": 0.0, "left": half_w * 0.8, "right": -half_w * 0.8}[ha_s]
                va_shift = {"center": 0.0, "bottom": 0.055, "top": -0.055}[va_s]
                cx = lx + dx_p * 0.01 + ha_shift
                cy = y_val + dy_p * 0.004 + va_shift
                collision = any(abs(cx - px) < (half_w + pw) and abs(cy - py) < 0.115 for px, py, pw in placed_boxes)
                if not collision:
                    best = cand
                    break
            dx_p, dy_p, va_s, ha_s = best
            ha_shift = {"center": 0.0, "left": half_w * 0.8, "right": -half_w * 0.8}[ha_s]
            va_shift = {"center": 0.0, "bottom": 0.055, "top": -0.055}[va_s]
            placed_boxes.append((lx + dx_p * 0.01 + ha_shift, y_val + dy_p * 0.004 + va_shift, half_w))

            ax.annotate(
                f"{name}\n({y_val:.4f})",
                xy=(x_val, y_val),
                xytext=(dx_p, dy_p),
                textcoords="offset points",
                ha=ha_s,
                va=va_s,
                fontsize=8.0,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.6) if abs(dy_p) > 20 or abs(dx_p) > 10 else None,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor=c, alpha=0.92, lw=1.1),
                zorder=7,
            )

        if ref_baselines:
            ax.legend(loc="lower right", fontsize=8.0, framealpha=0.9)

    ax_sample.set_xlabel("Total Active Rollout Transitions (Log Scale)")
    ax_sample.set_title("Advantage Alignment vs. Active Rollout Samples")

    compute_lbl = "Estimated LLM FLOPs (Units of $N$ Params, Log Scale)" if has_llm_flops else "Elapsed Wall-Clock Time (s)"
    ax_compute.set_xlabel(compute_lbl)
    ax_compute.set_title("Advantage Alignment vs. LLM FLOPs")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_path = os.path.join(save_dir, f"{prefix}_pareto_frontier.pdf")
    png_path = os.path.join(save_dir, f"{prefix}_pareto_frontier.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Plotting] Saved Pareto frontier plots to:\n  - {pdf_path}\n  - {png_path}")


def generate_analysis_suite(
    dag_data_path: str = "results/dag_full/benchmark_results.pkl",
    tree_data_path: str = "results/tree_full/benchmark_results.pkl",
    output_dir: str = "results/analysis",
) -> None:
    """
    Generates the focused comparative analysis suite across DAG Full and Tree Full:
      1. MC vs. Vine MC (K=1, 2, 4)
      2. FBI Four Implementations (Batched vs. Staged, lam=0.0 vs. 0.8)
      3. Head-to-Head: Best FBI (Batched lam=0.8) vs. Best Vine MC (Pareto Frontier)
    Excludes compute-matched MC.
    Saves in output_dir (and links to ./analysis).
    """
    os.makedirs(output_dir, exist_ok=True)
    symlink_path = os.path.abspath("analysis")
    output_abs = os.path.abspath(output_dir)
    if symlink_path != output_abs:
        if os.path.islink(symlink_path):
            os.unlink(symlink_path)
        if not os.path.exists(symlink_path):
            try:
                os.symlink(output_abs, symlink_path)
                print(f"[Analysis] Created symlink: {symlink_path} -> {output_abs}")
            except Exception as e:
                print(f"[Analysis] Could not create symlink: {e}")

    with open(dag_data_path, "rb") as f:
        dag_results = pickle.load(f)
    with open(tree_data_path, "rb") as f:
        tree_results = pickle.load(f)

    # -------------------------------------------------------------
    # 1. Compare MC, Vine MC for a few K's
    # -------------------------------------------------------------
    dir_q1 = os.path.join(output_dir, "1_mc_vs_vine")
    methods_q1 = ["Standard MC (K=1)", "Vine-Target (K=2)", "Vine-Target (K=4)"]
    print("\n" + "=" * 70)
    print("ANALYSIS PART 1: MC vs. Vine-Target (K=1, 2, 4)")
    print("=" * 70)

    for env_name, raw_data in [("dag", dag_results), ("tree", tree_results)]:
        sub_dir = os.path.join(dir_q1, env_name)
        os.makedirs(sub_dir, exist_ok=True)
        sub_results = filter_methods(raw_data, include_methods=methods_q1, exclude_compute_matched=True)
        plot_comparison_curves(sub_results, sub_dir, prefix=f"{env_name}_mc_vs_vine")
        plot_scatter_true_vs_estimated(sub_results, sub_dir, prefix=f"{env_name}_mc_vs_vine")
        compute_and_save_auc_summary(sub_results, sub_dir, prefix=f"{env_name}_mc_vs_vine")
        plot_difficulty_breakdown(sub_results, sub_dir, prefix=f"{env_name}_mc_vs_vine")

    # -------------------------------------------------------------
    # 2. Compare FBI four implementations
    # -------------------------------------------------------------
    dir_q2 = os.path.join(output_dir, "2_fbi_variants")
    methods_q2 = ["Batched FBI (lam=0.0)", "Batched FBI (lam=0.8)", "Staged FBI (lam=0.0)", "Staged FBI (lam=0.8)"]
    print("\n" + "=" * 70)
    print("ANALYSIS PART 2: FBI Four Implementations (Batched vs. Staged, lam=0.0 vs. 0.8)")
    print("=" * 70)

    for env_name, raw_data in [("dag", dag_results), ("tree", tree_results)]:
        sub_dir = os.path.join(dir_q2, env_name)
        os.makedirs(sub_dir, exist_ok=True)
        sub_results = filter_methods(raw_data, include_methods=methods_q2, exclude_compute_matched=True)
        plot_comparison_curves(sub_results, sub_dir, prefix=f"{env_name}_fbi_variants")
        plot_scatter_true_vs_estimated(sub_results, sub_dir, prefix=f"{env_name}_fbi_variants")
        compute_and_save_auc_summary(sub_results, sub_dir, prefix=f"{env_name}_fbi_variants")
        plot_difficulty_breakdown(sub_results, sub_dir, prefix=f"{env_name}_fbi_variants")

    # -------------------------------------------------------------
    # 3. Head-to-Head: Best FBI vs Best Vine MC vs TD(lambda) (Pareto Frontier)
    # -------------------------------------------------------------
    dir_q3 = os.path.join(output_dir, "3_head_to_head_best")
    methods_q3 = ["Standard MC (K=1)", "TD (lam=0.0)", "TD (lam=0.8)", "Vine-Target (K=4)", "Batched FBI (lam=0.8)"]
    pareto_candidates = ["Standard MC (K=1)", "TD (lam=0.0)", "TD (lam=0.8)", "Vine-Target (K=2)", "Vine-Target (K=4)", "Batched FBI (lam=0.8)"]
    print("\n" + "=" * 70)
    print("ANALYSIS PART 3: Head-to-Head Best FBI vs Best Vine MC")
    print("=" * 70)

    for env_name, raw_data in [("dag", dag_results), ("tree", tree_results)]:
        sub_dir = os.path.join(dir_q3, env_name)
        os.makedirs(sub_dir, exist_ok=True)
        sub_results = filter_methods(raw_data, include_methods=methods_q3, exclude_compute_matched=True)
        plot_comparison_curves(sub_results, sub_dir, prefix=f"{env_name}_head_to_head")
        plot_scatter_true_vs_estimated(sub_results, sub_dir, prefix=f"{env_name}_head_to_head")
        compute_and_save_auc_summary(sub_results, sub_dir, prefix=f"{env_name}_head_to_head")
        plot_difficulty_breakdown(sub_results, sub_dir, prefix=f"{env_name}_head_to_head")

        pareto_sub = filter_methods(raw_data, include_methods=pareto_candidates, exclude_compute_matched=True)
        plot_pareto_frontier(pareto_sub, sub_dir, prefix=f"{env_name}_head_to_head")

    print("\n[Analysis] Full analysis suite successfully generated in:", output_dir)


def plot_dag_comparisons(root, save_dir=None):
    """Unified critic/PPO panels from submit_all_dag_comparisons.sh outputs."""
    import csv
    import json
    from pathlib import Path
    root = Path(root)
    dest = Path(save_dir) if save_dir else root/'unified'
    dest.mkdir(parents=True, exist_ok=True)
    envs = ('dag','dag_quality','dag_approach')
    titles = ('Binary DAG','Quality DAG','Approach-and-error DAG')
    names = {'mc':'MC','td_lam08':'TD λ=0.8','fbi_lam08_k2':'FBI λ=0.8, K=2',
             'vine_k2':'Vine-target K=2','vine_k4':'Vine-target K=4'}
    colors = dict(zip(names, ('#1f77b4','#6366f1','#2ca02c','#ff7f0e','#d62728')))
    values, ppos, summary = {}, {}, []
    def sem(a):
        return a.std(0, ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.zeros(a.shape[1:])
    def smooth(a):
        return np.array([np.mean(a[max(0,i-4):i+1]) for i in range(len(a))])
    def band(ax,x,a,label,color):
        mean=a.mean(0);err=sem(a)
        ax.plot(x,mean,label=label,color=color)
        ax.fill_between(x,mean-err,mean+err,alpha=.13,color=color)
    for env in envs:
        with (root/'value'/f'{env}_full'/'benchmark_results.pkl').open('rb') as f:values[env]=pickle.load(f)
        ppos[env]={}
        for method in names:
            folder=root/'ppo'/env/method
            config=json.load((folder/'config.json').open())
            rows=[list(csv.DictReader(p.open())) for p in sorted(folder.glob('metrics_seed*.csv'))]
            arrays={key:np.array([[float(r[key]) for r in seed] for seed in rows]) for key in rows[0][0]}
            evals=json.load((folder/'summary.json').open())
            arrays['initial']=np.array([e['initial_success'] for e in evals])
            arrays['final']=np.array([e['final_success'] for e in evals])
            arrays['total_samples']=np.cumsum(arrays['trunk_transitions']+arrays['aux_transitions'],axis=1)
            arrays['trunk_samples']=np.cumsum(arrays['trunk_transitions'],axis=1)
            arrays['episode_length']=arrays['trunk_transitions']/config['num_envs']
            arrays['config']=config
            ppos[env][method]=arrays
            summary.append(dict(environment=env,method=method,seeds=len(rows),
                                initial_evaluation=arrays['initial'].mean(),final_evaluation=arrays['final'].mean(),
                                final_evaluation_sem=sem(arrays['final']),last10_training_success=arrays['success_rate'][:,-10:].mean(),
                                total_samples=arrays['total_samples'][:,-1].mean(),trunk_samples=arrays['trunk_samples'][:,-1].mean(),
                                final_episode_length=arrays['episode_length'][:,-10:].mean()))
    def finish(fig,stem,handles):
        fig.legend(*handles,loc='lower center',ncol=3,fontsize=9)
        fig.tight_layout(rect=(0,.12,1,.94))
        for ext in ('png','pdf'):fig.savefig(dest/f'{stem}.{ext}',dpi=170)
        plt.close(fig)
    # Critic panels share method colors across task families.
    for axis,label in (('x_updates','Gradient steps'),('x_samples','Active rollout transitions'),('x_llm_flops','Estimated compute (N-FLOPs)')):
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        for col,env in enumerate(envs):
            for idx,(name,m) in enumerate(values[env]['methods'].items()):
                for row,key in enumerate(('msve','advantage_cosine_sim')):
                    band(axs[row,col],m[axis],np.asarray(m[key]),name,get_color(name,idx))
            axs[0,col].set_title(titles[col])
            for row in range(2):
                axs[row,col].set_xlabel(label);axs[row,col].grid(alpha=.2)
                if axis!='x_updates':axs[row,col].set_xscale('log')
            axs[0,col].set_yscale('log');axs[0,col].set_ylabel('Value MSE ↓')
            axs[1,col].set_ylabel('Advantage cosine ↑')
        fig.suptitle('Fixed-policy critic learning — mean ± SEM across seeds')
        finish(fig,'critic_'+axis,axs[0,0].get_legend_handles_labels())
    # Per-seed interpolation inside observed support only; no extrapolation.
    for axis,label in (('update','PPO update'),('trunk_samples','Trunk transitions'),('total_samples','Trunk + auxiliary transitions')):
        fig,axs=plt.subplots(1,3,figsize=(15,4.8))
        for col,env in enumerate(envs):
            for method,m in ppos[env].items():
                y=np.array([smooth(a) for a in m['success_rate']]);x=m[axis]
                lo,hi=x[:,0].max(),x[:,-1].min()
                grid=np.linspace(lo,hi,200)
                aligned=np.array([np.interp(grid,xx,yy) for xx,yy in zip(x,y)])
                band(axs[col],grid,aligned,names[method],colors[method])
            axs[col].set_title(titles[col]);axs[col].set_xlabel(label);axs[col].set_ylabel('Training success rate')
            axs[col].set_ylim(-.01,1.01);axs[col].grid(alpha=.2)
            if axis!='update':axs[col].set_xscale('log')
        fig.suptitle('PPO training — 5-update trailing average; mean ± SEM across seeds')
        finish(fig,'ppo_'+axis,axs[0].get_legend_handles_labels())
    fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    for col,env in enumerate(envs):
        for i,(method,m) in enumerate(ppos[env].items()):
            axs[col].errorbar(i,m['final'].mean(),yerr=sem(m['final']),fmt='o',color=colors[method],capsize=4,label=names[method])
            axs[col].scatter(i,m['initial'].mean(),marker='x',color=colors[method])
        axs[col].set_title(titles[col]);axs[col].set_xticks([]);axs[col].set_ylabel('Held-out stochastic evaluation success')
        axs[col].set_ylim(-.01,1.01);axs[col].grid(alpha=.2)
    fig.suptitle('PPO evaluations: dots = final ± seed SEM; crosses = initial (after BC)')
    finish(fig,'ppo_evaluation',axs[0].get_legend_handles_labels())
    fig,axs=plt.subplots(3,3,figsize=(15,10))
    for col,env in enumerate(envs):
        for method,m in ppos[env].items():
            for row,key in enumerate(('episode_length','entropy','critic_loss')):
                band(axs[row,col],m['update'][0],np.array([smooth(a) for a in m[key]]),names[method],colors[method])
        axs[0,col].set_title(titles[col])
        for row,label in enumerate(('Episode length','Actor entropy','Critic training loss')):
            axs[row,col].set_ylabel(label);axs[row,col].set_xlabel('PPO update');axs[row,col].grid(alpha=.2)
    fig.suptitle('PPO diagnostics — 5-update trailing average; mean ± SEM')
    finish(fig,'ppo_diagnostics',axs[0,0].get_legend_handles_labels())
    # Common generated-transition budget across all methods/seeds in each task.
    for env in envs:
        budget=min(m['total_samples'][:,-1].min() for m in ppos[env].values())
        for record in (r for r in summary if r['environment']==env):
            m=ppos[env][record['method']]
            at_budget=[np.interp(budget,x,smooth(y)) if x[0]<=budget<=x[-1] else np.nan for x,y in zip(m['total_samples'],m['success_rate'])]
            record['common_sample_budget']=budget
            record['training_success_at_common_budget']=np.mean(at_budget)
    with (dest/'ppo_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    critic_rows=[]
    for env in envs:
        methods=values[env]['methods'];budget=min(m['x_llm_flops'][-1] for m in methods.values())
        for name,m in methods.items():
            mse=np.asarray(m['msve']);adv=np.asarray(m['advantage_cosine_sim'])
            critic_rows.append(dict(environment=env,method=name,final_msve=mse[:,-1].mean(),final_msve_sem=sem(mse[:,-1]),
                                     final_advcos=adv[:,-1].mean(),final_advcos_sem=sem(adv[:,-1]),
                                     total_samples=m['x_samples'][-1],total_compute=m['x_llm_flops'][-1],common_compute_budget=budget,
                                     msve_at_common_compute=np.interp(budget,m['x_llm_flops'],mse.mean(0)),
                                     advcos_at_common_compute=np.interp(budget,m['x_llm_flops'],adv.mean(0))))
    with (dest/'critic_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(critic_rows[0]));w.writeheader();w.writerows(critic_rows)
    print(f'Unified comparison figures and summary CSVs saved to {dest.resolve()}')


def main():
    parser = argparse.ArgumentParser(description="Consolidated Plotting and Analysis for Value Benchmark Suite")
    parser.add_argument("--data_path", type=str, default=None, help="Path to benchmark_results.pkl")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save figures and CSVs")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Filename prefix")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated subset of methods to plot")
    parser.add_argument("--include_compute_matched", action="store_true", default=False, help="Include compute-matched Standard MC baseline (default: False)")
    parser.add_argument("--analysis", action="store_true", default=False, help="Generate comprehensive analysis suite across DAG Full and Tree Full in subdir analysis")
    parser.add_argument("--dag_path", type=str, default="results/dag_full/benchmark_results.pkl", help="Path to DAG full results for analysis")
    parser.add_argument("--tree_path", type=str, default="results/tree_full/benchmark_results.pkl", help="Path to Tree full results for analysis")
    parser.add_argument("--analysis_dir", type=str, default="results/analysis", help="Output subdir for analysis suite (default: results/analysis)")
    parser.add_argument("--comparison_root", default=None, help="Extracted all-DAG value/PPO comparison directory")
    args = parser.parse_args()
    if args.comparison_root:
        plot_dag_comparisons(args.comparison_root, args.save_dir)
        return

    if args.analysis:
        generate_analysis_suite(dag_data_path=args.dag_path, tree_data_path=args.tree_path, output_dir=args.analysis_dir)
        return

    if not args.data_path:
        # Default behavior if no arguments: run analysis suite if benchmark data exists
        if os.path.exists(args.dag_path) and os.path.exists(args.tree_path):
            generate_analysis_suite(dag_data_path=args.dag_path, tree_data_path=args.tree_path, output_dir=args.analysis_dir)
            return
        else:
            parser.error("--data_path is required when not running with --analysis")

    save_dir = args.save_dir or os.path.dirname(args.data_path) or "results"
    os.makedirs(save_dir, exist_ok=True)

    with open(args.data_path, "rb") as f:
        results = pickle.load(f)

    include_list = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    results = filter_methods(results, include_methods=include_list, exclude_compute_matched=not args.include_compute_matched)

    plot_comparison_curves(results, save_dir, prefix=args.prefix)
    plot_scatter_true_vs_estimated(results, save_dir, prefix=args.prefix)
    compute_and_save_auc_summary(results, save_dir, prefix=args.prefix)
    plot_difficulty_breakdown(results, save_dir, prefix=args.prefix)
    plot_pareto_frontier(results, save_dir, prefix=args.prefix)
    plot_quality_breakdown(results, save_dir)



def plot_quality_breakdown(results, save_dir):
    """Final critic errors by latent-state groups and remaining horizon."""
    import csv
    approach = any("msve_by_approach" in v for v in results["methods"].values())
    groups = ("approach", "error_status", "remaining_horizon") if approach else ("quality", "remaining_horizon")
    stem = "benchmark_state_breakdown" if approach else "benchmark_quality_breakdown"
    methods = {k:v for k,v in results["methods"].items() if "msve_by_"+groups[0] in v}
    if not methods:
        return
    fig,axes = plt.subplots(2,len(groups),figsize=(6.5*len(groups),8))
    with open(os.path.join(save_dir,stem+'.csv'),'w') as f:
        writer=csv.writer(f)
        writer.writerow(['Algorithm','Stratum','Index','Eval_Count','MSVE_Mean','MSVE_SEM','AdvCos_Mean','AdvCos_SEM'])
        for col,key in enumerate(groups):
            for label,method in methods.items():
                count=np.asarray(method['count_by_'+key])[0,-1]
                mse=np.asarray(method['msve_by_'+key])[:,-1]
                adv=np.asarray(method['adv_cos_by_'+key])[:,-1]
                mean=mse.mean(0); amean=adv.mean(0)
                se=mse.std(0)/np.sqrt(len(mse)); ase=adv.std(0)/np.sqrt(len(adv))
                idx=np.flatnonzero(count>0)
                categorical = key in ("approach", "error_status")
                style = dict(marker="o", linestyle="none") if categorical else {}
                axes[0,col].plot(idx,mean[idx],label=label,**style)
                axes[1,col].plot(idx,amean[idx],label=label,**style)
                for i in idx: writer.writerow([label,key,i,int(count[i]),mean[i],se[i],amean[i],ase[i]])
            for row in range(2):
                axes[row,col].set_xlabel(key.replace('_',' ').title())
                axes[row,col].set_ylabel('Value MSE' if row==0 else 'Advantage cosine similarity')
                axes[row,col].grid(alpha=.25)
        if approach:
            for row in range(2):
                approaches = results.get("environment_config", {}).get("num_approaches", 3)
                axes[row,0].set_xticks(np.arange(approaches))
                axes[row,1].set_xticks([0,1,2], ["Sound", "Slip", "Misconception"])
        axes[0,0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,stem+'.png'),dpi=160)
        fig.savefig(os.path.join(save_dir,stem+'.pdf'))
        plt.close(fig)


if __name__ == "__main__":
    main()
