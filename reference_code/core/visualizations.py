import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from core.utils import load_run_data_from_path

def generate_and_save_grid_gif(run_info, save_gif=True, fps=5, max_frames=50, display_inline=False, metrics=None):
    """
    Loads run metrics and generates a 1x4 animation comparing:
    1. State Distribution Grid
    2. True Value Grid (V)
    3. Predicted Value Grid (v)
    4. Signed Value Error Grid (v - V, overestimation > 0)

    Parameters:
    - run_info (dict): Dict containing 'name', 'path', 'seed_idx', and 'gif_output'.
    - save_gif (bool): Whether to export the animation as a GIF file.
    - fps (int): Frames per second for the output GIF.
    - max_frames (int): Maximum frames to include (sub-sampled for efficiency).
    - display_inline (bool): Whether to render interactive JS HTML animation.
    """
    run_name = run_info.get("name", "Run")
    seed_idx = run_info.get("seed_idx", 0)
    gif_path = run_info.get("gif_output", f"{run_name.lower().replace(' ', '_')}.gif")
    
    if metrics is None:
        path = run_info.get("path")
        loaded = False
        if path:
            if os.path.exists(path):
                config, metrics = load_run_data_from_path(path)
                loaded = True
                    
        if not loaded:
            raise FileNotFoundError(f"Could not locate run data for {run_name} at path '{path}'")

    # Extract required grid arrays
    state_dist = metrics.get("state_dist_grid", metrics.get("stat_dist"))
    v_true = metrics.get("V_grid", metrics.get("value_grid"))
    v_pred = metrics.get("nn_grid")

    if state_dist is None or v_true is None or v_pred is None:
        raise ValueError(
            f"Missing required grid metrics for {run_name}. "
            f"Ensure run has complete metrics (LIGHT_METRICS=False)."
        )

    # Index seed if 4D tensor (seeds, updates, height, width)
    if state_dist.ndim == 4:
        state_dist = state_dist[seed_idx]
    if v_true.ndim == 4:
        v_true = v_true[seed_idx]
    if v_pred.ndim == 4:
        v_pred = v_pred[seed_idx]

    num_updates = len(state_dist)

    # Broadcast v_true if static (2D)
    if v_true.ndim == 2:
        v_true = np.tile(v_true[None, :, :], (num_updates, 1, 1))

    # Subsample frames if length exceeds max_frames
    if max_frames and num_updates > max_frames:
        frame_indices = np.linspace(0, num_updates - 1, max_frames, dtype=int)
        state_dist = state_dist[frame_indices]
        v_true = v_true[frame_indices]
        v_pred = v_pred[frame_indices]
        num_updates = len(frame_indices)

    # Compute Signed Value Error: v_pred - v_true (overestimation is positive)
    signed_error = v_pred - v_true

    # Color limits for True & Predicted Value Grids
    v_min = float(min(np.min(v_true), np.min(v_pred)))
    v_max = float(max(np.max(v_true), np.max(v_pred)))
    if v_min == v_max:
        v_min, v_max = v_min - 1.0, v_max + 1.0

    # Independent symmetric color limits for Signed Value Error (centered at 0)
    err_abs_max = float(np.max(np.abs(signed_error)))
    if err_abs_max == 0:
        err_abs_max = 1.0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(f"{run_name} (Seed {seed_idx})", fontsize=15, fontweight='bold', y=1.03)

    # 1. State Distribution Grid
    im1 = ax1.imshow(state_dist[0], cmap="viridis", animated=True)
    ax1.set_title("State Distribution Grid", fontsize=12)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. True Value Grid
    im2 = ax2.imshow(v_true[0], cmap="RdBu_r", vmin=v_min, vmax=v_max, animated=True)
    ax2.set_title("True Value Grid (V)", fontsize=12)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Predicted Value Grid
    im3 = ax3.imshow(v_pred[0], cmap="RdBu_r", vmin=v_min, vmax=v_max, animated=True)
    ax3.set_title("Predicted Value Grid (v)", fontsize=12)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Signed Value Error Grid (v - V)
    im4 = ax4.imshow(signed_error[0], cmap="RdBu_r", animated=True, norm=colors.SymLogNorm(linthresh=0.01, vmin=-v_max, vmax=v_max, base=10))
    ax4.set_title("Signed Value Error (v - V)", fontsize=12)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    frame_text = fig.text(0.5, 0.01, f"Update 1 / {num_updates}", ha="center", fontsize=11, fontweight='bold')
    plt.tight_layout()

    def update(frame):
        im1.set_array(state_dist[frame])
        im1.set_clim(np.min(state_dist[frame]), np.max(state_dist[frame]))

        im2.set_array(v_true[frame])
        im3.set_array(v_pred[frame])
        im4.set_array(signed_error[frame])
        frame_text.set_text(f"Update {frame + 1} / {num_updates}")
        return im1, im2, im3, im4, frame_text

    ani = animation.FuncAnimation(fig, update, frames=num_updates, interval=1000 // fps, blit=False)

    if save_gif:
        out_dir = os.path.dirname(gif_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"[{run_name}] Saved GIF video to: {gif_path}")

    if display_inline:
        from IPython.display import HTML, display
        display(HTML(ani.to_jshtml()))

    plt.close(fig)
    return ani
