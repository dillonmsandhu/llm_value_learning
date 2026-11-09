# This file contains helpers relating to logging, checkpointing, and loading the data.
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt
from core.networks import *
import pandas as pd


def merge_hparams(config, hparams):
    """Merges a PyTree/dictionary of hyperparameters into the config."""
    if hparams is None:
        return config
    merged = config.copy()
    for k, v in hparams.items():
        merged[k] = v
        # Standard fallbacks for learning rate schedules (do not overwrite if already defined in config)
        if k == 'LR' and 'LR_END' not in hparams and config.get('LR_END') is None:
            merged['LR_END'] = v
        if k == 'ACTOR_LR' and 'ACTOR_LR_END' not in hparams and config.get('ACTOR_LR_END') is None:
            merged['ACTOR_LR_END'] = v
    return merged

def parse_config_override(config_str):
    """Parse config override from command line argument."""
    if config_str is None:
        return {}
    
    try:
        # Parse as JSON
        cfg = json.loads(config_str)
    except json.JSONDecodeError:
        try:
            import ast
            cfg = ast.literal_eval(config_str)
        except Exception as e:
            print(f"Error parsing config override: {e}")
            print("Config override should be valid JSON or Python dict, e.g.: '{\"LR\": 0.001, \"LAMBDA\": 0.0}'")
            exit(1)

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, str):
                if v.lower() == "false":
                    cfg[k] = False
                elif v.lower() == "true":
                    cfg[k] = True
    return cfg

def save_config(config, env_dir):
    config_path = os.path.join(env_dir, f"config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

def save_results(data, config, env_name, env_dir):
    # Create a subdirectory for the environment within the main run directory
    os.makedirs(env_dir, exist_ok=True)

    # Save the pickle file
    pickle_path = os.path.join(env_dir, "out.pkl")
    with open(pickle_path, 'wb') as f:
        cloudpickle.dump(data, f)
    print(f"Results saved to {pickle_path}")
        
    save_config(config, env_dir)
    print(f"Config saved to {os.path.join(env_dir, f'config.json')}")

    return env_dir

def save_plot(env_dir, env_name, steps_per_pi, episodic_return, title, logscale=False):
    y = jnp.asarray(episodic_return)
    if y.ndim == 0:
        y = y[None]
    if y.ndim != 1:
        print(f"Skipping plot {title}: expected 1D series, got shape {tuple(y.shape)}")
        return
    if y.shape[0] == 0:
        print(f"Skipping plot {title}: empty series")
        return

    plt.figure()
    x = [i * steps_per_pi for i in range(int(y.shape[0]))]
    plt.plot(x, y, 'o-', label=title)
    if logscale:
        plt.yscale('log') # Sets the scale for the active plot    
    elif "cosine" in title.lower() or "rho" in title.lower() or "cossim" in title.lower():
        plt.ylim(-1.05, 1.05)
    plt.xlabel("Env. Step")
    plt.ylabel(f"{title}")
    plt.title(env_name)
    plt.legend()

    # Save plot as a .png file in the environment directory
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def save_multi_plot(env_dir, env_name, steps_per_pi, metrics_dict, title="Value_Errors", ylabel="Mean Squared Error (MSVE)", log_scale=False):
    """
    Plots multiple 1D metrics on a single graph.
    
    Args:
        env_dir: Directory to save the plot.
        env_name: Name of the environment (for the title).
        steps_per_pi: Number of steps between data points.
        metrics_dict: Dictionary mapping metric names to 1D arrays/lists.
        title: The filename and y-axis label.
    """
    plt.figure(figsize=(10, 6))
    lines_plotted = 0

    for label, data in metrics_dict.items():
        y = jnp.asarray(data)
        
        if y.ndim == 0:
            y = y[None]
        if y.ndim != 1:
            print(f"Skipping line {label}: expected 1D series, got shape {tuple(y.shape)}")
            continue
        if y.shape[0] == 0:
            print(f"Skipping line {label}: empty series")
            continue

        x = [i * steps_per_pi for i in range(y.shape[0])]
        # Using a smaller markersize is usually better when lines overlap
        plt.plot(x, y, 'o-', label=label, markersize=3, alpha=0.8)
        lines_plotted += 1

    if lines_plotted == 0:
        print(f"Skipping plot {title}: no valid data series provided.")
        plt.close()
        return

    plt.xlabel("Env. Step")
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    elif "cosine" in ylabel.lower() or "rho" in ylabel.lower() or "cossim" in ylabel.lower() or "alignment" in title.lower():
        plt.ylim(-1.05, 1.05)
    plt.title(f"{env_name} - {title}")
    
    # Adding a grid makes it much easier to compare error asymptotes visually
    plt.grid(True, linestyle='--', alpha=0.6) 
    plt.legend()

    # Save plot
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path, bbox_inches='tight') # bbox_inches prevents cut-off legends
    plt.close()
    print(f"Plot saved to {plot_path}")

def save_heatmap(env_dir, env_name, quantity, title):
    plt.figure()
    plt.matshow(quantity, label=title, cmap = "RdBu_r")
    plt.xlabel("Env. Step")
    plt.ylabel(f"{title}")
    plt.title(env_name)
    plt.colorbar()
    plt.legend()

    # Save plot as a .png file in the environment directory
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def save_heatmap_stack(env_dir, env_name, heatmap_stack, title):
    import numpy as np
    """
    Plots a 1x5 grid of the top Jacobian singular vectors.
    
    Args:
        env_dir: Directory to save the plot.
        env_name: Name of the environment.
        heatmap_stack: array-like of shape (5, H, W).
    """
    # Create a 1x5 subplot layout
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"{env_name} - Top 5 Jacobian Singular Vectors", fontsize=16)

    for i in range(5):
        grid = heatmap_stack[i]
        
        # 1. Symmetric Scaling: Find the max absolute value so 0 is perfectly centered
        max_abs = np.max(np.abs(grid))
        if max_abs == 0:  # Prevent divide-by-zero if a component is totally flat
            max_abs = 1.0 

        ax = axes[i]
        
        # 2. Diverging Colormap: 'RdBu_r' makes positive red, negative blue, and zero white
        im = ax.matshow(grid, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        
        ax.set_title(f"Component {i+1}")
        
        # 3. Clean up spatial visualization by removing axis ticks
        ax.set_xticks([])  
        ax.set_yticks([])
        
        # Add individual colorbars since the scale of each singular vector will decay
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(env_dir, exist_ok=True)
    save_path = os.path.join(env_dir, f"jacobian_singular_vectors.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import numpy as np

def save_feature_spectra(env_dir, env_name, sv_start, sv_end, title):
    """
    Plots the singular value spectrum of the feature matrix at the start and end of training.
    
    Args:
        env_dir: Directory to save the plot.
        env_name: Name of the environment.
        sv_start: 1D array of singular values at initialization.
        sv_end: 1D array of singular values at the end of training.
    """
    plt.figure(figsize=(8, 6))
    
    # Convert JAX arrays to NumPy arrays
    S_start_np = np.array(sv_start)
    S_end_np = np.array(sv_end)
    
    indices_start = np.arange(1, len(S_start_np) + 1)
    indices_end = np.arange(1, len(S_end_np) + 1)
    
    # Plot Start (using a neutral/lighter color like gray or dashed lines)
    plt.plot(indices_start, S_start_np, marker='o', markersize=4, color='gray', 
             linestyle='--', alpha=0.7, label='Start (Initialization)')
    
    # Plot End (using a bold color)
    plt.plot(indices_end, S_end_np, marker='o', markersize=4, color='blue', 
             label='End (Trained)')
    
    # Log scale is critical to visualize the rank cutoff
    plt.yscale('log')
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value Magnitude (Log Scale)")
    plt.title(f"{title}")
    
    # Grid lines and legend
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Save the figure
    os.makedirs(env_dir, exist_ok=True)
    save_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def load_run_data(run_folder_name, env_name, results_base_path="results"):
    """
    Load the configuration and output data for a run given the run folder and environment.
    
    Parameters:
    - run_folder_name (str): The timestamped run folder name, e.g., "dpi_20241110_193658"
    - env_name (str): The environment name, e.g., "Asterix-MinAtar"
    - results_base_path (str): Base path to the results directory.
    
    Returns:
    - config (dict): Loaded JSON configuration.
    - results (object): Loaded output data from pickle.
    """
    # Construct paths with automatic fallback
    run_path = os.path.join(results_base_path, run_folder_name, env_name)
    if not os.path.exists(run_path):
        for candidate_base in [results_base_path, "../results", "results", os.path.join("..", results_base_path)]:
            test_path = os.path.join(candidate_base, run_folder_name, env_name)
            if os.path.exists(test_path):
                run_path = test_path
                break
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "out.pkl")
    
    # Load the config
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # Load the results
    with open(results_path, 'rb') as pkl_file:
        results = cloudpickle.load(pkl_file)
    
    
    return config, results


def load_run_data_from_path(run_path):
    """
    Load the configuration and output data for a run given the run folder and environment.
    
    Parameters:
    - run_folder_name (str): The timestamped run folder name, e.g., "dpi_20241110_193658"
    - env_name (str): The environment name, e.g., "Asterix-MinAtar"
    - results_base_path (str): Base path to the results directory, default is a sibling "results" directory.
    
    Returns:
    - config (dict): Loaded JSON configuration.
    - results (object): Loaded output data from pickle.
    """
    # Construct paths
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "out.pkl")
    
    # Load the config
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # Load the results
    with open(results_path, 'rb') as pkl_file:
        results = cloudpickle.load(pkl_file)
    
    
    return config, results