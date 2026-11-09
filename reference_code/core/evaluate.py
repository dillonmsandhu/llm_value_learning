# This file is responsible for running a single training run, called by runner.py
from core.utils import save_results, save_plot, save_multi_plot, save_feature_spectra, save_heatmap, save_heatmap_stack
import os
import numpy as np
import jax
import jax.numpy as jnp


def evaluate(run_config, make_train, run_dir, args, rng):
    # Setup specific to this run_config
    steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]

    # JIT the train function for this specific config (important if env changes)
    run_fn = jax.jit(jax.vmap(make_train(run_config)))

    rngs = jax.random.split(rng, run_config["N_SEEDS"])
    out = run_fn(rngs)
    metrics = out["metrics"]
    ret = metrics.get("returned_discounted_episode_returns", 0.0)
    print(f"[{run_config['ENV_NAME']}] Mean return: {jnp.mean(ret):.4f}")
    print(f"[{run_config['ENV_NAME']}] Max return:  {jnp.max(ret):.4f}")

    # Directory structure: results/run_dir/timestamp/EnvName-Size/
    base_env_name = run_config["ENV_NAME"]
    env_size = run_config.get("ENV_SIZE")

    # Create the full name (e.g., DeepSea-bsuite-45)
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name

    env_dir = os.path.join(run_dir, full_env_name)

    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {full_env_name} results to {env_dir}")

    # Ensure save_results uses the full name for the filename
    if args.save_checkpoint:
        save_results(out, run_config, full_env_name, env_dir)
    elif args.save_metrics:
        save_results(metrics, run_config, full_env_name, env_dir)
    else:  # save config only
        save_results(run_config, run_config, full_env_name, env_dir)

    # --- Helper for Metrics extraction ---
    def _mean_over_seeds(data):
        arr = jnp.asarray(data)
        if arr.ndim > 0 and arr.shape[0] == run_config["N_SEEDS"]:
            arr = arr.mean(0)
        return arr

    def _extract_series(data):
        arr = _mean_over_seeds(data)
        arr = jnp.squeeze(arr)
        if arr.ndim == 0:
            return arr[None]
        if arr.ndim == 1:
            return arr
        return None

    def get_metric(name, slice_idx=0):
        if name not in metrics:
            return None
        series = _extract_series(metrics[name])
        if series is None:
            return None
        return series[slice_idx:]

    standard_plots = {
        "v_pred": "v_pred",
        "mean_rew": "mean_rew",
        "returned_episode_returns": "returned_episode_returns",
        "returned_discounted_episode_returns": "returned_discounted_episode_returns",
        "effective_rank": "effective_rank",
        "nn_lstd_diff": "nn_lstd_diff",
        "forward_loss": "forward_loss",
        "done_loss": "done_loss",
        "reward_loss": "reward_loss",
        "vic_loss_cov": "vic_loss_cov",
        "vic_loss_var": "vic_loss_var",
        "v_loss": "v_loss",
        "NTK_rank": "NTK_rank",
        "Direlechet_energy": "Direlechet_energy",
        "V_start": "V_start",
        "v_pred_start": "v_pred_start",
        "Mean_A": "Mean_A",
        "entropy": "entropy",
        "V_mean": "V_mean",
        "SA_min_eigenvalue": "SA_min_eigenvalue",
        "non_reversible_coeff": "non_reversible_coeff",
        "nn_greedy_performance": "nn_greedy_performance",
        "nn_advantage_cossim_uniform": "nn_advantage_cossim_uniform",
        "nn_advantage_cossim": "nn_advantage_cossim",
        "policy_tv": "policy_tv",
        "policy_tv_on_policy": "policy_tv_on_policy",
        "policy_tv_max": "policy_tv_max",
        "mu_tv": "mu_tv",
        "state_coverage": "state_coverage",
        "state_entropy_coverage": "state_entropy_coverage",
        "sigma_v_sq_td0": "sigma_v_sq_td0",
        "sigma_v_td0": "sigma_v_td0",
        "snr_v_td0": "snr_v_td0",
        "rho_v_td0": "rho_v_td0",
        "sigma_v_sq_e0": "sigma_v_sq_e0",
        "sigma_v_e0": "sigma_v_e0",
        "snr_v_e0": "snr_v_e0",
        "rho_v_e0": "rho_v_e0",
        "sigma_v_sq_td_lambda": "sigma_v_sq_td_lambda",
        "sigma_v_td_lambda": "sigma_v_td_lambda",
        "snr_v_td_lambda": "snr_v_td_lambda",
        "rho_v_td_lambda": "rho_v_td_lambda",
        "sigma_v_sq_mc": "sigma_v_sq_mc",
        "sigma_v_mc": "sigma_v_mc",
        "snr_v_mc": "snr_v_mc",
        "rho_v_mc": "rho_v_mc",
        "sigma_v_sq_e_lambda": "sigma_v_sq_e_lambda",
        "sigma_v_e_lambda": "sigma_v_e_lambda",
        "snr_v_e_lambda": "snr_v_e_lambda",
        "rho_v_e_lambda": "rho_v_e_lambda",
        "E_exact": "E_exact",
    }

    data = get_metric("E", 1)
    E_local = get_metric("E_local", 1)
    # a few log plots:
    save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, data, "E", True)
    save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, E_local, "E_local", True)
    try:
        data = get_metric("alignment_condition", 1)
        save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, E_local, "E_local", True)
        save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, data, "E", True)
    except:
        data = get_metric("alignment_condition", 1)
        save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, E_local, "E_local", False)
        save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, data, "E", False)

    try:
        save_heatmap(env_dir, run_config["ENV_NAME"], metrics["eNTK"][0, -1], "ntk")
    except Exception as e:
        print("failed to save ntk", e)

    try:
        save_heatmap(
            env_dir, run_config["ENV_NAME"], metrics["gradient_covariance_matrix"][0, -1], "gradient_covariance_matrix"
        )
    except Exception as e:
        print("failed to save gradient_covariance_matrix", e)

    try:
        grid_data = metrics.get("V_grid", metrics.get("value_grid"))
        if grid_data is not None:
            save_heatmap(env_dir, run_config["ENV_NAME"], grid_data[0, -1], "V_grid")
    except Exception as e:
        print("failed to save value grid", e)

    try:
        if "nn_grid" in metrics:
            save_heatmap(env_dir, run_config["ENV_NAME"], metrics["nn_grid"][0, -1], "nn_grid")
    except Exception as e:
        print("failed to save nn grid", e)

    try:
        if "state_dist_grid" in metrics:
            save_heatmap(env_dir, run_config["ENV_NAME"], metrics["state_dist_grid"][0, -1], "state_dist_grid")
    except Exception as e:
        pass

    try:
        if "min_eigenvector_grid" in metrics:
            save_heatmap(env_dir, run_config["ENV_NAME"], metrics["min_eigenvector_grid"][0, -1], "min_eigenvector_grid")
    except Exception as e:
        pass

    # Variance spatial heatmaps
    for grid_name in ["var_grid_td0", "var_grid_td_lambda", "var_grid_mc", "var_grid_e_lambda"]:
        if grid_name in metrics:
            try:
                save_heatmap(env_dir, run_config["ENV_NAME"], metrics[grid_name][0, -1], grid_name)
            except Exception as e:
                pass

    try:
        save_heatmap_stack(
            env_dir, run_config["ENV_NAME"], metrics["Jacobian_top_singular_vectors"][0][-1], "J Top Singular Vs"
        )
        save_heatmap_stack(
            env_dir, run_config["ENV_NAME"], metrics["feature_top_singular_vectors"][0][-1], "Feature Top Singular Vs"
        )

    except Exception as e:
        print("failed to save top five jacovian left singular vectors", e)

    try:
        save_feature_spectra(
            env_dir,
            run_config["ENV_NAME"],
            metrics["feature_singular_values"][0][0],
            metrics["feature_singular_values"][0][-1],
            "Feature Singuar Vals",
        )

        save_feature_spectra(
            env_dir,
            run_config["ENV_NAME"],
            metrics["jacobian_singular_values"][0][0],
            metrics["jacobian_singular_values"][0][-1],
            "J Singular Vals",
        )
    except Exception as e:
        print("failed to plot singular value spectrum", e)

    for m_key, save_name in standard_plots.items():
        data = get_metric(m_key, 1)
        if data is not None:
            try:
                save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, data, save_name)
            except:
                print("failed to save plot for", m_key)

    # 1. Add the ylabel string to each configuration tuple
    plot_configs = [
        (
            "Weighted Value Errors",
            "MSVE (mu-weighted)",  # <--- New Y-Label
            {
                "LSTD_weighted_VE": "LSTD (on-policy) VE",
                "VR_weighted_VE": "VR (on-policy) VE",
                "nn_weighted_VE": "NN (on-policy) VE",
                "BR_VE": "BR (on-policy) VE",
            },
            True,
        ),
        (
            "Value Learning Greedy Accuracy",
            "Greedy Accuracy",  # <--- New Y-Label
            {
                "LSTD_greedy_correct": "LSTD Greedy Acc.",
                "VR_greedy_correct": "VR Greedy Acc.",
                "nn_greedy_correct": "Network Greedy Acc.",
                "BR_greedy_correct": "BR Greedy Acc.",
            },
            False,
        ),
        (
            "Function Space Update Variance",
            "sigma_v_sq",
            {
                "sigma_v_sq_td0": "TD(0)",
                "sigma_v_sq_e0": "E(0)",
                "sigma_v_sq_td_lambda": "TD(lambda)",
                "sigma_v_sq_e_lambda": "E(lambda)",
                "sigma_v_sq_mc": "MC",
            },
            True,
        ),
        (
            "Update Directional Alignment",
            "rho_v (Cosine Similarity)",
            {
                "rho_v_td0": "TD(0)",
                "rho_v_e0": "E(0)",
                "rho_v_td_lambda": "TD(lambda)",
                "rho_v_e_lambda": "E(lambda)",
                "rho_v_mc": "MC",
            },
            False,
        ),
        # (
        #     "E Uncorrected Comparison",
        #     "E Loss",
        #     {
        #         "E_exact": "Exact E",
        #         "E_exp_uncorr": "Exp Uncorr E",
        #         "E_samp_uncorr": "Samp Uncorr E",
        #     },
        #     True,
        # ),
        # (
        #     "E Corrected Comparison",
        #     "E Loss",
        #     {
        #         "E_exact": "Exact E",
        #         "E_exp_corr": "Exp Corr E",
        #         "E_samp_corr": "Samp Corr E",
        #     },
        #     True,
        # ),
        (
            "Sampling Gradient Correlation",
            "Cosine Similarity (rho)",
            {
                "rho_samp_uncorr_exact": "Sept 14 (Uncorrected)",
                "rho_samp_corr_exact": "Oct 5 (Corrected)",
                "rho_samp_oct8_exact": "Oct 8 (Start-Anchored)",
            },
            False,
        ),
    ]

    # 2. Unpack title, ylabel, and metric_keys'
    for title, ylabel, metric_keys, logscale in plot_configs:
        plot_data = {legend: get_metric(m_key, 1) for m_key, legend in metric_keys.items()}
        plot_data = {k: v for k, v in plot_data.items() if v is not None}
        if len(plot_data) > 0:
            save_multi_plot(
                env_dir=env_dir,
                env_name=run_config["ENV_NAME"],
                steps_per_pi=steps_per_pi,
                metrics_dict=plot_data,
                title=title,
                ylabel=ylabel,
                log_scale=logscale,
            )

    if "policy_tv" in metrics:
        policy_tv_dict = {
            "Uniform Mean": get_metric("policy_tv", 1),
            "On-Policy Mean": get_metric("policy_tv_on_policy", 1),
            "Max State": get_metric("policy_tv_max", 1),
        }
        # Filter out None values if any
        policy_tv_dict = {k: v for k, v in policy_tv_dict.items() if v is not None}
        if len(policy_tv_dict) > 0:
            save_multi_plot(
                env_dir=env_dir,
                env_name=run_config["ENV_NAME"],
                steps_per_pi=steps_per_pi,
                metrics_dict=policy_tv_dict,
                title="Policy Total Variation",
                ylabel="Total Variation Distance",
                log_scale=False,
            )

    # Composite 5-subplot figure for Sampling E Test (Sept 14 vs Oct 5 vs Oct 8)
    if get_metric("E_samp_corr") is not None:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 5, figsize=(25, 4.5), dpi=150)

            # Subplot 1: Returns
            ret = get_metric("V_start", 1)
            if ret is None:
                ret = get_metric("returned_episode_returns", 1)
            if ret is not None:
                x_ret = [i * steps_per_pi for i in range(len(ret))]
                axes[0].plot(x_ret, ret, 'o-', color='tab:blue', lw=2)
            axes[0].set_title(f"Return ({run_config['ENV_NAME']})", fontweight="bold")
            axes[0].set_xlabel("Env Steps")
            axes[0].set_ylabel("Discounted Return")
            axes[0].grid(True, alpha=0.3)

            # Subplot 2: E Objective Values (Exact vs Sept 14 vs Oct 5 vs Oct 8)
            e_exact = get_metric("E_exact", 1)
            e_samp_u = get_metric("E_samp_uncorr", 1)
            e_samp_c = get_metric("E_samp_corr", 1)
            e_samp_o = get_metric("E_samp_oct8", 1)
            if e_exact is not None:
                x_e = [i * steps_per_pi for i in range(len(e_exact))]
                axes[1].plot(x_e, e_exact, 'k-', label="Exact E", linewidth=2.2)
                if e_samp_u is not None:
                    axes[1].plot(x_e, e_samp_u, '--', color='#1f77b4', lw=1.8, label="Sampled Sept 14")
                if e_samp_c is not None:
                    axes[1].plot(x_e, e_samp_c, ':', color='#d62728', lw=1.8, label="Sampled Oct 5")
                if e_samp_o is not None:
                    axes[1].plot(x_e, e_samp_o, '-.', color='#2ca02c', lw=1.8, label="Sampled Oct 8")
                axes[1].set_yscale('log')
                axes[1].set_title("E Objective Values", fontweight="bold")
                axes[1].set_xlabel("Env Steps")
                axes[1].set_ylabel("E Loss (log scale)")
                axes[1].legend(fontsize=8)
                axes[1].grid(True, alpha=0.3)

            # Subplot 3: Gradient Cosine Alignment to Exact E
            rho_samp_u = get_metric("rho_samp_uncorr_exact", 1)
            rho_samp_c = get_metric("rho_samp_corr_exact", 1)
            rho_samp_o = get_metric("rho_samp_oct8_exact", 1)
            if rho_samp_u is not None:
                x_rho = [i * steps_per_pi for i in range(len(rho_samp_u))]
                axes[2].plot(x_rho, rho_samp_u, label="Sept 14 (Uncorr)", color='#1f77b4', lw=1.8)
                if rho_samp_c is not None:
                    axes[2].plot(x_rho, rho_samp_c, label="Oct 5 (Corr)", color='#d62728', lw=1.8)
                if rho_samp_o is not None:
                    axes[2].plot(x_rho, rho_samp_o, label="Oct 8 (Start-Anc)", color='#2ca02c', lw=2.0)
                axes[2].axhline(1.0, color='black', ls=':', alpha=0.6)
                axes[2].set_title("Gradient Alignment (Cosine)", fontweight="bold")
                axes[2].set_xlabel("Env Steps")
                axes[2].set_ylabel("Cosine Similarity (rho)")
                axes[2].set_ylim(-0.3, 1.05)
                axes[2].legend(fontsize=8)
                axes[2].grid(True, alpha=0.3)

            # Subplot 4: Gradient MSE to Exact E
            mse_u = get_metric("mse_samp_uncorr", 1)
            mse_c = get_metric("mse_samp_corr", 1)
            mse_o = get_metric("mse_samp_oct8", 1)
            if mse_u is not None:
                x_mse = [i * steps_per_pi for i in range(len(mse_u))]
                axes[3].plot(x_mse, mse_u, label="Sept 14 (Uncorr)", color='#1f77b4', lw=1.8)
                if mse_c is not None:
                    axes[3].plot(x_mse, mse_c, label="Oct 5 (Corr)", color='#d62728', lw=1.8)
                if mse_o is not None:
                    axes[3].plot(x_mse, mse_o, label="Oct 8 (Start-Anc)", color='#2ca02c', lw=2.0)
                axes[3].set_yscale('log')
                axes[3].set_title("Gradient MSE to Exact E", fontweight="bold")
                axes[3].set_xlabel("Env Steps")
                axes[3].set_ylabel("MSE (log scale)")
                axes[3].legend(fontsize=8)
                axes[3].grid(True, alpha=0.3)

            # Subplot 5: Gradient Variance (Spread across trajectories)
            var_traj_u = get_metric("var_traj_samp_uncorr", 1)
            if var_traj_u is None:
                var_traj_u = get_metric("var_samp_uncorr", 1)
            var_traj_c = get_metric("var_traj_samp_corr", 1)
            if var_traj_c is None:
                var_traj_c = get_metric("var_samp_corr", 1)
            var_traj_o = get_metric("var_traj_samp_oct8", 1)
            if var_traj_o is None:
                var_traj_o = get_metric("var_samp_oct8", 1)

            if var_traj_u is not None:
                v_vals = [
                    float(var_traj_u[-1]),
                    float(var_traj_c[-1]) if var_traj_c is not None else 0.0,
                    float(var_traj_o[-1]) if var_traj_o is not None else 0.0,
                ]
                labels = ["Sept 14", "Oct 5", "Oct 8"]
                colors = ["#1f77b4", "#d62728", "#2ca02c"]
                x_pos = np.arange(len(labels))
                axes[4].bar(x_pos, v_vals, color=colors, alpha=0.85, edgecolor="black", lw=0.8)
                axes[4].set_xticks(x_pos)
                axes[4].set_xticklabels(labels, fontweight="bold")
                axes[4].set_yscale('log')
                axes[4].set_title("Trajectory Gradient Variance (Final)\nSpread Across Trajectories", fontweight="bold")
                axes[4].set_ylabel("Variance (log scale)")
                axes[4].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            single_env_plot_path = os.path.join(env_dir, "E_sampling_summary.png")
            plt.savefig(single_env_plot_path, dpi=200)
            plt.close()
            print(f"Saved single-env sampling E summary plot to {single_env_plot_path}")
        except Exception as e:
            print("Failed to save single-env sampling E summary plot:", e)

    if hasattr(args, "save_video") and args.save_video:
        try:
            # Extract the single-seed train state (seed 0)
            train_state_seed0 = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
            from core.video import generate_policy_video
            generate_policy_video(
                run_config=run_config,
                train_state=train_state_seed0,
                env_dir=env_dir,
                seed=run_config.get("SEED", 42),
                save_name="policy.gif"
            )
        except Exception as e:
            print("Failed to generate policy video:", e)
