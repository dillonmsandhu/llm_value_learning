from __future__ import annotations
import jax
import jax.numpy as jnp


def compute_runtime_metrics(
    train_state,
    network,
    traj_batch,
    loss_info,
    start_obs=None,
    config=None,
    evaluator=None,
) -> dict[str, jax.Array]:
    """
    Computes runtime metrics for sampled RL algorithms (PPO, etc.).
    
    Tracked metrics:
        - returned_episode_returns: Undiscounted episode return (primary optimization metric)
        - returned_discounted_episode_returns: Discounted episode return (G_0)
        - returned_episode_lengths: Episode length
        - V_start: Analytical true policy start-state value (computed when evaluator is enabled)
        - v_pred_start: Critic value prediction on initial observation(s) (diagnostic only, not policy performance)
        - total_loss, value_loss, actor_loss, entropy: Training loss components
        - approx_kl: Approximate KL divergence between old rollout policy and updated policy
        - clip_fraction: Fraction of policy ratio clips in PPO
        - Exact Bellman error and MSVE metrics if evaluator is provided
    """
    if config is None:
        config = {}

    # 1. Episode Returns and Durations from LogWrapper
    info = traj_batch.info
    if "returned_episode" in info and "returned_episode_returns" in info:
        done_mask = info["returned_episode"].astype(jnp.float32)
        n_done = jnp.sum(done_mask)
        has_done = n_done > 0

        returned_episode_returns = jnp.where(
            has_done,
            jnp.sum(info["returned_episode_returns"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_episode_returns"].mean(),
        )
        returned_discounted_episode_returns = jnp.where(
            has_done,
            jnp.sum(info["returned_discounted_episode_returns"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_discounted_episode_returns"].mean(),
        )
        returned_episode_lengths = jnp.where(
            has_done,
            jnp.sum(info["returned_episode_lengths"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_episode_lengths"].mean(),
        )
    elif "returned_episode_returns" in info:
        returned_episode_returns = info["returned_episode_returns"].mean()
        returned_discounted_episode_returns = info.get(
            "returned_discounted_episode_returns", returned_episode_returns
        ).mean()
        returned_episode_lengths = info.get("returned_episode_lengths", jnp.array(0.0)).mean()
    else:
        returned_episode_returns = traj_batch.reward.sum(axis=0).mean()
        returned_discounted_episode_returns = returned_episode_returns
        returned_episode_lengths = jnp.array(traj_batch.obs.shape[0], dtype=jnp.float32)

    # 2. Start-State Value Prediction (V_start)
    if start_obs is not None:
        v_start_pred = network.apply(train_state.params, start_obs, method=network.value).mean()
    else:
        v_start_pred = traj_batch.value[0].mean()

    # 3. Loss Info Handling (supports tuple or dict)
    if isinstance(loss_info, dict):
        total_loss = loss_info.get("total_loss", jnp.array(0.0)).mean()
        value_loss = loss_info.get("value_loss", jnp.array(0.0)).mean()
        actor_loss = loss_info.get("actor_loss", jnp.array(0.0)).mean()
        entropy = loss_info.get("entropy", jnp.array(0.0)).mean()
    else:
        total_loss = loss_info[0].mean()
        value_loss = loss_info[1].mean()
        actor_loss = loss_info[2].mean()
        entropy = loss_info[3].mean()

    # 4. Policy Diagnostics on Rollout Batch
    pi = network.apply(train_state.params, traj_batch.obs, method=network.policy)
    new_log_prob = pi.log_prob(traj_batch.action)
    log_ratio = new_log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio)

    # Approximate KL divergence: E[exp(log_ratio) - 1 - log_ratio]
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

    # Clip fraction
    clip_eps = config.get("CLIP_EPS", 0.2)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    metrics = {
        "returned_episode_returns": returned_episode_returns,
        "returned_discounted_episode_returns": returned_discounted_episode_returns,
        "returned_episode_lengths": returned_episode_lengths,
        "mean_rew": traj_batch.reward.mean(),
        "v_pred_start": v_start_pred,
        "v_pred": traj_batch.value.mean(),
        "total_loss": total_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }

    # Automatically add any auxiliary loss metrics (e.g. magnitude_loss, laplacian_loss, dirichlet_loss)
    if isinstance(loss_info, dict):
        for k, v in loss_info.items():
            if k not in metrics:
                metrics[k] = v.mean()

    # Pass through any custom scalar environment info keys if present
    for k, v in info.items():
        if k not in [
            "real_next_obs",
            "real_next_state",
            "is_timeout",
            "returned_episode",
            "returned_episode_returns",
            "returned_discounted_episode_returns",
            "returned_episode_lengths",
        ]:
            metrics[k] = v.mean()

    # 5. Exact Tabular Evaluator Metrics (if evaluator provided and enabled)
    if evaluator is not None and config.get("CALC_TRUE_VALUES", False):
        import core.bellman_error as bellman_error
        value_metrics = bellman_error.value_metrics(
            evaluator, network, train_state.params, random_policy=False, light=config.get("LIGHT_METRICS", True)
        )
        metrics.update(value_metrics)

        if config.get("LOG_FEATURE_METRICS", False):
            from core.feature_metrics import feature_metrics
            metrics.update(feature_metrics(
                evaluator, network, train_state.params, random_policy=False
            ))

        if hasattr(evaluator, "obs_stack"):
            s_states = evaluator.obs_stack
            v_pred_all = network.apply(train_state.params, s_states, method=network.value)
            if hasattr(evaluator, "start_idx"):
                metrics["v_pred_start"] = v_pred_all[evaluator.start_idx]

    return metrics
