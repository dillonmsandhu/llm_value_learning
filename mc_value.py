from __future__ import annotations
import os
import json
import argparse
import datetime
import cloudpickle
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from typing import NamedTuple, Tuple, Dict, Any

from envs.tree_mrp import TreeMRPExactValue, TreeMRP
from envs.tabular_matrix_env import TabularParams
from transformer import TransformerCritic

SAVE_DIR = "mc_value"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, Any]


default_config = {
    "N_SEEDS": 1,
    "ENV_NAME": "TreeMRP",
    "DEPTH": 4,
    "P_SUCCESS": 0.8,
    "CLUSTER_SIZE": 2,
    "GAMMA": 1.0,
    "NUM_ENVS": 64,
    "NUM_STEPS": 5,        # Trajectory length (depth + 1)
    "TOTAL_TIMESTEPS": 48000,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 64,
    "LR": 0.001,
    "WARMUP_RATIO": 0.1,
    "WEIGHT_DECAY": 0.01,
    "VF_CLIP": 0.2,        # PPO-style value clipping epsilon (set <= 0 for unclipped MSE)
    "MAX_GRAD_NORM": 1.0,
    "D_MODEL": 64,
    "N_HEADS": 4,
    "N_LAYERS": 2,
    "FINAL_HIDDEN_DIM": 32,
    "SEED": 42,
}


def compute_mc_returns(traj_batch: Transition, gamma: float = 1.0) -> jnp.ndarray:
    """
    Computes backward Monte Carlo returns G_t:
      G_t = r_{t+1} + gamma * (1 - done_{t+1}) * G_{t+1}
    traj_batch fields have shape (num_steps, num_envs, ...)
    """
    def _step(next_g, step_data):
        r, done = step_data
        g = r + gamma * (1.0 - done.astype(jnp.float32)) * next_g
        return g, g

    initial_g = jnp.zeros_like(traj_batch.reward[0])
    _, targets = jax.lax.scan(
        _step, initial_g, (traj_batch.reward, traj_batch.done), reverse=True
    )
    return targets


def shuffle_and_batch(rng: jax.random.PRNGKey, data: Any, n_minibatches: int) -> Any:
    """
    Flattens the first two dimensions (num_steps, num_envs) of each array in the PyTree,
    shuffles them, and divides into n_minibatches.
    """
    def _process(x):
        x_flat = x.reshape(-1, *x.shape[2:])
        shuffled = jax.random.permutation(rng, x_flat)
        return shuffled.reshape(n_minibatches, -1, *shuffled.shape[1:])

    return jax.tree.map(_process, data)


def compute_explained_variance(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Explained variance EV = 1 - Var(y_true - y_pred) / Var(y_true)."""
    var_y = jnp.var(y_true)
    var_diff = jnp.var(y_true - y_pred)
    return jnp.where(var_y > 1e-8, 1.0 - (var_diff / var_y), 0.0)


def compute_sibling_ranking_accuracy(all_preds: jnp.ndarray, v_true_all: jnp.ndarray, num_internal: int) -> jnp.ndarray:
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
    Factory creating a 100% JIT-compilable pure-JAX Monte Carlo value learning loop.
    Structured like reference_code/ppo.py.
    """
    depth = int(config.get("DEPTH", 4))
    p_success = float(config.get("P_SUCCESS", 0.8))
    cluster_size = int(config.get("CLUSTER_SIZE", 2))
    gamma = float(config.get("GAMMA", 1.0))
    vf_clip = float(config.get("VF_CLIP", 0.2))

    # 1. Environment & Evaluator
    evaluator = TreeMRPExactValue(depth=depth, p=p_success, cluster_size=cluster_size, gamma=gamma)
    env = TreeMRP(depth=depth, p=p_success, cluster_size=cluster_size, gamma=gamma)
    env_params = TabularParams(max_steps_in_episode=depth + 2)

    # Pre-extract exact ground truth values for JIT-metrics
    v_true_all = evaluator.solve_backward_induction()[:evaluator.num_states]
    num_internal = evaluator.leaf_offset
    all_state_obs = evaluator.obs_stack[:evaluator.num_states]
    root_obs = all_state_obs[0]
    root_v_true = v_true_all[0]

    # Batch dimensions
    num_steps = int(config.get("NUM_STEPS", depth + 1))
    num_envs = int(config["NUM_ENVS"])
    batch_size = num_steps * num_envs
    minibatch_size = int(config["MINIBATCH_SIZE"])
    num_minibatches = batch_size // minibatch_size
    total_timesteps = int(config["TOTAL_TIMESTEPS"])
    num_updates = total_timesteps // batch_size
    num_epochs = int(config["NUM_EPOCHS"])

    def train(rng: jax.random.PRNGKey):
        # 2. Transformer Critic Initialization
        network = TransformerCritic(
            d_model=int(config.get("D_MODEL", 64)),
            n_heads=int(config.get("N_HEADS", 4)),
            n_layers=int(config.get("N_LAYERS", 2)),
            final_hidden_dim=int(config.get("FINAL_HIDDEN_DIM", 32)),
            max_len=depth + 4,
        )

        total_grad_steps = num_updates * num_epochs * num_minibatches
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

        # 3. Environment Reset
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)

        # 4. Main Update Step (Scanned across num_updates)
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, update_idx = runner_state

            # --- COLLECT TRAJECTORIES FROM ROOT ---
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                # Critic evaluates V(s_t)
                value = network.apply(train_state.params, last_obs, method=network.value)

                # In MRP, action is 0
                actions = jnp.zeros(num_envs, dtype=jnp.int32)
                log_prob = jnp.zeros(num_envs, dtype=jnp.float32)

                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, env_state, actions, env_params)

                next_value = network.apply(train_state.params, next_obs, method=network.value)

                transition = Transition(
                    done=done,
                    action=actions,
                    value=value,
                    next_value=next_value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    info=info,
                )
                return (train_state, next_env_state, next_obs, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, num_steps
            )

            # --- MONTE CARLO RETURN TARGETS ---
            targets = compute_mc_returns(traj_batch, gamma=gamma)

            # --- UPDATE NETWORK ---
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_data):
                    minibatch_transitions, minibatch_targets = batch_data

                    def loss_fn(p):
                        v_pred = network.apply(p, minibatch_transitions.obs, method=network.value)
                        unclipped = jnp.square(v_pred - minibatch_targets)

                        if vf_clip > 0:
                            v_clipped = minibatch_transitions.value + jnp.clip(
                                v_pred - minibatch_transitions.value, -vf_clip, vf_clip
                            )
                            clipped = jnp.square(v_clipped - minibatch_targets)
                            v_loss = 0.5 * jnp.mean(jnp.maximum(unclipped, clipped))
                        else:
                            v_loss = 0.5 * jnp.mean(unclipped)

                        return v_loss, v_pred

                    (v_loss, v_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                    grad_norm = optax.global_norm(grads)
                    ev = compute_explained_variance(v_pred, minibatch_targets)

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (v_loss, grad_norm, ev)

                train_state, batch, rng = update_state
                rng, shuffle_rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(shuffle_rng, batch, num_minibatches)

                train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (train_state, batch, rng), loss_info

            epoch_data = (traj_batch, targets)
            initial_update_state = (train_state, epoch_data, rng)
            (train_state, _, rng), loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, num_epochs
            )

            # --- METRICS COLLECTION (Fully JITted) ---
            v_loss_mean = jnp.mean(loss_info[0])
            grad_norm_mean = jnp.mean(loss_info[1])
            ev_mean = jnp.mean(loss_info[2])
            mean_rew = jnp.mean(traj_batch.reward)

            # Evaluate critic across entire tree against ground-truth backward induction V*(s)
            all_v_preds = network.apply(train_state.params, all_state_obs, method=network.value)
            value_error = jnp.mean(jnp.square(all_v_preds - v_true_all))
            root_v_pred = all_v_preds[0]
            root_abs_error = jnp.abs(root_v_pred - root_v_true)
            ranking_acc = compute_sibling_ranking_accuracy(all_v_preds, v_true_all, num_internal)
            adv_cos_sim = compute_advantage_cosine_similarity(all_v_preds, v_true_all, num_internal)

            metric = {
                "value_loss": v_loss_mean,
                "grad_norm": grad_norm_mean,
                "explained_variance": ev_mean,
                "mean_rew": mean_rew,
                "root_value": root_v_pred,
                "root_abs_error": root_abs_error,
                "value_error": value_error,
                "sibling_ranking_acc": ranking_acc,
                "advantage_cosine_sim": adv_cos_sim,
                "update_step": update_idx,
            }

            runner_state = (train_state, env_state, last_obs, rng, update_idx + 1)
            return runner_state, metric

        # 5. Run Entire Loop via jax.lax.scan
        runner_state = (train_state, env_state, obsv, rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ==============================================================================
# Evaluation, Plotting, and Checkpointing (Matching reference_code/core/evaluate.py)
# ==============================================================================

def save_plot(env_dir: str, env_name: str, x_steps: jnp.ndarray, y_series: jnp.ndarray, title: str):
    """Saves a metric plot to png."""
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_steps, y_series, label=title, color="#1f77b4", linewidth=2)
    plt.title(f"{env_name}: {title}", fontsize=13)
    plt.xlabel("Timesteps", fontsize=11)
    plt.ylabel(title, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()


def save_results(data: Any, config: dict, env_name: str, env_dir: str):
    """Saves checkpoints (out.pkl) and config.json matching reference_code/core/utils.py."""
    os.makedirs(env_dir, exist_ok=True)
    pickle_path = os.path.join(env_dir, "out.pkl")
    with open(pickle_path, "wb") as f:
        cloudpickle.dump(data, f)
    print(f"Results saved to {pickle_path}")

    config_path = os.path.join(env_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")


def evaluate(run_config: dict, make_train_fn, run_dir: str, args, rng: jax.random.PRNGKey):
    """
    Executes JIT-compiled multi-seed training, logs statistics, and writes artifacts.
    """
    batch_size = run_config["NUM_STEPS"] * run_config["NUM_ENVS"]
    total_timesteps = run_config["TOTAL_TIMESTEPS"]
    num_updates = total_timesteps // batch_size
    n_seeds = run_config.get("N_SEEDS", 1)

    print(f"\nJIT-compiling Monte Carlo Value Learning across {n_seeds} seed(s)...")
    run_fn = jax.jit(jax.vmap(make_train_fn(run_config)))

    rngs = jax.random.split(rng, n_seeds)
    out = run_fn(rngs)
    metrics = out["metrics"]

    def _mean_seeds(arr):
        arr = jnp.asarray(arr)
        return arr.mean(axis=0) if arr.ndim > 1 else arr

    final_root_err = float(_mean_seeds(metrics["root_abs_error"])[-1])
    final_val_err = float(_mean_seeds(metrics["value_error"])[-1])
    final_ev = float(_mean_seeds(metrics["explained_variance"])[-1])
    final_loss = float(_mean_seeds(metrics["value_loss"])[-1])

    env_name = f"{run_config['ENV_NAME']}-d{run_config.get('DEPTH', 4)}"
    env_dir = os.path.join(run_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)

    print(f"\n[{env_name}] Completed {total_timesteps} timesteps ({num_updates} updates)")
    print(f"  Final Value Loss:         {final_loss:.6f}")
    print(f"  Final Explained Variance: {final_ev:.4f}")
    print(f"  Final Root Abs Error:     {final_root_err:.4f}")
    print(f"  Final Total Value Error:  {final_val_err:.6f}")

    if getattr(args, "save_checkpoint", True):
        save_results(out, run_config, env_name, env_dir)
    elif getattr(args, "save_metrics", False):
        save_results(metrics, run_config, env_name, env_dir)

    timesteps = jnp.arange(1, num_updates + 1) * batch_size
    save_plot(env_dir, env_name, timesteps, _mean_seeds(metrics["value_loss"]), "value_loss")
    save_plot(env_dir, env_name, timesteps, _mean_seeds(metrics["root_abs_error"]), "root_abs_error")
    save_plot(env_dir, env_name, timesteps, _mean_seeds(metrics["value_error"]), "value_error")
    save_plot(env_dir, env_name, timesteps, _mean_seeds(metrics["explained_variance"]), "explained_variance")
    save_plot(env_dir, env_name, timesteps, _mean_seeds(metrics["mean_rew"]), "mean_rew")
    print(f"Diagnostic plots saved to {env_dir}/*.png\n")


def run_experiment_main(make_train_fn, save_dir: str):
    """CLI runner matching reference_code/core/runner.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON string of config overrides")
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--save-checkpoint", action="store_true", default=True)
    parser.add_argument("--save-metrics", action="store_true", default=False)
    args = parser.parse_args()

    cfg = default_config.copy()
    if args.config:
        try:
            overrides = json.loads(args.config)
            cfg.update(overrides)
        except Exception as e:
            print(f"Could not parse --config: {e}")

    if args.n_seeds:
        cfg["N_SEEDS"] = args.n_seeds

    timestamp = args.run_suffix or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", save_dir, timestamp)

    rng = jax.random.PRNGKey(cfg.get("SEED", 42))
    evaluate(cfg, make_train_fn, out_dir, args, rng)


if __name__ == "__main__":
    run_experiment_main(make_train, SAVE_DIR)
