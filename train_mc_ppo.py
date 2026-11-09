from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
import optax
from typing import NamedTuple, Tuple

from envs.tree_mrp import TreeMRPExactValue, TreeMRP
from envs.tabular_matrix_env import TabularParams, TabularState
from transformer import TransformerCritic


class TrajectoryBatch(NamedTuple):
    obs: jnp.ndarray        # (num_envs, traj_len, depth)
    actions: jnp.ndarray    # (num_envs, traj_len)
    rewards: jnp.ndarray    # (num_envs, traj_len)
    dones: jnp.ndarray      # (num_envs, traj_len)
    values: jnp.ndarray     # (num_envs, traj_len)
    targets: jnp.ndarray    # (num_envs, traj_len) Monte Carlo returns G_t


class PPOCriticMetrics(NamedTuple):
    loss: jnp.ndarray
    grad_norm: jnp.ndarray
    explained_variance: jnp.ndarray
    mean_batch_return: jnp.ndarray
    root_value: jnp.ndarray
    root_error: jnp.ndarray


def create_train_state(
    model: TransformerCritic,
    rng: jax.random.PRNGKey,
    sample_obs: jnp.ndarray,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    total_steps: int = 500,
) -> TrainState:
    """Creates a Flax TrainState with SOTA AdamW, cosine schedule, and gradient clipping."""
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=learning_rate,
        warmup_steps=int(0.1 * total_steps),
        decay_steps=total_steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=0.9, b2=0.95),
    )
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, sample_obs)
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def collect_trajectories(
    env: TreeMRP,
    params: TabularParams,
    train_state: TrainState,
    rng: jax.random.PRNGKey,
    num_envs: int,
    traj_len: int,
    gamma: float = 1.0,
) -> TrajectoryBatch:
    """
    Collects a batch of parallel Monte Carlo rollouts from the start state (root),
    evaluating old values V_old(s) and computing backward discounted returns G_t.
    """
    reset_rng, step_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, num_envs)

    # 1. Reset all environments to root state
    init_obs, init_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)

    def env_step_fn(carry, _):
        curr_state, curr_obs, k = carry
        k, step_k = jax.random.split(k)
        step_keys = jax.random.split(step_k, num_envs)

        # In MRP, action is dummy 0
        actions = jnp.zeros(num_envs, dtype=jnp.int32)

        # Step environments in parallel
        next_obs, next_state, rewards, dones, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, curr_state, actions, params)

        # Evaluate V_old(s_t) with current critic parameters
        v_old = train_state.apply_fn(train_state.params, curr_obs)

        transition = (curr_obs, actions, rewards, dones, v_old)
        next_carry = (next_state, next_obs, k)
        return next_carry, transition

    _, (obs_seq, actions_seq, rewards_seq, dones_seq, values_seq) = jax.lax.scan(
        env_step_fn, (init_state, init_obs, step_rng), None, length=traj_len
    )

    # Transpose from (traj_len, num_envs, ...) to (num_envs, traj_len, ...)
    obs_batch = jnp.swapaxes(obs_seq, 0, 1)
    actions_batch = jnp.swapaxes(actions_seq, 0, 1)
    rewards_batch = jnp.swapaxes(rewards_seq, 0, 1)
    dones_batch = jnp.swapaxes(dones_seq, 0, 1)
    values_batch = jnp.swapaxes(values_seq, 0, 1)

    # 2. Compute Monte Carlo return G_t backwards from terminal:
    # G_t = r_{t+1} + gamma * (1 - done_{t+1}) * G_{t+1}
    def compute_mc_targets(rewards, dones):
        def backward_scan(next_g, step_data):
            r, d = step_data
            # If done at this transition, future return is 0
            g = r + gamma * (1.0 - d.astype(jnp.float32)) * next_g
            return g, g

        # Scan backwards along time axis (length traj_len)
        _, targets = jax.lax.scan(
            backward_scan, 0.0, (rewards, dones), reverse=True
        )
        return targets

    targets_batch = jax.vmap(compute_mc_targets)(rewards_batch, dones_batch)

    return TrajectoryBatch(
        obs=obs_batch,
        actions=actions_batch,
        rewards=rewards_batch,
        dones=dones_batch,
        values=values_batch,
        targets=targets_batch,
    )


def ppo_critic_loss(
    params,
    apply_fn,
    batch_obs: jnp.ndarray,
    batch_v_old: jnp.ndarray,
    batch_targets: jnp.ndarray,
    vf_clip: float = 0.2,
) -> jnp.ndarray:
    """
    PPO Clipped Value Loss:
      L^VF = 0.5 * max( (V - G)^2, (clip(V, V_old - eps, V_old + eps) - G)^2 )
    If vf_clip <= 0, reduces to standard unclipped MSE.
    """
    v_pred = apply_fn(params, batch_obs)
    unclipped_loss = jnp.square(v_pred - batch_targets)

    if vf_clip is not None and vf_clip > 0:
        v_clipped = batch_v_old + jnp.clip(v_pred - batch_v_old, -vf_clip, vf_clip)
        clipped_loss = jnp.square(v_clipped - batch_targets)
        loss = 0.5 * jnp.mean(jnp.maximum(unclipped_loss, clipped_loss))
    else:
        loss = 0.5 * jnp.mean(unclipped_loss)

    return loss, v_pred


def compute_explained_variance(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Explained variance EV = 1 - Var(y_true - y_pred) / Var(y_true)."""
    var_y = jnp.var(y_true)
    var_diff = jnp.var(y_true - y_pred)
    return jnp.where(var_y > 1e-8, 1.0 - (var_diff / var_y), 0.0)


from functools import partial


@partial(jax.jit, static_argnames=("vf_clip",))
def update_critic_minibatch(
    state: TrainState,
    batch_obs: jnp.ndarray,
    batch_v_old: jnp.ndarray,
    batch_targets: jnp.ndarray,
    vf_clip: float = 0.2,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single mini-batch update step on the collected Monte Carlo targets."""
    def loss_fn(p):
        return ppo_critic_loss(p, state.apply_fn, batch_obs, batch_v_old, batch_targets, vf_clip)

    (loss, v_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grad_norm = optax.global_norm(grads)
    ev = compute_explained_variance(v_pred, batch_targets)

    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, grad_norm, ev


def train_mc_ppo(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    num_iterations: int = 150,     # PPO iterations
    num_envs: int = 64,            # Parallel trajectories per iteration
    num_epochs: int = 4,           # PPO critic training epochs per batch
    minibatch_size: int = 64,      # Transitions per gradient step
    lr: float = 2e-3,
    vf_clip: float = 0.2,          # PPO value function clipping epsilon
    seed: int = 42,
):
    """
    Full Monte Carlo PPO training loop for the Transformer Critic:
    Interleaves rollout collection from start state with mini-batch PPO critic learning.
    """
    print(f"=== Monte Carlo PPO Critic Training (Tree MRP Depth={depth}, p={p}) ===")
    evaluator = TreeMRPExactValue(depth=depth, p=p, cluster_size=cluster_size)
    V_true = evaluator.solve_backward_induction()
    root_target = float(V_true[0])
    root_obs = evaluator.obs_stack[0]

    env = TreeMRP(depth=depth, p=p, cluster_size=cluster_size)
    env_params = TabularParams(max_steps_in_episode=depth + 2)
    traj_len = depth + 1  # root -> depth 1 -> ... -> leaf -> terminal exit

    # Initialize Transformer Critic
    model = TransformerCritic(
        d_model=64,
        n_heads=4,
        n_layers=2,
        final_hidden_dim=32,
        max_len=16,
    )

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    sample_obs = root_obs

    total_training_steps = num_iterations * num_epochs * max(1, (num_envs * traj_len) // minibatch_size)
    state = create_train_state(
        model=model,
        rng=init_rng,
        sample_obs=sample_obs,
        learning_rate=lr,
        total_steps=total_training_steps,
    )

    print(f"States in tree: {evaluator.num_states} | Exact V*(root) = {root_target:.4f}")
    print(f"Rollout batch: {num_envs} trajectories x {traj_len} steps = {num_envs * traj_len} transitions/iter")
    print(f"PPO Critic epochs: {num_epochs} | Minibatch: {minibatch_size} | VF Clip: {vf_clip}\n")

    header = f"{'Iter':>5} | {'Samples':>7} | {'Mean Ret':>8} | {'VF Loss':>10} | {'Grad Norm':>9} | {'Expl Var':>8} | {'V_pred(root)':>12} | {'V*(root)':>9} | {'Root Err':>9}"
    print(header)
    print("-" * len(header))

    total_samples = 0

    for iteration in range(1, num_iterations + 1):
        rng, roll_rng = jax.random.split(rng)

        # 1. ROLLOUT PHASE: Collect Monte Carlo trajectories from root
        traj_batch = collect_trajectories(
            env=env,
            params=env_params,
            train_state=state,
            rng=roll_rng,
            num_envs=num_envs,
            traj_len=traj_len,
            gamma=1.0,
        )
        total_samples += num_envs * traj_len

        # Flatten transitions across envs and time: (num_envs * traj_len, ...)
        flat_obs = traj_batch.obs.reshape(-1, depth)
        flat_v_old = traj_batch.values.reshape(-1)
        flat_targets = traj_batch.targets.reshape(-1)
        mean_return = float(jnp.mean(traj_batch.targets[:, 0]))  # empirical win-rate from root

        num_transitions = flat_obs.shape[0]

        # 2. LEARNING PHASE: Interleaved PPO critic updates over K epochs
        epoch_losses, epoch_evs, epoch_norms = [], [], []

        for _ in range(num_epochs):
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, num_transitions)
            shuffled_obs = flat_obs[permutation]
            shuffled_v_old = flat_v_old[permutation]
            shuffled_targets = flat_targets[permutation]

            for start_idx in range(0, num_transitions, minibatch_size):
                end_idx = min(start_idx + minibatch_size, num_transitions)
                m_obs = shuffled_obs[start_idx:end_idx]
                m_v_old = shuffled_v_old[start_idx:end_idx]
                m_targets = shuffled_targets[start_idx:end_idx]

                state, loss_val, norm_val, ev_val = update_critic_minibatch(
                    state, m_obs, m_v_old, m_targets, vf_clip=vf_clip
                )
                epoch_losses.append(float(loss_val))
                epoch_norms.append(float(norm_val))
                epoch_evs.append(float(ev_val))

        # 3. LOGGING: Track evaluation metrics
        if iteration % 15 == 0 or iteration == 1 or iteration == num_iterations:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_norm = sum(epoch_norms) / len(epoch_norms)
            avg_ev = sum(epoch_evs) / len(epoch_evs)
            root_v = float(state.apply_fn(state.params, root_obs))
            root_err = abs(root_v - root_target)

            print(
                f"{iteration:5d} | "
                f"{total_samples:7d} | "
                f"{mean_return:8.4f} | "
                f"{avg_loss:10.6f} | "
                f"{avg_norm:9.4f} | "
                f"{avg_ev:8.4f} | "
                f"{root_v:12.4f} | "
                f"{root_target:9.4f} | "
                f"{root_err:9.4f}"
            )

    # 4. FINAL EVALUATION: Benchmark against exact backward induction across the tree
    print("\n=== Final Value Function Comparison against Exact V*(s) ===")
    sample_paths = ["", "1", "0", "11", "10", "111", "1110", "1111", "0000"]
    sample_indices = [evaluator.string_to_idx(p) for p in sample_paths]
    sample_obs = jnp.stack([evaluator.idx_to_obs(idx) for idx in sample_indices])
    sample_true = [float(V_true[idx]) for idx in sample_indices]
    sample_preds = state.apply_fn(state.params, sample_obs)

    eval_header = f"{'Path':<10} | {'Observation Vector':<22} | {'V_MC(s)':<10} | {'V*(s)':<10} | {'Abs Error':<10}"
    print(eval_header)
    print("-" * len(eval_header))
    for p, obs, v_p, v_t in zip(sample_paths, sample_obs, sample_preds, sample_true):
        p_str = f'"{p}"' if p != "" else '"" (Root)'
        err = abs(float(v_p) - float(v_t))
        print(f"{p_str:<10} | {str(list(obs.tolist())):<22} | {float(v_p):<10.4f} | {float(v_t):<10.4f} | {err:<10.4f}")

    return state


if __name__ == "__main__":
    train_mc_ppo()
