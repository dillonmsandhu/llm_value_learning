from __future__ import annotations
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from typing import NamedTuple

from envs.tree_mrp import TreeMRPExactValue
from transformer import TransformerCritic


class Metrics(NamedTuple):
    loss: jnp.ndarray
    grad_norm: jnp.ndarray
    root_value_error: jnp.ndarray


def create_train_state(
    model: TransformerCritic,
    rng: jax.random.PRNGKey,
    sample_obs: jnp.ndarray,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_grad_norm: float = 1.0,
    total_steps: int = 200,
    warmup_steps: int = 20,
) -> TrainState:
    """Creates a Flax TrainState with SOTA AdamW, cosine decay, and gradient clipping."""
    # 1. Cosine learning rate schedule with linear warmup (standard in modern LLM / RLVR training)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5,
    )

    # 2. AdamW optimizer with decoupled weight decay and global gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, b1=0.9, b2=0.95),
    )

    # 3. Initialize parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, sample_obs)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jax.jit
def train_step(
    state: TrainState,
    batch_obs: jnp.ndarray,
    batch_targets: jnp.ndarray,
    root_obs: jnp.ndarray,
    root_target: jnp.ndarray,
) -> tuple[TrainState, Metrics]:
    """Single JIT-compiled supervised step minimizing Mean Squared Error (MSE)."""
    def loss_fn(params):
        preds = state.apply_fn(params, batch_obs)
        # Mean Squared Error: 0.5 * (V_pred - V_target)^2
        loss = 0.5 * jnp.mean((preds - batch_targets) ** 2)
        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)

    # Track gradient norm for monitoring stability
    grad_norm = optax.global_norm(grads)

    # Track error on root node V("")
    root_pred = state.apply_fn(state.params, root_obs)
    root_err = jnp.abs(root_pred - root_target)

    new_state = state.apply_gradients(grads=grads)
    metrics = Metrics(loss=loss, grad_norm=grad_norm, root_value_error=root_err)
    return new_state, metrics


def train_tree_critic_supervised(
    depth: int = 4,
    p: float = 0.8,
    cluster_size: int = 2,
    num_steps: int = 300,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 42,
):
    """
    Supervised training loop fitting the Transformer Critic to ground-truth values V*(s).
    """
    print(f"--- Setting up Tree MRP (Depth={depth}, p={p}, ClusterSize={cluster_size}) ---")
    evaluator = TreeMRPExactValue(depth=depth, p=p, cluster_size=cluster_size)
    V_true = evaluator.solve_backward_induction()

    # Active states (excluding terminal)
    all_obs = evaluator.obs_stack[:evaluator.num_states]
    all_targets = V_true[:evaluator.num_states]
    root_obs = all_obs[0]
    root_target = all_targets[0]

    num_samples = len(all_obs)
    print(f"Total active states in tree: {num_samples} (2^{depth+1} - 1)")
    print(f"Exact Root Value V*(root) = {float(root_target):.4f}")

    # Initialize SOTA Transformer Critic
    model = TransformerCritic(
        d_model=64,
        n_heads=4,
        n_layers=2,
        final_hidden_dim=32,
        max_len=16,
    )

    rng = jax.random.PRNGKey(seed)
    rng, state_rng, train_rng = jax.random.split(rng, 3)

    state = create_train_state(
        model=model,
        rng=state_rng,
        sample_obs=root_obs,
        learning_rate=learning_rate,
        total_steps=num_steps,
    )

    print(f"\n--- Training Transformer Critic with AdamW + Cosine Warmup ({num_steps} steps) ---")
    header = f"{'Step':>6} | {'MSE Loss':>10} | {'Grad Norm':>10} | {'V_pred(root)':>12} | {'V*(root)':>10} | {'Root Abs Err':>12}"
    print(header)
    print("-" * len(header))

    for step in range(1, num_steps + 1):
        train_rng, batch_rng = jax.random.split(train_rng)
        # Random mini-batch sampling from the tree state space
        batch_indices = jax.random.randint(batch_rng, (batch_size,), 0, num_samples)
        batch_obs = all_obs[batch_indices]
        batch_targets = all_targets[batch_indices]

        state, metrics = train_step(state, batch_obs, batch_targets, root_obs, root_target)

        if step % 50 == 0 or step == 1 or step == num_steps:
            root_pred = float(state.apply_fn(state.params, root_obs))
            print(
                f"{step:6d} | "
                f"{float(metrics.loss):10.6f} | "
                f"{float(metrics.grad_norm):10.4f} | "
                f"{root_pred:12.4f} | "
                f"{float(root_target):10.4f} | "
                f"{float(metrics.root_value_error):12.4f}"
            )

    print("\n--- Final Evaluation Across Sample Reasoning Paths ---")
    sample_paths = ["", "1", "0", "11", "10", "111", "1110", "1111", "0000"]
    sample_indices = [evaluator.string_to_idx(p) for p in sample_paths]
    sample_obs = jnp.stack([evaluator.idx_to_obs(idx) for idx in sample_indices])
    sample_true = [float(V_true[idx]) for idx in sample_indices]
    sample_preds = state.apply_fn(state.params, sample_obs)

    eval_header = f"{'Path':<10} | {'Observation Vector':<22} | {'V_pred(s)':<10} | {'V*(s)':<10} | {'Error':<10}"
    print(eval_header)
    print("-" * len(eval_header))
    for p, obs, v_p, v_t in zip(sample_paths, sample_obs, sample_preds, sample_true):
        p_str = f'"{p}"' if p != "" else '"" (Root)'
        err = abs(float(v_p) - float(v_t))
        print(f"{p_str:<10} | {str(list(obs.tolist())):<22} | {float(v_p):<10.4f} | {float(v_t):<10.4f} | {err:<10.4f}")

    return state


if __name__ == "__main__":
    train_tree_critic_supervised()
