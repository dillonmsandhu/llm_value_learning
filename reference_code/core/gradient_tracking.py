import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any
from flax.training.train_state import TrainState

import core.helpers as helpers


def extract_critic_flat_grads(grads) -> jnp.ndarray:
    """Extracts and flattens only the critic parameter gradients from a PyTree."""
    if not isinstance(grads, dict) or "params" not in grads:
        leaves = jax.tree_util.tree_leaves(grads)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

    critic_subtrees = {
        k: v for k, v in grads["params"].items() if "critic" in k
    }
    if not critic_subtrees:
        leaves = jax.tree_util.tree_leaves(grads["params"])
    else:
        leaves = jax.tree_util.tree_leaves(critic_subtrees)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


def cos_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Computes directional cosine similarity between two flat vectors."""
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    denom = jnp.maximum(norm_a * norm_b, 1e-12)
    return jnp.dot(a, b) / denom


def compute_all_exact_critic_gradients(
    train_state: TrainState,
    evaluator,
    network,
    gamma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, Dict[str, float]]:
    """
    Computes exact parameter gradients and geometric alignment for:
    1. g_exact_E: Symmetrized Bellman error gradient (Dirichlet form)
    2. g_exact_TD: Expected TD(0) semi-gradient (with detached target)
    3. g_exact_MC: Expected Monte Carlo mean squared error gradient
    along with start-state value V^pi(s_0), relative spectral norm ||K||/||S||, and pairwise cosines.
    """
    S_states = evaluator.obs_stack
    n_actions = evaluator.num_actions

    # Extract current policy matrix pi(s, a)
    pi_dist, _ = network.apply(train_state.params, S_states)
    if hasattr(pi_dist, "probs"):
        old_pi = pi_dist.probs
    else:
        action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
        old_log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
        old_pi = jax.nn.softmax(old_log_probs, axis=-1)

    terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
    old_pi_full = jnp.vstack([old_pi, terminal_policy])

    # Value function & stationary distribution
    V_true = evaluator.compute_true_values_raw(old_pi_full)
    start_val = V_true[evaluator.start_idx]

    mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
    mu_full = jnp.append(mu, 0.0)
    D = jnp.diag(mu_full)

    P = evaluator.P
    I = jnp.eye(evaluator.num_total_states)
    P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
    R_pi = jnp.einsum("sa,sam->s", old_pi_full, evaluator.R)

    A_mat = D @ (I - gamma * P_pi)
    S_mat = 0.5 * (A_mat + A_mat.T)
    K_mat = 0.5 * (A_mat - A_mat.T)

    norm_S_2 = jnp.linalg.svd(S_mat, compute_uv=False)[0]
    norm_K_2 = jnp.linalg.svd(K_mat, compute_uv=False)[0]
    rel_spectral_norm = norm_K_2 / (norm_S_2 + 1e-12)

    # 1. Exact E Loss: (V_true - v)^T S_mat (V_true - v)
    def exact_e_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        diff = V_true - jnp.append(v, 0.0)
        return diff.T @ S_mat @ diff

    # 2. Exact TD Loss: 0.5 * sum_s mu(s) (v(s) - sg(T(v)(s)))^2
    def exact_td_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        T_v = R_pi + gamma * (P_pi @ v_full)
        err = v_full - jax.lax.stop_gradient(T_v)
        return 0.5 * jnp.sum(mu_full * (err ** 2))

    # 3. Exact MC Loss: 0.5 * sum_s mu(s) (v(s) - V_true(s))^2
    def exact_mc_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        err = v_full - V_true
        return 0.5 * jnp.sum(mu_full * (err ** 2))

    g_exact_E = extract_critic_flat_grads(jax.grad(exact_e_loss)(train_state.params))
    g_exact_TD = extract_critic_flat_grads(jax.grad(exact_td_loss)(train_state.params))
    g_exact_MC = extract_critic_flat_grads(jax.grad(exact_mc_loss)(train_state.params))

    alignments = {
        "rho_exactE_exactTD": cos_similarity(g_exact_E, g_exact_TD),
        "rho_exactTD_exactMC": cos_similarity(g_exact_TD, g_exact_MC),
        "rho_exactE_exactMC": cos_similarity(g_exact_E, g_exact_MC),
    }

    return g_exact_E, g_exact_TD, g_exact_MC, start_val, rel_spectral_norm, alignments


def compute_exact_e_lambda_critic_gradient(
    train_state: TrainState,
    evaluator,
    network,
    gamma: float,
    lmbda: float,
) -> Tuple[jnp.ndarray, float]:
    """
    Computes exact parameter gradient for the E(lambda) Dirichlet energy form:
    L_{E(lambda)}(theta) = (V_true - v_theta)^T S_lambda (V_true - v_theta)
    where S_lambda = 0.5 * (A L + L^T A^T) with L = (I - gamma * lambda * P_pi)^{-1} and A = D (I - gamma * P_pi).
    """
    S_states = evaluator.obs_stack
    n_actions = evaluator.num_actions

    # Extract current policy matrix pi(s, a)
    pi_dist, _ = network.apply(train_state.params, S_states)
    if hasattr(pi_dist, "probs"):
        old_pi = pi_dist.probs
    else:
        action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
        old_log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
        old_pi = jax.nn.softmax(old_log_probs, axis=-1)

    terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
    old_pi_full = jnp.vstack([old_pi, terminal_policy])

    # Value function & stationary distribution
    V_true = evaluator.compute_true_values_raw(old_pi_full)
    start_val = V_true[evaluator.start_idx]

    mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
    mu_full = jnp.append(mu, 0.0)
    D = jnp.diag(mu_full)

    P = evaluator.P
    I = jnp.eye(evaluator.num_total_states)
    P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)

    A_mat = D @ (I - gamma * P_pi)
    gl = gamma * lmbda
    L = jnp.linalg.inv(I - gl * P_pi)
    AL = A_mat @ L
    S_lambda = 0.5 * (AL + AL.T)

    def exact_e_lambda_loss(params):
        v = network.apply(params, S_states, method=network.value).squeeze()
        diff = V_true - jnp.append(v, 0.0)
        return diff.T @ S_lambda @ diff

    g_exact_E_lambda = extract_critic_flat_grads(jax.grad(exact_e_lambda_loss)(train_state.params))
    return g_exact_E_lambda, start_val


def compute_sampled_critic_gradients(
    train_state: TrainState,
    network,
    traj_batch,
    targets: jnp.ndarray,
    gamma: float,
) -> Dict[str, jnp.ndarray]:
    """
    Computes flat sampled gradients for:
    1. g_samp_sep14: Sept 14 (Uncorrected, magnitude + Dirichlet smoothness)
    2. g_samp_oct5:  Oct 5 (Corrected with sample difference of second moments)
    3. g_samp_oct8:  Oct 8 (Start-anchored with boundary weighting nu * mean(e_s0^2))
    4. g_samp_TD:    Sampled TD(0) semi-gradient with detached target
    """
    obs = traj_batch.obs
    next_obs = traj_batch.next_obs
    is_timeout = traj_batch.info["is_timeout"] if "is_timeout" in traj_batch.info else jnp.zeros_like(traj_batch.done, dtype=bool)
    true_terminal = traj_batch.done & ~is_timeout

    # Align next return targets G_{t+1} safely from targets
    if hasattr(traj_batch, "next_target"):
        next_targets = traj_batch.next_target
    else:
        next_targets = jnp.roll(targets, shift=-1, axis=0)
        if hasattr(traj_batch, "next_value"):
            next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
            next_targets = jnp.where(is_timeout, traj_batch.next_value, next_targets)
        next_targets = jnp.where(true_terminal, 0.0, next_targets)

    # Start-state mask: a step is an episode start if the preceding step terminated via true terminal (not timeout)
    is_start = jnp.roll(true_terminal, shift=1, axis=0)
    is_start = is_start.at[0].set(False)

    # 1. Sept 14 (Uncorrected) Sampled E critic loss
    def loss_sep14(params):
        v_i = network.apply(params, obs, method=network.value)
        v_j = network.apply(params, next_obs, method=network.value)
        v_loss, _, _ = helpers.e_critic_loss(
            v_i, targets, v_j, next_targets, true_terminal, gamma, variant="sep14"
        )
        return v_loss

    # 2. Oct 5 (Corrected) Sampled E critic loss
    def loss_oct5(params):
        v_i = network.apply(params, obs, method=network.value)
        v_j = network.apply(params, next_obs, method=network.value)
        v_loss, _, _ = helpers.e_critic_loss(
            v_i, targets, v_j, next_targets, true_terminal, gamma, variant="corrected"
        )
        return v_loss

    # 3. Oct 8 (Start-Anchored) Sampled E critic loss
    def loss_oct8(params):
        v_i = network.apply(params, obs, method=network.value)
        v_j = network.apply(params, next_obs, method=network.value)
        v_loss, _, _ = helpers.e_critic_loss(
            v_i, targets, v_j, next_targets, true_terminal, gamma, variant="oct8", is_start=is_start
        )
        return v_loss

    # 4. Sampled TD(0) critic loss: 0.5 * (v(s) - sg(r + gamma * (1-d) * v(s')))^2
    def loss_td(params):
        v_s = network.apply(params, obs, method=network.value)
        v_next = network.apply(params, next_obs, method=network.value)
        td_target = traj_batch.reward + gamma * (1.0 - true_terminal.astype(jnp.float32)) * jax.lax.stop_gradient(v_next)
        td_target = jnp.where(is_timeout, jax.lax.stop_gradient(v_next), td_target)
        return 0.5 * jnp.mean((v_s - td_target) ** 2)

    g_samp_sep14 = extract_critic_flat_grads(jax.grad(loss_sep14)(train_state.params))
    g_samp_oct5 = extract_critic_flat_grads(jax.grad(loss_oct5)(train_state.params))
    g_samp_oct8 = extract_critic_flat_grads(jax.grad(loss_oct8)(train_state.params))
    g_samp_TD = extract_critic_flat_grads(jax.grad(loss_td)(train_state.params))

    return {
        "g_samp_sep14": g_samp_sep14,
        "g_samp_oct5": g_samp_oct5,
        "g_samp_oct8": g_samp_oct8,
        "g_samp_E": g_samp_oct8,  # Default sampled E gradient
        "g_samp_TD": g_samp_TD,
    }


def compute_gradient_tracking_metrics(
    train_state: TrainState,
    evaluator,
    network,
    traj_batch,
    targets: jnp.ndarray,
    gamma: float,
) -> Dict[str, Any]:
    """
    Top-level helper called optionally inside ppo/sampled_E.py.
    Computes exact critic gradients, sampled gradients, squared errors, and cosine similarities.
    """
    g_exact_E, g_exact_TD, g_exact_MC, start_val, rel_spectral_norm, alignments = (
        compute_all_exact_critic_gradients(train_state, evaluator, network, gamma)
    )
    samp_grads = compute_sampled_critic_gradients(train_state, network, traj_batch, targets, gamma)
    g_samp_sep14 = samp_grads["g_samp_sep14"]
    g_samp_oct5 = samp_grads["g_samp_oct5"]
    g_samp_oct8 = samp_grads["g_samp_oct8"]
    g_samp_TD = samp_grads["g_samp_TD"]

    # Alignments of the three sampled E formulations to exact matrix g*
    rho_sep14 = cos_similarity(g_samp_sep14, g_exact_E)
    rho_oct5 = cos_similarity(g_samp_oct5, g_exact_E)
    rho_oct8 = cos_similarity(g_samp_oct8, g_exact_E)

    # Squared errors to exact matrix g*
    sq_err_sep14 = jnp.sum((g_samp_sep14 - g_exact_E) ** 2)
    sq_err_oct5 = jnp.sum((g_samp_oct5 - g_exact_E) ** 2)
    sq_err_oct8 = jnp.sum((g_samp_oct8 - g_exact_E) ** 2)

    # TD alignments
    sq_err_TD = jnp.sum((g_samp_TD - g_exact_TD) ** 2)
    rho_SE_ETD = cos_similarity(g_samp_oct8, g_exact_TD)
    rho_STD_ETD = cos_similarity(g_samp_TD, g_exact_TD)

    return {
        "sq_err": sq_err_oct8,
        "sq_err_sep14": sq_err_sep14,
        "sq_err_oct5": sq_err_oct5,
        "sq_err_oct8": sq_err_oct8,
        "sq_err_td": sq_err_TD,
        "rho_SE_EE": rho_oct8,
        "rho_samp_sep14_exactE": rho_sep14,
        "rho_samp_oct5_exactE": rho_oct5,
        "rho_samp_oct8_exactE": rho_oct8,
        "rho_SE_ETD": rho_SE_ETD,
        "rho_STD_ETD": rho_STD_ETD,
        "rho_exactE_exactTD": alignments["rho_exactE_exactTD"],
        "rho_exactTD_exactMC": alignments["rho_exactTD_exactMC"],
        "rho_exactE_exactMC": alignments["rho_exactE_exactMC"],
        "relative_spectral_norm": rel_spectral_norm,
        "v_true_start": start_val,
        "g_samp_E": g_samp_oct8,
        "g_samp_sep14": g_samp_sep14,
        "g_samp_oct5": g_samp_oct5,
        "g_samp_oct8": g_samp_oct8,
        "g_samp_TD": g_samp_TD,
        "g_exact_E": g_exact_E,
    }

