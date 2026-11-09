"""
core/sampling_e_metrics.py

Computes exact vs sampled Tang & Munos E objective metrics and critic parameter gradients:
1. Exact Matrix E: e^T S e
2. Oct 8 (Standard / Full-Branch Start-Anchored): L = (1-gamma) E[e^2] + (gamma/2) E[(e - e')^2] + (gamma/2) nu E_{s0}[e(s0)^2]
3. Forward Only: L = (1-gamma) E[e^2] + (gamma/2) E[(e - sg(e'))^2] + (gamma/2) nu E_{s0}[e(s0)^2]
4. V_aux: L = (1-gamma) E[e^2] + (gamma/2) E[(e - sg(e'))^2] + (gamma/2) E[(sg(e_tilde) - e_tilde')^2] + (gamma/2) nu E_{s0}[e(s0)^2]
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional

from core.gradient_tracking import extract_critic_flat_grads, cos_similarity


def compute_sampling_e_metrics(
    evaluator,
    network,
    params,
    mu: jnp.ndarray,
    V_true: jnp.ndarray,
    P_pi: jnp.ndarray,
    S_mat: jnp.ndarray,
    gamma: float,
    rng: jax.Array,
    num_trajectories: int = 128,
    num_steps: int = 128,
    aux_network: Optional[Any] = None,
    aux_train_state: Optional[Any] = None,
    aux_steps: int = 32,
    aux_target_mode: str = "true_value",
) -> Tuple[Dict[str, jnp.ndarray], Optional[Any]]:
    """
    Computes exact, expected, and sampled E metrics and gradients for logging across
    Oct 8, Forward Only, and V_aux variants.
    """
    S_states = evaluator.obs_stack
    N = evaluator.num_states
    start_idx = evaluator.start_idx

    # Forward value prediction helper
    def get_e_full(p):
        v = network.apply(p, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        return V_true - v_full

    # 1. Rollout sampling from mu: B parallel trajectories of T steps with rewards
    rng_init, rng_steps = jax.random.split(rng)
    mu_trans = mu[:N] / jnp.maximum(jnp.sum(mu[:N]), 1e-12)
    s0 = jax.random.choice(rng_init, N, shape=(num_trajectories,), p=mu_trans)

    # Current policy on discrete state stack
    pi_probs = network.apply(params, S_states, method=network.policy).probs
    pi_full = jnp.vstack([pi_probs, jnp.ones([1, pi_probs.shape[-1]]) / pi_probs.shape[-1]])
    P_env = evaluator.P
    R_env = evaluator.R

    def step_fn(carry_s, step_rng):
        rng_a, rng_s = jax.random.split(step_rng)
        step_rngs_a = jax.random.split(rng_a, num_trajectories)
        step_rngs_s = jax.random.split(rng_s, num_trajectories)
        logits_pi = jnp.log(jnp.maximum(pi_full[carry_s], 1e-12))
        actions = jax.vmap(jax.random.categorical)(step_rngs_a, logits_pi)
        P_sa = P_env[carry_s, actions]
        logits_p = jnp.log(jnp.maximum(P_sa, 1e-12))
        next_s_raw = jax.vmap(jax.random.categorical)(step_rngs_s, logits_p)
        rewards = R_env[carry_s, actions, next_s_raw]
        dones = (next_s_raw == N)
        next_s_carry = jnp.where(dones, start_idx, next_s_raw)
        return next_s_carry, (carry_s, next_s_raw, rewards, dones)

    step_rngs = jax.random.split(rng_steps, num_steps)
    final_s, (traj_s, traj_next_s, traj_r, traj_done) = jax.lax.scan(step_fn, s0, step_rngs)

    # 1b. Compute empirical returns G_t along each trajectory
    v_cur = network.apply(params, S_states, method=network.value).squeeze()
    v_full_cur = jnp.append(v_cur, 0.0)
    v_final = v_full_cur[final_s]

    def rev_scan_fn(g_next, step_data):
        r_t, d_t = step_data
        g_cur = r_t + gamma * jnp.where(d_t, 0.0, g_next)
        return g_cur, g_cur

    _, returns = jax.lax.scan(rev_scan_fn, v_final, (traj_r, traj_done), reverse=True)
    next_returns = jnp.roll(returns, shift=-1, axis=0)
    next_returns = next_returns.at[-1].set(v_final)
    next_returns = jnp.where(traj_done, 0.0, next_returns)

    returns_sg = jax.lax.stop_gradient(returns)
    next_returns_sg = jax.lax.stop_gradient(next_returns)
    is_start = jnp.concatenate([jnp.zeros((1, num_trajectories), dtype=bool), traj_done[:-1]], axis=0)

    # 1c. Update auxiliary network state for v_aux if provided
    new_aux_train_state = aux_train_state
    if aux_train_state is not None and aux_network is not None:
        if aux_target_mode == "true_value":
            def aux_step_fn(ts, _):
                def loss_fn(p_aux):
                    v_aux_pred = aux_network.apply(p_aux, S_states).squeeze()
                    return jnp.mean((V_true[:N] - v_aux_pred) ** 2)
                grads = jax.grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), None

            new_aux_train_state, _ = jax.lax.scan(aux_step_fn, aux_train_state, None, length=aux_steps)
        else:
            # Self-bootstrapped empirical returns for v_aux
            v_aux_cur = jnp.append(aux_network.apply(aux_train_state.params, S_states).squeeze(), 0.0)
            v_aux_final = v_aux_cur[final_s]
            _, aux_returns = jax.lax.scan(rev_scan_fn, v_aux_final, (traj_r, traj_done), reverse=True)
            aux_returns_sg = jax.lax.stop_gradient(aux_returns)

            def aux_step_fn(ts, _):
                def loss_fn(p_aux):
                    v_aux_pred = aux_network.apply(p_aux, S_states).squeeze()
                    v_aux_full = jnp.append(v_aux_pred, 0.0)
                    return jnp.mean((aux_returns_sg - v_aux_full[traj_s]) ** 2)
                grads = jax.grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), None

            new_aux_train_state, _ = jax.lax.scan(aux_step_fn, aux_train_state, None, length=aux_steps)

        v_aux = jnp.append(aux_network.apply(new_aux_train_state.params, S_states).squeeze(), 0.0)
    else:
        v_aux = V_true

    v_aux_sg = jax.lax.stop_gradient(v_aux)

    # 2. Objective functions (differentiable wrt critic params)
    def loss_exact(p):
        e_cur = get_e_full(p)
        return e_cur.T @ S_mat @ e_cur

    flux = mu - (mu @ P_pi)

    def loss_exp_oct8(p):
        e_cur = get_e_full(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e_cur ** 2))
        dirichlet = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e_cur[:, None] - e_cur[None, :]) ** 2))
        start_anchor = 0.5 * gamma * jnp.sum(flux * (e_cur ** 2))
        return mc + dirichlet + start_anchor

    def loss_exp_fwd(p):
        e_cur = get_e_full(p)
        mc = (1.0 - gamma) * jnp.sum(mu * (e_cur ** 2))
        dirichlet_fwd = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e_cur[:, None] - jax.lax.stop_gradient(e_cur[None, :])) ** 2))
        start_anchor = 0.5 * gamma * jnp.sum(flux * (e_cur ** 2))
        return mc + dirichlet_fwd + start_anchor

    def loss_exp_v_aux(p):
        e_cur = get_e_full(p)
        e_tilde = v_aux_sg - jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
        mc = (1.0 - gamma) * jnp.sum(mu * (e_cur ** 2))
        branch_vi = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((e_cur[:, None] - jax.lax.stop_gradient(e_cur[None, :])) ** 2))
        branch_vj = 0.5 * gamma * jnp.sum(mu[:, None] * P_pi * ((jax.lax.stop_gradient(e_tilde[:, None]) - e_tilde[None, :]) ** 2))
        start_anchor = 0.5 * gamma * jnp.sum(flux * (e_cur ** 2))
        return mc + branch_vi + branch_vj + start_anchor

    # Overall sample losses over full rollout using empirical returns G_t
    e_s_all = returns_sg - v_full_cur[traj_s]
    e_next_all = jnp.where(traj_done, 0.0, next_returns_sg - v_full_cur[traj_next_s])
    e_tilde_s_all = v_aux_sg[traj_s] - v_full_cur[traj_s]
    e_tilde_next_all = jnp.where(traj_done, 0.0, v_aux_sg[traj_next_s] - v_full_cur[traj_next_s])

    mc_all = (1.0 - gamma) * jnp.mean(e_s_all ** 2)
    dir_all = 0.5 * gamma * jnp.mean((e_s_all - e_next_all) ** 2)
    dir_all_fwd = 0.5 * gamma * jnp.mean((e_s_all - jax.lax.stop_gradient(e_next_all)) ** 2)
    dir_all_v_aux = (
        0.5 * gamma * jnp.mean((e_s_all - jax.lax.stop_gradient(e_next_all)) ** 2)
        + 0.5 * gamma * jnp.mean(jnp.where(traj_done, 0.0, (jax.lax.stop_gradient(e_tilde_s_all) - e_tilde_next_all) ** 2))
    )
    start_anchor_all = 0.5 * gamma * jnp.mean(jnp.where(is_start, e_s_all ** 2, 0.0))

    l_exact = loss_exact(params)
    l_exp_oct8 = loss_exp_oct8(params)
    l_exp_fwd = loss_exp_fwd(params)
    l_exp_v_aux = loss_exp_v_aux(params)

    l_samp_oct8 = mc_all + dir_all + start_anchor_all
    l_samp_fwd = mc_all + dir_all_fwd + start_anchor_all
    l_samp_v_aux = mc_all + dir_all_v_aux + start_anchor_all

    # 3. Exact & Expected Critic Parameter Gradients
    g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
    g_exp_oct8 = extract_critic_flat_grads(jax.grad(loss_exp_oct8)(params))
    g_exp_fwd = extract_critic_flat_grads(jax.grad(loss_exp_fwd)(params))
    g_exp_v_aux = extract_critic_flat_grads(jax.grad(loss_exp_v_aux)(params))

    norm_g_exact = jnp.linalg.norm(g_exact)

    rho_exp_oct8_exact = cos_similarity(g_exp_oct8, g_exact)
    rho_exp_fwd_exact = cos_similarity(g_exp_fwd, g_exact)
    rho_exp_v_aux_exact = cos_similarity(g_exp_v_aux, g_exact)

    # 4. Trajectory-Level Gradient Analysis & Full Batch Cosine Alignment
    s_trajs = jnp.transpose(traj_s, (1, 0))
    next_s_trajs = jnp.transpose(traj_next_s, (1, 0))
    done_trajs = jnp.transpose(traj_done, (1, 0))
    ret_trajs = jnp.transpose(returns_sg, (1, 0))
    next_ret_trajs = jnp.transpose(next_returns_sg, (1, 0))
    is_start_trajs = jnp.transpose(is_start, (1, 0))

    def single_traj_grad_oct8(s_b, next_b, done_b, ret_b, next_ret_b, is_start_b):
        def l_fn(p):
            v = jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
            es = ret_b - v[s_b]
            en = jnp.where(done_b, 0.0, next_ret_b - v[next_b])
            start_anchor = 0.5 * gamma * jnp.mean(jnp.where(is_start_b, es ** 2, 0.0))
            return (
                (1.0 - gamma) * jnp.mean(es ** 2)
                + 0.5 * gamma * jnp.mean((es - en) ** 2)
                + start_anchor
            )
        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    def single_traj_grad_fwd(s_b, next_b, done_b, ret_b, next_ret_b, is_start_b):
        def l_fn(p):
            v = jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
            es = ret_b - v[s_b]
            en = jnp.where(done_b, 0.0, next_ret_b - v[next_b])
            start_anchor = 0.5 * gamma * jnp.mean(jnp.where(is_start_b, es ** 2, 0.0))
            return (
                (1.0 - gamma) * jnp.mean(es ** 2)
                + 0.5 * gamma * jnp.mean((es - jax.lax.stop_gradient(en)) ** 2)
                + start_anchor
            )
        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    def single_traj_grad_v_aux(s_b, next_b, done_b, ret_b, next_ret_b, is_start_b):
        def l_fn(p):
            v = jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
            es = ret_b - v[s_b]
            en = jnp.where(done_b, 0.0, next_ret_b - v[next_b])
            e_tilde_s = v_aux_sg[s_b] - v[s_b]
            e_tilde_n = jnp.where(done_b, 0.0, v_aux_sg[next_b] - v[next_b])

            mag = (1.0 - gamma) * jnp.mean(es ** 2)
            branch_vi = 0.5 * gamma * jnp.mean((es - jax.lax.stop_gradient(en)) ** 2)
            branch_vj = 0.5 * gamma * jnp.mean(jnp.where(done_b, 0.0, (jax.lax.stop_gradient(e_tilde_s) - e_tilde_n) ** 2))
            start_anchor = 0.5 * gamma * jnp.mean(jnp.where(is_start_b, es ** 2, 0.0))
            return mag + branch_vi + branch_vj + start_anchor
        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    # Vectorized gradient computation across all B individual trajectories
    G_oct8 = jax.vmap(single_traj_grad_oct8)(s_trajs, next_s_trajs, done_trajs, ret_trajs, next_ret_trajs, is_start_trajs)
    G_fwd = jax.vmap(single_traj_grad_fwd)(s_trajs, next_s_trajs, done_trajs, ret_trajs, next_ret_trajs, is_start_trajs)
    G_v_aux = jax.vmap(single_traj_grad_v_aux)(s_trajs, next_s_trajs, done_trajs, ret_trajs, next_ret_trajs, is_start_trajs)

    # Total average gradient across the full rollout batch (B trajectories)
    g_bar_o = jnp.mean(G_oct8, axis=0)
    g_bar_f = jnp.mean(G_fwd, axis=0)
    g_bar_v = jnp.mean(G_v_aux, axis=0)

    # Full batch cosine similarity of the total average gradient to exact g*
    rho_s_o = cos_similarity(g_bar_o, g_exact)
    rho_s_f = cos_similarity(g_bar_f, g_exact)
    rho_s_v = cos_similarity(g_bar_v, g_exact)

    # Trajectory gradient spread: Tr(Var_traj(g)) = (1/B) sum ||g_b - g_bar||^2
    var_traj_o = jnp.mean(jnp.sum((G_oct8 - g_bar_o[None, :]) ** 2, axis=1))
    var_traj_f = jnp.mean(jnp.sum((G_fwd - g_bar_f[None, :]) ** 2, axis=1))
    var_traj_v = jnp.mean(jnp.sum((G_v_aux - g_bar_v[None, :]) ** 2, axis=1))

    # Variance of the average gradient estimator g_bar: Var(g_bar) = Var_traj / B
    var_o = var_traj_o / float(num_trajectories)
    var_f = var_traj_f / float(num_trajectories)
    var_v = var_traj_v / float(num_trajectories)

    # Total squared error of the average gradient to exact g*
    mse_o = jnp.sum((g_bar_o - g_exact) ** 2)
    mse_f = jnp.sum((g_bar_f - g_exact) ** 2)
    mse_v = jnp.sum((g_bar_v - g_exact) ** 2)

    norm_o = jnp.linalg.norm(g_bar_o)
    norm_f = jnp.linalg.norm(g_bar_f)
    norm_v = jnp.linalg.norm(g_bar_v)

    metrics = {
        "E_exact": l_exact,
        "E_exp_oct8": l_exp_oct8,
        "E_exp_fwd": l_exp_fwd,
        "E_exp_v_aux": l_exp_v_aux,
        "E_samp_oct8": l_samp_oct8,
        "E_samp_fwd": l_samp_fwd,
        "E_samp_v_aux": l_samp_v_aux,
        "norm_g_exact": norm_g_exact,
        # Cosine similarities of expected gradients
        "rho_exp_oct8_exact": rho_exp_oct8_exact,
        "rho_exp_fwd_exact": rho_exp_fwd_exact,
        "rho_exp_v_aux_exact": rho_exp_v_aux_exact,
        # Full-Batch Average Gradient Cosine Similarities to Exact E
        "rho_samp_oct8_exact": rho_s_o,
        "rho_samp_fwd_exact": rho_s_f,
        "rho_samp_v_aux_exact": rho_s_v,
        # Trajectory gradient variance (spread of trajectories)
        "var_traj_samp_oct8": var_traj_o,
        "var_traj_samp_fwd": var_traj_f,
        "var_traj_samp_v_aux": var_traj_v,
        # Variance of the average gradient estimator
        "var_samp_oct8": var_o,
        "var_samp_fwd": var_f,
        "var_samp_v_aux": var_v,
        # Total MSE of average gradient to exact g*
        "mse_samp_oct8": mse_o,
        "mse_samp_fwd": mse_f,
        "mse_samp_v_aux": mse_v,
        # Gradient norms of average gradients
        "norm_g_samp_oct8": norm_o,
        "norm_g_samp_fwd": norm_f,
        "norm_g_samp_v_aux": norm_v,
        # Auxiliary value network accuracy to V_true
        "mse_v_aux_target": jnp.mean((V_true - v_aux) ** 2),
    }

    if aux_train_state is not None:
        return metrics, new_aux_train_state
    return metrics
