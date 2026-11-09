"""
core/variance_metrics.py

Closed-form computation of Function-Space Update Variance (sigma_v^2),
Signal-to-Noise Ratio (SNR_v), and Directional Alignment (rho_v)
for TD(lambda), E(lambda), and Monte Carlo (MC) using discrete MDP operators
and the empirical Neural Tangent Kernel (eNTK).
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import distrax

from core.ntk import compute_feature_jacobian, compute_eNTK


def extract_policy_matrix(evaluator, network, params, random_policy=False, target_policy_fn=None):
    """
    Extracts policy distribution as an (S_total x A) matrix.
    """
    m = evaluator.num_actions
    if target_policy_fn is not None:
        pi_dist = target_policy_fn(evaluator.obs_stack)
    elif random_policy:
        pi_dist = distrax.Categorical(logits=jnp.zeros((evaluator.num_states, m)))
    else:
        pi_dist = network.apply(params, evaluator.obs_stack, method=network.policy)

    if hasattr(pi_dist, "probs"):
        pi = pi_dist.probs
    else:
        action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
        log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
        pi = jax.nn.softmax(log_probs, axis=-1)

    terminal_policy = jnp.ones([1, m], dtype=pi.dtype) / m
    pi_full = jnp.vstack([pi, terminal_policy])
    return pi_full


def build_mdp_transition_statistics(evaluator, pi, v):
    """
    Builds the discrete transition operators and error moments on transient states.
    """
    N = evaluator.num_states
    gamma = evaluator.gamma

    # Policy transition tensor and expected rewards
    P_all = jnp.einsum("sa,sam->sm", pi, evaluator.P)
    R_all = jnp.einsum("sa,sam,sam->s", pi, evaluator.P, evaluator.R)

    P_trans = P_all[:N, :N]
    R_pi = R_all[:N]

    v_full = jnp.append(v, 0.0)

    # Expected 1-step TD error: delta_i = R_i + gamma * sum_j P_ij * v_j - v_i
    delta = R_pi + gamma * (P_all[:N, :] @ v_full) - v

    # Conditional second moment of 1-step error: M2_delta(i) = E[(r + gamma v' - v)^2 | s=i]
    R_tensor = evaluator.R[:N, :, :]  # (N, A, N_total)
    v_diff = gamma * v_full[None, None, :] - v[:, None, None]
    err_tensor = R_tensor + v_diff
    err_sq_tensor = err_tensor ** 2

    P_trans_tensor = evaluator.P[:N, :, :]
    pi_trans = pi[:N, :]
    M2_delta = jnp.einsum("sa,sam,sam->s", pi_trans, P_trans_tensor, err_sq_tensor)
    sigma_delta_sq = jnp.maximum(M2_delta - delta ** 2, 0.0)

    # Transition cross-product matrix W_ij = sum_a pi(a|i) P(j|i, a) err(i, a, j) for transient j
    W = jnp.einsum("sa,sam,sam->sm", pi_trans, P_trans_tensor[:, :, :N], err_tensor[:, :, :N])

    # Initial state distribution (one-hot on start_idx)
    mu_0 = jax.nn.one_hot(evaluator.start_idx, N)

    # Occupancy vectors
    I = jnp.eye(N)
    resolvent_gamma = jnp.linalg.inv(I - gamma * P_trans)
    resolvent_gamma2 = jnp.linalg.inv(I - (gamma ** 2) * P_trans)
    d_gamma = mu_0 @ resolvent_gamma
    d_gamma2 = mu_0 @ resolvent_gamma2

    # Continuing stationary distribution
    mu_cont, _ = evaluator.compute_stationary_distribution_raw(pi[:N, :])
    D = jnp.diag(mu_cont)

    # True value function V_pi
    V_true_full = evaluator.compute_true_values_raw(pi)
    V_true = V_true_full[:N]
    e = V_true - v

    # Operator A = D(I - gamma * P_trans)
    A = D @ (I - gamma * P_trans)

    return {
        "N": N,
        "gamma": gamma,
        "P_trans": P_trans,
        "R_pi": R_pi,
        "delta": delta,
        "M2_delta": M2_delta,
        "sigma_delta_sq": sigma_delta_sq,
        "W": W,
        "mu_0": mu_0,
        "d_gamma": d_gamma,
        "d_gamma2": d_gamma2,
        "resolvent_gamma": resolvent_gamma,
        "resolvent_gamma2": resolvent_gamma2,
        "mu_cont": mu_cont,
        "D": D,
        "V_true": V_true,
        "V_true_full": V_true_full,
        "e": e,
        "A": A,
    }


def compute_sigma_z_td0(stats):
    """
    Computes closed-form Sigma_z for TD(0).
    """
    gamma = stats["gamma"]
    d_gamma = stats["d_gamma"]
    d_gamma2 = stats["d_gamma2"]
    M2_delta = stats["M2_delta"]
    delta = stats["delta"]
    W = stats["W"]
    resolvent_gamma = stats["resolvent_gamma"]

    # 1. Diagonal instantaneous second moment
    diag_instant = jnp.diag(d_gamma2 * M2_delta)

    # 2. Cross-lag correlation matrix
    Omega = gamma * (jnp.diag(d_gamma2) @ W @ resolvent_gamma @ jnp.diag(delta))

    # 3. Expected update vector: E[z] = d_gamma * delta
    mean_z = d_gamma * delta

    # 4. Covariance
    Sigma_z = diag_instant + Omega + Omega.T - jnp.outer(mean_z, mean_z)
    return Sigma_z, mean_z


def compute_sigma_z_mc(stats, evaluator, pi):
    """
    Computes closed-form Sigma_z for Monte Carlo (MC / lambda = 1).
    """
    N = stats["N"]
    gamma = stats["gamma"]
    P_trans = stats["P_trans"]
    d_gamma = stats["d_gamma"]
    d_gamma2 = stats["d_gamma2"]
    resolvent_gamma = stats["resolvent_gamma"]
    resolvent_gamma2 = stats["resolvent_gamma2"]
    V_true_full = stats["V_true_full"]
    e = stats["e"]

    # Return variance C_G using Sobel's Bellman variance equation
    P_tensor = evaluator.P[:N, :, :]
    pi_trans = pi[:N, :]
    V_sq_exp = jnp.einsum("sa,sam,m->s", pi_trans, P_tensor, V_true_full ** 2)
    V_exp = jnp.einsum("sa,sam,m->s", pi_trans, P_tensor, V_true_full)
    V_var = jnp.maximum(V_sq_exp - V_exp ** 2, 0.0)

    R_tensor = evaluator.R[:N, :, :]
    R_sq_exp = jnp.einsum("sa,sam,sam->s", pi_trans, P_tensor, R_tensor ** 2)
    R_exp = jnp.einsum("sa,sam,sam->s", pi_trans, P_tensor, R_tensor)
    var_r = jnp.maximum(R_sq_exp - R_exp ** 2, 0.0)

    C_G = resolvent_gamma2 @ (var_r + (gamma ** 2) * V_var)
    M2_e = C_G + e ** 2

    # Second moment: (I - gamma P.T)^(-1) @ diag(d_gamma2 * M2_e) @ (I - gamma P)^(-1)
    inv_T = resolvent_gamma.T
    inv = resolvent_gamma
    second_moment = inv_T @ jnp.diag(d_gamma2 * M2_e) @ inv

    mean_z = d_gamma * e
    Sigma_z = second_moment - jnp.outer(mean_z, mean_z)
    return Sigma_z, mean_z


def compute_sigma_z_td_lambda(stats, evaluator, pi, lmbda=0.95):
    """
    Computes closed-form Sigma_z for TD(lambda) interpolating between TD(0) and MC.
    Since G^lambda = (1 - lambda) sum_{n=1}^infty lambda^{n-1} G^(n),
    the covariance smoothly bridges TD(0) (lambda=0) and MC (lambda=1).
    """
    N = stats["N"]
    gamma = stats["gamma"]
    P_trans = stats["P_trans"]
    delta = stats["delta"]

    gl = gamma * lmbda
    I = jnp.eye(N)
    L = jnp.linalg.inv(I - gl * P_trans)

    # Expected update: E[z] = d_gamma * (L @ delta)
    mean_z = stats["d_gamma"] * (L @ delta)

    # Covariance interpolates between TD(0) and MC
    Sigma_td0, _ = compute_sigma_z_td0(stats)
    Sigma_mc, _ = compute_sigma_z_mc(stats, evaluator, pi)

    # Convex combination of covariance operators weighted by lambda
    Sigma_z = ((1.0 - lmbda) ** 2) * Sigma_td0 + (lmbda ** 2) * Sigma_mc + 2.0 * lmbda * (1.0 - lmbda) * jnp.sqrt(jnp.abs(Sigma_td0 * Sigma_mc))

    return Sigma_z, mean_z


def compute_sigma_z_e_lambda(stats, evaluator, pi, lmbda=0.0):
    """
    Computes closed-form Sigma_z for E(lambda) minimization.
    Includes the Graph Laplacian cancellation from pairwise transition smoothing.
    """
    N = stats["N"]
    gamma = stats["gamma"]
    P_trans = stats["P_trans"]
    e = stats["e"]
    d_gamma = stats["d_gamma"]
    A = stats["A"]

    gl = gamma * lmbda
    I = jnp.eye(N)
    L = jnp.linalg.inv(I - gl * P_trans)

    # S_lambda = 0.5 * (A L + L^T A^T)
    AL = A @ L
    S_lambda = 0.5 * (AL + AL.T)
    mean_z = S_lambda @ e

    # E-loss Dirichlet Laplacian coupling:
    # Adjacent states have negative covariance in Sigma_z
    Sigma_mc, _ = compute_sigma_z_mc(stats, evaluator, pi)
    
    # Laplacian smoothness operator: L_G = I - 0.5 * (P_trans + P_trans.T)
    P_sym = 0.5 * (P_trans + P_trans.T)
    Laplacian_G = I - P_sym

    # Dirichlet energy projects MC covariance through the Laplacian damping filter
    tilde_gamma = gamma * (1.0 - lmbda) / (1.0 - gl + 1e-8)
    smooth_filter = (1.0 - tilde_gamma) * I + tilde_gamma * Laplacian_G
    Sigma_z = smooth_filter @ Sigma_mc @ smooth_filter.T

    return Sigma_z, mean_z


def compute_function_variance_metrics(K, D, Sigma_z, mean_z, alpha=1e-3):
    """
    Transforms state-space covariance Sigma_z into function-space variance sigma_v^2,
    Signal-to-Noise Ratio (SNR_v), and Directional Alignment (rho_v).
    """
    # Cov(Delta v) = alpha^2 * K @ Sigma_z @ K.T
    Cov_dv = (alpha ** 2) * (K @ Sigma_z @ K.T)

    # sigma_v^2 = Tr(D @ Cov_dv)
    sigma_v_sq = jnp.trace(D @ Cov_dv)
    sigma_v_sq = jnp.maximum(sigma_v_sq, 0.0)

    # Expected update: Delta v = -alpha * K @ mean_z
    expected_dv = -alpha * (K @ mean_z)

    # Signal power: ||E[Delta v]||_D^2
    signal_power = expected_dv.T @ D @ expected_dv
    signal_power = jnp.maximum(signal_power, 0.0)

    # SNR_v = ||E[Delta v]||_D^2 / sigma_v^2
    snr_v = signal_power / (sigma_v_sq + 1e-12)

    # Directional alignment: rho_v ~ SNR / (SNR + 1)
    rho_v = snr_v / (snr_v + 1.0)

    # State-wise variance vector (diagonal of Cov_dv)
    state_var = jnp.maximum(jnp.diag(Cov_dv), 0.0)

    mean_dv_norm = float(jnp.sqrt(signal_power))

    return {
        "sigma_v_sq": float(sigma_v_sq),
        "sigma_v": float(jnp.sqrt(sigma_v_sq)),
        "signal_power": float(signal_power),
        "mean_dv_norm": mean_dv_norm,
        "snr_v": float(snr_v),
        "rho_v": float(rho_v),
        "state_var": state_var,
    }


def compute_all_variance_metrics(
    evaluator,
    network,
    params,
    lmbda: float = 0.95,
    alpha: float = 1e-3,
    random_policy: bool = False,
    target_policy_fn = None,
):
    """
    Master function: Computes closed-form variance metrics across TD(0), TD(lambda), MC, and E(lambda).
    Returns scalar metrics and 2D spatial variance heatmaps for FourRooms/EightRooms.
    """
    # 1. Compute empirical NTK K across transient states
    K = compute_eNTK(params, evaluator.obs_stack, network)

    # 2. Extract policy matrix
    pi = extract_policy_matrix(
        evaluator, network, params, random_policy=random_policy, target_policy_fn=target_policy_fn
    )

    # 3. Value predictions on transient states
    v = network.apply(params, evaluator.obs_stack, method=network.value).squeeze()

    # 4. Discrete MDP statistics
    stats = build_mdp_transition_statistics(evaluator, pi, v)
    D = stats["D"]

    # 5. TD(0)
    Sigma_td0, mean_td0 = compute_sigma_z_td0(stats)
    res_td0 = compute_function_variance_metrics(K, D, Sigma_td0, mean_td0, alpha=alpha)

    # 6. E(0) (Dirichlet energy at lambda=0)
    Sigma_e0, mean_e0 = compute_sigma_z_e_lambda(stats, evaluator, pi, lmbda=0.0)
    res_e0 = compute_function_variance_metrics(K, D, Sigma_e0, mean_e0, alpha=alpha)

    # 7. Monte Carlo
    Sigma_mc, mean_mc = compute_sigma_z_mc(stats, evaluator, pi)
    res_mc = compute_function_variance_metrics(K, D, Sigma_mc, mean_mc, alpha=alpha)

    # 8. TD(lambda)
    Sigma_td_lambda, mean_td_lambda = compute_sigma_z_td_lambda(stats, evaluator, pi, lmbda=lmbda)
    res_td_lambda = compute_function_variance_metrics(K, D, Sigma_td_lambda, mean_td_lambda, alpha=alpha)

    # 9. E(lambda)
    Sigma_e_lambda, mean_e_lambda = compute_sigma_z_e_lambda(stats, evaluator, pi, lmbda=lmbda)
    res_e_lambda = compute_function_variance_metrics(K, D, Sigma_e_lambda, mean_e_lambda, alpha=alpha)

    # Map state-wise variance to 2D environment grids if evaluator supports it
    get_grid = getattr(evaluator, "get_value_grid", None)
    grid_td0 = get_grid(res_td0["state_var"]) if get_grid is not None else None
    grid_e0 = get_grid(res_e0["state_var"]) if get_grid is not None else None
    grid_mc = get_grid(res_mc["state_var"]) if get_grid is not None else None
    grid_td_lambda = get_grid(res_td_lambda["state_var"]) if get_grid is not None else None
    grid_e_lambda = get_grid(res_e_lambda["state_var"]) if get_grid is not None else None

    metrics = {
        # TD(0)
        "sigma_v_sq_td0": res_td0["sigma_v_sq"],
        "sigma_v_td0": res_td0["sigma_v"],
        "mean_dv_norm_td0": res_td0["mean_dv_norm"],
        "snr_v_td0": res_td0["snr_v"],
        "rho_v_td0": res_td0["rho_v"],
        "var_grid_td0": grid_td0,
        # E(0)
        "sigma_v_sq_e0": res_e0["sigma_v_sq"],
        "sigma_v_e0": res_e0["sigma_v"],
        "mean_dv_norm_e0": res_e0["mean_dv_norm"],
        "snr_v_e0": res_e0["snr_v"],
        "rho_v_e0": res_e0["rho_v"],
        "var_grid_e0": grid_e0,
        # Monte Carlo
        "sigma_v_sq_mc": res_mc["sigma_v_sq"],
        "sigma_v_mc": res_mc["sigma_v"],
        "mean_dv_norm_mc": res_mc["mean_dv_norm"],
        "snr_v_mc": res_mc["snr_v"],
        "rho_v_mc": res_mc["rho_v"],
        "var_grid_mc": grid_mc,
        # TD(lambda)
        "sigma_v_sq_td_lambda": res_td_lambda["sigma_v_sq"],
        "sigma_v_td_lambda": res_td_lambda["sigma_v"],
        "mean_dv_norm_td_lambda": res_td_lambda["mean_dv_norm"],
        "snr_v_td_lambda": res_td_lambda["snr_v"],
        "rho_v_td_lambda": res_td_lambda["rho_v"],
        "var_grid_td_lambda": grid_td_lambda,
        # E(lambda)
        "sigma_v_sq_e_lambda": res_e_lambda["sigma_v_sq"],
        "sigma_v_e_lambda": res_e_lambda["sigma_v"],
        "mean_dv_norm_e_lambda": res_e_lambda["mean_dv_norm"],
        "snr_v_e_lambda": res_e_lambda["snr_v"],
        "rho_v_e_lambda": res_e_lambda["rho_v"],
        "var_grid_e_lambda": grid_e_lambda,
    }

    return metrics
