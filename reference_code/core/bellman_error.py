# bellman_error.py
# Metrics:
# Errors: Value Error, Bellman Error, and Projected Bellman Error
# Greedy Policy Alignmnet: How many states agree for what action to take?
# Feature Space Quality: Effective Dim, Value Projection Angle, Value PCA
from core.imports import *
import distrax
from core.ntk import compute_eNTK, compute_feature_jacobian
# from sklearn.decomposition import PCA

ε =  0.0
num_components = 5
# Four Rooms has 104 states + an "invisible" terminal state.

def Bellman_Residual_Exact(D,Φ,P_π, R_π, γ):
    X = (Φ - γ * P_π @ Φ)
    reg = ε * jnp.eye(Φ.shape[-1]) / 100 # too large compared to LSTD...
    # w_br = jnp.linalg.solve(X.T @ D @ X + reg, X.T @ D @ R_π)
    w_br = jnp.linalg.pinv(X.T @ D @ X + reg) @ X.T @ D @ R_π
    V_br = Φ @ w_br
    return V_br, w_br

def LSTD_Exact(D, Φ, P_π, R_π, γ):
    # LSTD
    X = (Φ - γ * P_π @ Φ)
    A = Φ.T @ D @ X
    b = Φ.T @ D @ R_π
    reg =  ε * jnp.eye(Φ.shape[-1])
    w_lstd = jnp.linalg.pinv(A+reg) @ b
    # w_lstd = jnp.linalg.solve(A + reg, b)
    V_lstd = Φ @ w_lstd
    return V_lstd, w_lstd

def LeastSquaresValue(D,Φ,V_true):
    D_sqrt = jnp.sqrt(D)
    w_vr, _, _, _ = jnp.linalg.lstsq(D_sqrt @ Φ, D_sqrt @ V_true)
    V_vr = Φ @ w_vr
    return V_vr, w_vr

def get_error_vectors(V, v_pred, D, R_π, P_π, γ, Φ):
    Π_φ = Φ @ jnp.linalg.pinv(Φ.T @ D @ Φ) @ Φ.T @ D # projection matrix
    T = lambda v: R_π + γ * P_π @ v
    BE = T(v_pred) - v_pred
    PBE = Π_φ @ BE
    VE = V - v_pred
    Bellman_Orthogonal_Portion = T(v_pred) - Π_φ @ T(v_pred) # Tv - ΠTv
    return {'BE': BE, 'PBE': PBE, 'VE': VE, "Bellman_Orthogonal_Portion": Bellman_Orthogonal_Portion}

def compute_greedy_policy(P, R_env, γ, v):
    """
    Compute the greedy policy according to the value estimate v.
    
    Args:
        P: Transition dynamics tensor of shape (S, A, S)
        R_env: Extrinsic reward matrix of shape (S, A, S)
    """
    R_expected = jnp.einsum("sam,sam->sa", P, R_env)
    expected_v = jnp.einsum("sam,m->sa", P, v)
    Qs = R_expected + γ * expected_v
    return jnp.argmax(Qs, axis=-1)

def evaluate_greedy_policy(evaluator, v):
    greedy_actions = compute_greedy_policy(evaluator.P, evaluator.R, evaluator.gamma, v)
    pi_greedy = jax.nn.one_hot(greedy_actions, evaluator.num_actions)
    V_greedy = evaluator.compute_true_values_raw(pi_greedy)
    return V_greedy[evaluator.start_idx]

def weighted_PCA(D, Φ):
    # 1. Weight the features
    sqrt_d_weights = jnp.sqrt(jnp.diag(D))
    Weighted_Phi = sqrt_d_weights[:, None] * Φ
    
    # Ignore terminal state
    X = Weighted_Phi[:-1, :]
    
    # 2. Pure JAX PCA
    # Center the data
    X_centered = X - jnp.mean(X, axis=0)
    
    # Compute SVD
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    
    # Project down to the first 2 principal components
    # (U * S is mathematically equivalent to X_centered @ V, but faster since we have U & S)
    phi_2d = U[:, :2] * S[:2]
    
    return phi_2d # N x 2

def get_capacity_angle(V_true, V_vr, D):
    "cosine(theta), where the opposite of theta is VE"
    d_weights = jnp.diag(D)
    
    # D-weighted inner product
    inner_product = jnp.sum(d_weights * V_true * V_vr)
    
    # D-weighted norms
    norm_true = jnp.sqrt(jnp.sum(d_weights * (V_true ** 2)))
    norm_vr = jnp.sqrt(jnp.sum(d_weights * (V_vr ** 2)))
    
    return inner_product / (norm_true * norm_vr + ε)

def get_lstd_weights(evaluator, network, params, random_policy, target_policy_fn = None):
    m = evaluator.num_actions
    def get_policy_matrix():
        if target_policy_fn is not None:
            pi_dist = target_policy_fn(evaluator.obs_stack)
        elif random_policy:
            pi_dist = distrax.Categorical(
                logits=jnp.zeros((evaluator.num_states, m))
            )
        else:
            pi_dist = network.apply(params, evaluator.obs_stack, method=network.policy)

        if hasattr(pi_dist, "probs"):
            pi = pi_dist.probs
        else:
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
            log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
            pi = jax.nn.softmax(log_probs, axis=-1)
        terminal_policy = jnp.ones( [1,m], dtype=pi.dtype) / m
        pi = jnp.vstack([pi, terminal_policy])
        return pi
    
    # Get policy as S x A matrix
    pi = get_policy_matrix()
    # Get the value features:
    Φ = network.apply(params, evaluator.obs_stack, method=network.value_features)
    # terminal state
    Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))]) 
    
    # (add bias, but keep terminal state strictly zero):
    bias_col = jnp.ones((Φ.shape[0], 1)).at[-1].set(0.0)
    Φ = jnp.concatenate([Φ, bias_col], axis=-1)

    # Compute stationary dist (no terminal state)
    mu, _ = evaluator.compute_stationary_distribution_raw(pi[:-1, :])
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu) 

    # Get the exact formulation of the MDP
    γ = evaluator.gamma
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", pi, P)
    R_π = jnp.einsum("sa,sam,sam->s", pi, P, evaluator.R)
    V_lstd, w_lstd = LSTD_Exact(D, Φ, P_π, R_π, γ)
    return w_lstd    



def compute_advantage_metrics(P, R_env, γ, v_true, v_pred, mu, pi):
    R_expected = jnp.einsum("sam,sam->sa", P, R_env)
    
    Q_true = R_expected + γ * jnp.einsum("sam,m->sa", P, v_true)
    A_true = Q_true - v_true[:, None]
    
    Q_pred = R_expected + γ * jnp.einsum("sam,m->sa", P, v_pred)
    A_pred = Q_pred - v_pred[:, None]
    
    # 1. On-Policy Advantage MSE
    A_mse = jnp.sum(mu * jnp.sum(pi * (A_true - A_pred)**2, axis=-1))
    
    # 2. Uniform Advantage MSE (Uniform over states and actions)
    uniform_mu = jnp.ones_like(mu) / len(mu)
    uniform_pi = jnp.ones_like(pi) / pi.shape[-1]
    A_mse_uniform = jnp.sum(uniform_mu * jnp.sum(uniform_pi * (A_true - A_pred)**2, axis=-1))
    
    # Vector dot products for Cosine Similarity
    dot = jnp.sum(A_true * A_pred, axis=-1)
    norm_t = jnp.linalg.norm(A_true, axis=-1)
    norm_p = jnp.linalg.norm(A_pred, axis=-1)
    valid_mask = (norm_t > 1e-6)
    
    # 3. On-Policy Cosine Similarity
    cos_sim = jnp.sum(mu * valid_mask * (dot / (norm_t * norm_p + 1e-8))) / (jnp.sum(mu * valid_mask) + 1e-8)
    
    # 4. Uniform Cosine Similarity
    cos_sim_uniform = jnp.sum(uniform_mu * valid_mask * (dot / (norm_t * norm_p + 1e-8))) / (jnp.sum(uniform_mu * valid_mask) + 1e-8)
    
    return A_mse, cos_sim, A_mse_uniform, cos_sim_uniform

def value_metrics(evaluator, network, params, random_policy=False, target_policy_fn = None, light=True):
    if isinstance(light, str):
        light = light.strip().lower() in ("true", "1", "yes")
    else:
        light = bool(light)
    m = evaluator.num_actions
    def get_policy_matrix():
        if target_policy_fn is not None:
            pi_dist = target_policy_fn(evaluator.obs_stack)
        elif random_policy:
            pi_dist = distrax.Categorical(
                logits=jnp.zeros((evaluator.num_states, m))
            )
        else:
            pi_dist = network.apply(params, evaluator.obs_stack, method=network.policy)

        if hasattr(pi_dist, "probs"):
            pi = pi_dist.probs
        else:
            action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
            log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
            pi = jax.nn.softmax(log_probs, axis=-1)
        terminal_policy = jnp.ones( [1,m], dtype=pi.dtype) / m
        pi = jnp.vstack([pi, terminal_policy])
        return pi
    
    # Get policy as S x A matrix
    pi = get_policy_matrix()
    # Get the value features:
    Φ = network.apply(params, evaluator.obs_stack, method=network.value_features)
    # terminal state
    Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))]) 
    
    # (add bias, but keep terminal state strictly zero):
    bias_col = jnp.ones((Φ.shape[0], 1)).at[-1].set(0.0)
    Φ = jnp.concatenate([Φ, bias_col], axis=-1)
    
    # Compute the true value:
    V_pi = evaluator.compute_true_values_raw(pi) 
    # Compute stationary dist (no terminal state)
    mu, P_pi_cont = evaluator.compute_stationary_distribution_raw(pi[:-1, :])
    # stationary dist error:
    stat_dist_error = jnp.mean( jnp.abs ( mu.T @ P_pi_cont - mu.T )) 
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu) 

    # Get the exact formulation of the MDP
    γ = evaluator.gamma
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", pi, P)
    R_π = jnp.einsum("sa,sam,sam->s", pi, P, evaluator.R)
    I = jnp.eye(D.shape[-1])

    # Fits: Value Error, MSPBE
    V_nn = network.apply(params, evaluator.obs_stack, method=network.value)
    V_nn = jnp.append(V_nn, 0.0)
    
    # Configuration setup (light vs full)
    if not light:
        V_lstd, w_lstd = LSTD_Exact(D, Φ, P_π, R_π, γ)
        V_vr, w_vr = LeastSquaresValue(D, Φ, V_pi)
        V_br, w_br = Bellman_Residual_Exact(D, Φ, P_π, R_π, γ)
        val_configs = {
            "LSTD": (V_lstd, D, w_lstd),
            "VR": (V_vr, D, w_vr),
            "BR": (V_br, D, w_br),
            "nn": (V_nn, D, None) 
        }
    else:
        val_configs = {
            "nn": (V_nn, D, None)
        }

    true_greedy_policy = compute_greedy_policy(P, evaluator.R, γ, V_pi)

    # 1. Key Matrix A (State Space)
    A = D @ (jnp.eye(D.shape[0]) - γ * P_π)
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)

    norm_S_2 = jnp.max(jnp.abs(jnp.linalg.eigvalsh(S)))
    norm_K_2 = jnp.max(jnp.abs(jnp.linalg.eigvalsh(1j * K)))
    relative_spectral_norm = norm_K_2 / (norm_S_2 + 1e-12)

    e = V_nn - V_pi
    # Compute the weighted value error E
    E = 0.5 * jnp.dot(e, A @ e)

    mask = (mu > 1e-3).astype(float)
    E_local = 0.5 * jnp.sum(mask * (e * (A @ e)))

    # Initialize shared scalar metrics (fast metrics common to both light and full modes)
    metrics = {
        "E": E,
        "E_local": E_local,
        "stat_dist_error": stat_dist_error,
        "V_start": V_pi[evaluator.start_idx],
        "relative_spectral_norm": relative_spectral_norm,
        "norm_S_spectral": norm_S_2,
        "norm_K_spectral": norm_K_2,
    }

    # Extract extra heavy computations and spatial grid allocations if light=False
    if not light:
        stat_dist = evaluator.get_value_grid(mu)
        v_grid = evaluator.get_value_grid(V_pi)
        metrics.update({
            "value_grid": v_grid,
            "V_grid": v_grid,
            "state_dist_grid": stat_dist,
            "stat_dist": stat_dist,
            "V_nn": V_nn,
        })

        # Consider the symmetry of the key matrix.
        # 2. Symmetric and Skew-Symmetric components
        norm_s = jnp.linalg.norm(S, ord='fro')
        norm_k = jnp.linalg.norm(K, ord='fro')

        # 3. Precompute matrices for the alignment condition
        S_sq = S @ S
        SK_KS = (S @ K) - (K @ S)
        SA = S @ A

        # 4. Check global positive definiteness of SA
        SA_symmetric = 0.5 * (SA + SA.T)

        # 5. Use 'eigh' to get BOTH eigenvalues and eigenvectors
        eigenvalues_SA, eigenvectors_SA = jnp.linalg.eigh(SA_symmetric)

        min_eig_idx = jnp.argmin(eigenvalues_SA)
        min_eigenvalue = eigenvalues_SA[min_eig_idx]
        min_eigenvector = eigenvectors_SA[:, min_eig_idx]
        min_eigenvector_grid = evaluator.get_value_grid(min_eigenvector)

        is_SA_pos_def = min_eigenvalue >= 0.0

        term_1 = jnp.dot(e, S_sq @ e)
        term_2 = 0.5 * jnp.dot(e, SK_KS @ e)
        alignment_condition = term_1 + term_2  # If > 0, TD update decreases E
        alignment_condition_sign = alignment_condition > 0

        e_norm = e / jnp.linalg.norm(e)
        alignment_condition_normalized = jnp.dot(e_norm, S_sq @ e_norm) + 0.5 * jnp.dot(e_norm, SK_KS @ e_norm)

        non_normality = jnp.linalg.norm(S @ K - K @ S)
        K_Phi = Φ.T @ K @ Φ
        S_Phi = Φ.T @ S @ Φ
        phi_space_non_normality = jnp.linalg.norm(S_Phi @ K_Phi - K_Phi @ S_Phi)

        Ke = jnp.linalg.norm(K @ e)  # Degree to which TD is not SGD.

        # non_reversible_coeff (degree to which the markov chain is asymmetric)
        non_reversible_coeff = (4.0 / (evaluator.gamma**2)) * (norm_k**2) / D.shape[0]

        VE_VR = get_error_vectors(V_pi, val_configs["VR"][0], D, R_π, P_π, γ, Φ)['VE']
        nn_orthogonal_portion = get_error_vectors(V_pi, V_nn, D, R_π, P_π, γ, Φ)['Bellman_Orthogonal_Portion']

        alignment_dot_product = jnp.sum(mu * VE_VR * nn_orthogonal_portion)
        negative_alignment = (alignment_dot_product < 0)

        norm_VE_VR = jnp.sqrt(jnp.sum(mu * (VE_VR ** 2)))
        norm_nn_ortho = jnp.sqrt(jnp.sum(mu * (nn_orthogonal_portion ** 2)))
        alignment = alignment_dot_product / (norm_VE_VR * norm_nn_ortho + 1e-8)

        metrics.update({
            "SA_min_eigenvalue": min_eigenvalue,
            "is_SA_positive_definite": is_SA_pos_def,
            "alignment_condition": alignment_condition,
            "alignment_condition_normalized": alignment_condition_normalized,
            "alignment_condition_sign": alignment_condition_sign,
            "norm_s": norm_s,
            "norm_k": norm_k,
            "non_normality": non_normality,
            "Ke": Ke,
            "min_eigenvector_grid": min_eigenvector_grid,
            "phi_space_non_normality": phi_space_non_normality,
            "non_reversible_coeff": non_reversible_coeff,
            "capacity_angle": jnp.mean(get_capacity_angle(V_pi, val_configs["VR"][0], D)),
            "nn_lstd_diff": jnp.mean((val_configs["LSTD"][0] - V_nn)**2),
            "negative_alignment": negative_alignment,
            "alignment": alignment,
            "projection_error_t": norm_nn_ortho,
        })

    # Iterate to compute Grids, Errors, Policies, MSEs, and Weights dynamically
    for prefix, (V, weight_mat, w) in val_configs.items():
        
        if w is not None:
            metrics[f"{prefix}_weights"] = w

        if not light and prefix in ["LSTD", "VR", "BR", "nn"]:
            metrics[f"{prefix}_grid"] = evaluator.get_value_grid(V)

        errs = get_error_vectors(V_pi, V, weight_mat, R_π, P_π, γ, Φ)

        if prefix in ["LSTD", "VR", "nn", "BR"]:
            weighted_mse = jax.tree.map(lambda x: jnp.sum(mu * x**2), errs)
            for k, v in weighted_mse.items():
                metrics[f"{prefix}_weighted_{k}"] = v

        greedy_pol = compute_greedy_policy(P, evaluator.R, γ, V)
        metrics[f"{prefix}_greedy_correct"] = jnp.mean(true_greedy_policy == greedy_pol)
        metrics[f"{prefix}_greedy_performance"] = evaluate_greedy_policy(evaluator, V)
        A_mse, A_cos, A_mse_uni, A_cos_uni = compute_advantage_metrics(P, evaluator.R, γ, V_pi, V, mu, pi)
        metrics[f"{prefix}_advantage_mse"] = A_mse
        metrics[f"{prefix}_advantage_cossim"] = A_cos
        metrics[f"{prefix}_advantage_mse_uniform"] = A_mse_uni
        metrics[f"{prefix}_advantage_cossim_uniform"] = A_cos_uni

    return metrics


def value_metrics_light(evaluator, network, params, random_policy=False, target_policy_fn = None):
    """
    Computes a lighter subset of value metrics utilizing only the NN configuration.
    """
    return value_metrics(
        evaluator=evaluator, 
        network=network, 
        params=params, 
        random_policy=random_policy, 
        target_policy_fn=target_policy_fn, 
        light=True
    )