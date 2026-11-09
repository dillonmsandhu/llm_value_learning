# bellman_error.py
# Metrics:
# Errors: Value Error, Bellman Error, and Projected Bellman Error
# Greedy Policy Alignmnet: How many states agree for what action to take?
# Feature Space Quality: Effective Dim, Value Projection Angle, Value PCA
from core.imports import *
import distrax
from core.ntk import compute_eNTK, compute_feature_jacobian, compute_centered_eNTK
# from sklearn.decomposition import PCA

ε =  0.0
num_components = 5

def get_capacity_angle(V_true, V_vr, D):
    "cosine(theta), where the opposite of theta is VE"
    d_weights = jnp.diag(D)
    
    # D-weighted inner product
    inner_product = jnp.sum(d_weights * V_true * V_vr)
    
    # D-weighted norms
    norm_true = jnp.sqrt(jnp.sum(d_weights * (V_true ** 2)))
    norm_vr = jnp.sqrt(jnp.sum(d_weights * (V_vr ** 2)))
    
    return inner_product / (norm_true * norm_vr + ε)

def feature_metrics(evaluator, network, params, random_policy=False, target_policy_fn = None, log_variance=False, lmbda=0.95):
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
    A = D @ (jnp.eye(D.shape[0]) - γ * P_π)

    # Feature Quality (effective rank and PCA).
    U, S, _ = jnp.linalg.svd(Φ, full_matrices=False)
    sig_level = (1-γ) / 10.0
    effective_rank = jnp.sum(S > sig_level)
    feature_singular_values = S
    top_u_vectors = U[:, :num_components].T 

    # 3. Vectorize your grid function
    # jax.vmap will apply get_value_grid to each row of top_u_vectors
    get_grids_fn = jax.vmap(evaluator.get_value_grid)

    # heatmaps_stack will have shape (5, H, W)
    feature_top_singular_vectors = get_grids_fn(top_u_vectors)
    
    # NTK
    eNTK = compute_eNTK(params, evaluator.obs_stack, network)
    
    # Use eigvalsh for symmetric matrices (returns eigenvalues in ascending order)
    eigenvalues = jnp.linalg.eigvalsh(eNTK)
    
    # Use a relative threshold (e.g., 0.01% of the max eigenvalue)
    threshold = 1e-4 * jnp.max(eigenvalues)
    eNTK_effective_rank = jnp.sum(eigenvalues > threshold)

    J = compute_feature_jacobian(params, evaluator.obs_stack, network)
    Uj, Sj, Vt_j = jnp.linalg.svd(J, full_matrices=False)

    Direlechet_energy = jnp.trace(Φ.T @ A @ Φ) # scalar
    # 2. Slice the top N components (statically sized for JIT)
    
    # Uj shape is (N, D). Slice to (N, 5), then transpose to (5, N)
    top_u_vectors = Uj[:, :num_components].T 
    heatmaps_stack = get_grids_fn(top_u_vectors)

    gradient_covariance_matrix = compute_centered_eNTK(params, evaluator.obs_stack, network)
    
    # 2. Initialize base metrics
    metrics = {
        "effective_rank": effective_rank,
        "NTK_rank": eNTK_effective_rank,
        "eNTK": eNTK, # stores the entire 133 x 133 matrix.
        "gradient_covariance_matrix": gradient_covariance_matrix,
        "feature_singular_values": feature_singular_values,
        "feature_top_singular_vectors": feature_top_singular_vectors,
        "Jacobian_top_singular_vectors": heatmaps_stack,
        "jacobian_singular_values": Sj,
        "Direlechet_energy": Direlechet_energy,
    }

    if log_variance:
        from core.variance_metrics import compute_all_variance_metrics
        var_metrics = compute_all_variance_metrics(
            evaluator, network, params, lmbda=lmbda, random_policy=random_policy, target_policy_fn=target_policy_fn
        )
        metrics.update(var_metrics)

    return metrics