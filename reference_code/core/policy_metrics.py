import jax
import jax.numpy as jnp


def compute_policy_metrics(pi, pi_old, mu=None, mu_old=None, threshold=None):
    """
    Computes tracking metrics between successive policies and state visitation distributions.
    
    Args:
        pi: Current policy matrix of shape (num_states, num_actions)
        pi_old: Previous policy matrix of shape (num_states, num_actions)
        mu: Current stationary distribution of shape (num_states,) or (num_states+1,)
        mu_old: Previous stationary distribution of shape (num_states,) or (num_states+1,)
        threshold: Support threshold for state coverage (default: 0.01 / num_states)
        
    Returns:
        dict containing:
            - policy_tv: Uniform state-average total variation distance between pi and pi_old
            - policy_tv_on_policy: mu_old-weighted total variation distance between pi and pi_old
            - policy_tv_max: Maximum TV distance across all individual states
            - mu_tv: Total variation distance between mu and mu_old
            - state_coverage: Percentage of active states with non-negligible visitation (mu > threshold)
            - state_entropy_coverage: Normalized perplexity exp(H(mu)) / num_states * 100%
    """
    # 1. Total variation distance between policies: D_TV(pi, pi_old) = 0.5 * sum_a |pi(a|s) - pi_old(a|s)|
    tv_per_state = 0.5 * jnp.sum(jnp.abs(pi - pi_old), axis=-1)
    policy_tv = jnp.mean(tv_per_state)
    policy_tv_max = jnp.max(tv_per_state)
    
    metrics = {
        "policy_tv": policy_tv,
        "policy_tv_max": policy_tv_max,
    }
    
    n_states = pi.shape[0]
    
    # 2. TV distance between state distributions: D_TV(mu, mu_old) = 0.5 * sum_s |mu(s) - mu_old(s)|
    if mu is not None and mu_old is not None:
        mu_act = mu[:n_states]
        mu_old_act = mu_old[:n_states]
        
        # On-policy weighted TV distance (ensure normalized over active states)
        mu_old_norm = mu_old_act / (jnp.sum(mu_old_act) + 1e-12)
        policy_tv_on_policy = jnp.sum(mu_old_norm * tv_per_state)
        metrics["policy_tv_on_policy"] = policy_tv_on_policy
        
        mu_tv = 0.5 * jnp.sum(jnp.abs(mu_act - mu_old_act))
        metrics["mu_tv"] = mu_tv
        
        # 3. State coverage: percentage of states with non-negligible support
        thresh = threshold if threshold is not None else (0.01 / n_states)
        coverage = 100.0 * jnp.mean(mu_act > thresh)
        metrics["state_coverage"] = coverage
        
        # Continuous perplexity-based coverage: exp(H(mu)) / n_states * 100%
        # (100% = perfectly uniform coverage over all states, ~0% = single state)
        safe_mu = jnp.maximum(mu_act, 1e-15)
        entropy = -jnp.sum(safe_mu * jnp.log(safe_mu))
        entropy_coverage = 100.0 * jnp.exp(entropy) / n_states
        metrics["state_entropy_coverage"] = entropy_coverage
        
    return metrics
