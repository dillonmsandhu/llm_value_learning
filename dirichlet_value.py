"""Double-sampled critic objective for e^T D (I-gamma P) e, e=V-v."""
import jax.numpy as jnp


def double_sampled_dirichlet_loss(value, next_value, mc_return, branch_reward,
                                  branch_done, active, gamma=1.0):
    """Unbiased loss for fixed v when G and the branch are independent given s.

    G and the branch transition are data. Both value arguments remain live:
    detaching the next prediction would omit the adjoint part of the gradient.
    The denominator counts fixed rollout slots, including zero-weight padding;
    renormalizing by a random active count can introduce length-dependent weights.
    Individual sampled losses may be negative. This is not squared TD error.
    """
    residual = branch_reward + gamma * (1.0-branch_done) * next_value - value
    return jnp.mean(jnp.where(active, (mc_return-value)*residual, 0.0))


def bilinear_double_sampled_loss(value, trunk_next_value, probe_next_value,
                                 mc_return, trunk_reward, trunk_done,
                                 probe_reward, probe_done, active, gamma=1.0):
    """Gradient surrogate for 0.5 * <V-v, (I-gamma P)(V-v)>_D.

    Matches E_min's Gymnax/Brax e_inner/e_min estimator. A complete trunk
    supplies G and transition 1; transition 2 is conditionally independent.
    Both TD errors can enter the coefficient of grad(v(s)), because that
    gradient depends only on s. Only the independent probe successor may
    multiply the trunk return error. Stop-gradient coefficients implement
    the product rule explicitly; the probe value itself must remain live.

    This scalar is a surrogate, not a numerical estimate of the energy.
    Mean over fixed slots avoids return-dependent active-count weighting.
    Averaging TD errors halves their conditional variance, but covariance
    with G means total gradient variance is not guaranteed to decrease.
    """
    from jax import lax
    delta_1 = trunk_reward + gamma * (1.0-trunk_done) * trunk_next_value - value
    delta_2 = probe_reward + gamma * (1.0-probe_done) * probe_next_value - value
    delta_bar = 0.5 * (delta_1+delta_2)
    error = mc_return-value
    surrogate = (-0.5*lax.stop_gradient(delta_bar+error)*value
                 +0.5*gamma*lax.stop_gradient(error)*(1.0-probe_done)*probe_next_value)
    return jnp.mean(jnp.where(active, surrogate, 0.0))
