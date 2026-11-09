from core.imports import *


def compute_feature_jacobian(params, S, network):
    """
    Computes the |S|x|S| empirical NTK matrix and returns its eigenvalues.
    """
    # 1. Define a function that outputs the scalar value for a SINGLE state
    def get_single_v(p, single_s):
        # Add batch dim, apply, squeeze back to scalar
        return network.apply(p, single_s, method=network.value).squeeze()

    # 2. Get the Jacobian of the value with respect to the parameters
    jacobian_fn = jax.jacrev(get_single_v)
    
    # 3. Vectorize over the 133 states in S
    # J is a pytree of the same structure as params, with a batch dimension of 133
    J_pytree = jax.vmap(jacobian_fn, in_axes=(None, 0))(params, S)
    
    # 4. Flatten the gradients for each state into a single vector
    # Shape of J_flat will be (133, total_number_of_parameters)
    leaves = jax.tree_util.tree_leaves(J_pytree)
    J_flat = jnp.concatenate([jnp.reshape(x, (x.shape[0], -1)) for x in leaves], axis=-1)
    
    return J_flat

def compute_eNTK(params, S, network):
    """
    Computes the |S|x|S| empirical NTK matrix and returns its eigenvalues.
    """
    J = compute_feature_jacobian(params, S, network)
    eNTK = J@J.T
    return eNTK

def compute_centered_eNTK(params, S, network):
    """
    Computes the |S|x|S| empirical NTK matrix and returns its eigenvalues.
    """
    J = compute_feature_jacobian(params, S, network)
    J = J-J.mean(axis=0)
    return J@J.T