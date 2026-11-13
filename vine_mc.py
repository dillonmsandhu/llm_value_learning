"""Bounded-memory Monte Carlo value estimates for trunk prefixes."""
import jax
import jax.numpy as jnp


def prefix_values(env, actor, params, states, key, k, batch_size, gamma):
    """K fresh policy completions per prefix; no critic or trunk reuse."""
    shape = states.done.shape
    flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), states)
    total = shape[0] * shape[1] * k
    n_batches = (total + batch_size - 1) // batch_size

    def complete(chunk):
        indices = chunk * batch_size + jnp.arange(batch_size)
        valid = indices < total
        st = jax.tree_util.tree_map(lambda x: x[jnp.minimum(indices // k, len(flat.done)-1)], flat)
        st = st._replace(done=st.done | ~valid)
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(indices)

        def condition(carry):
            step, st, _, _, _, _ = carry
            return (step < env.max_steps) & (~st.done).any()

        def step(carry):
            t, st, ret, discount, steps, keys = carry
            split = jax.vmap(lambda q: jax.random.split(q, 3))(keys)
            logits = actor.apply(params, st.history_tokens)
            action = jax.vmap(jax.random.categorical)(split[:, 1], logits).astype(jnp.int32)
            active = ~st.done
            _, ns, reward, done, _ = jax.vmap(env.step)(split[:, 2], st, action)
            ret += jnp.where(active, discount * reward, 0.)
            discount *= gamma * (1-done)
            return t+1, ns, ret, discount, steps+active.sum(), split[:, 0]

        _, _, ret, _, steps, _ = jax.lax.while_loop(
            condition, step, (jnp.array(0), st, jnp.zeros(batch_size),
                              jnp.ones(batch_size), jnp.array(0), keys))
        return ret, steps

    returns, steps = jax.lax.map(complete, jnp.arange(n_batches))
    values = returns.reshape(-1)[:total].reshape(*shape, k).mean(-1)
    return jax.lax.stop_gradient(values), steps.sum().astype(float)
