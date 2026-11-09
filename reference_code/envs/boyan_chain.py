import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any
from flax import struct
from gymnax.environments import spaces


class ContinuingBoyanRing:
    """
    Exact policy evaluation for a Continuing Boyan Ring.
    A purely forward-progressing, highly stochastic cycle to maximize K.
    Uses standard tabular vector observations [pos, goal].
    """


    def __init__(
        self,
        size: int = 20,
        gamma: float = 0.999,
        episodic: bool = False,
        use_visual_obs: bool = True, # <--- Added flag
    ):
        self.N = int(size)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs # <--- Store flag
        print('use visual obs is ', self.use_visual_obs)

        # Generate Map (1, N) for API compatibility with visual grids
        self.env_map = jnp.ones((1, self.N), dtype=bool)
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)
        
        # Coordinates: 1D positions wrapped in 2D array [y=0, x=pos]
        self.coords = jnp.stack([jnp.zeros(self.N, dtype=jnp.int32), jnp.arange(self.N, dtype=jnp.int32)], axis=1)
        
        self.num_states = int(self.coords.shape[0])
        self.num_actions = 2 
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1
        
        self.goal = jnp.array([0, 0], dtype=jnp.int32)
        self.start = jnp.array([0, self.N - 1], dtype=jnp.int32)
            
        self.start_idx = self.N - 1
        self.goal_idx = 0

        # Build Observations (Now supports Visual & Tabular)
        self.obs_stack = self._build_obs_stack()
        print('shape of obs stack', self.obs_stack.shape)
        
        # Build Dynamics
        self.P, self.R = self._build_env_dynamics()
        self.P_cont = self.P 

    def _build_obs_stack(self) -> jax.Array:
        """
        Supports both Tabular Vector Obs and Visual Grid Obs.
        Returns ONLY the active states
        """
        if self.use_visual_obs:
            agent_maps = np.zeros((self.num_states, 1, self.N), dtype=np.float32)
            y_coords = self.coords[:, 0]
            x_coords = self.coords[:, 1]
            agent_maps[np.arange(self.num_states), y_coords, x_coords] = 1.0

            wall_stack = np.broadcast_to(self.occupied_map, agent_maps.shape)
            obs = np.stack([wall_stack, agent_maps], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)
            
        else:
            pos_x = np.arange(self.num_states)[:, None]
            goal_x = np.zeros_like(pos_x) 
            obs = np.concatenate([pos_x, goal_x], axis=1).astype(np.float32)
            return jnp.asarray(obs)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        # For a 1D ring mapped to [0, x], the index is just x
        return int(coord[1])

    def _build_env_dynamics(self) -> Tuple[jax.Array, jax.Array]:
        """
        Transitions: +1 or +2 steps forward around the ring. No backward steps.
        Reward: +1 for entering state 0.
        """
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions), dtype=np.float32)

        for s_idx in range(self.num_states):
            # Action 0: +1 Step
            next_idx_0 = (s_idx + 1) % self.num_states
            P[s_idx, 0, next_idx_0] = 1.0
            if next_idx_0 == self.goal_idx:
                R[s_idx, 0] = 1.0
                
            # Action 1: +2 Steps
            next_idx_1 = (s_idx + 2) % self.num_states
            P[s_idx, 1, next_idx_1] = 1.0
            if next_idx_1 == self.goal_idx:
                R[s_idx, 1] = 1.0

        # Isolated terminal state
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        
        return jnp.asarray(P), jnp.asarray(R)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        R_pi_delayed = jnp.einsum("sa,sa->s", pi, R_env)
        R_pi_shifted = P_pi @ R_pi_delayed
        
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi_shifted)

    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Maps the 1D state values onto the (1, N) grid for compatibility 
        with the compute_v_error_on_d_pi masking logic.
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        grid = jnp.zeros((1, self.N), dtype=values.dtype)
        return grid.at[self.coords[:, 0], self.coords[:, 1]].set(values)

    def compute_true_values_raw(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        return self.solve_linear_system(pi, self.P, self.R)
    
    def compute_true_values(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        V_pi = self.compute_true_values_raw(pi)
        return self.get_value_grid(V_pi)

    def compute_stationary_distribution_raw(self, pi: jax.Array) -> jax.Array:
        P_env = self.P_cont[:self.num_states, :, :self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)
        b = jnp.zeros(self.num_states)
        b = b.at[-1].set(1.0)
        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum()

    def compute_stationary_distribution(self, pi: jax.Array) -> jax.Array:
        mu = self.compute_stationary_distribution_raw(pi)
        return self.get_value_grid(mu)
    
    def compute_v_error_on_d_pi(self, pi: jax.Array, v_pred: jax.Array, v_true: jax.Array):
        mu = self.compute_stationary_distribution(pi)
        valid_mask = self.occupied_map == 0
        mu_flat = mu[valid_mask]
        v_pred_flat = v_pred[valid_mask]
        v_true_flat = v_true[valid_mask]
        return jnp.sum(mu_flat * (v_pred_flat - v_true_flat)**2)
        
    def compute_discounted_visitation_raw(self, pi: jax.Array) -> jax.Array:
        P_env = self.P[:self.num_states, :, :self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        rho_0 = jnp.zeros(self.num_states)
        rho_0 = rho_0.at[self.start_idx].set(1.0)
        A = jnp.eye(self.num_states) - self.gamma * P_pi.T
        d_gamma = jnp.linalg.solve(A, (1 - self.gamma) * rho_0)
        d_gamma_norm = d_gamma / jnp.sum(d_gamma)
        return d_gamma_norm
    
    def compute_discounted_visitation(self, pi: jax.Array) -> jax.Array:
        mu = self.compute_discounted_visitation_raw(pi)
        return self.get_value_grid(mu)


# Minimal structural mock to satisfy the Gym/Gymnax API wrappers downstream
class MatrixMockEnv:
    def __init__(self, size=20, use_visual_obs=True):
        self.size = size
        self.use_visual_obs = use_visual_obs
        
    def observation_space(self, params):
        if self.use_visual_obs:
            # Returns a 1D grid shape (1, size, 2 channels) matching your visual design
            return spaces.Box(low=0.0, high=1.0, shape=(1, self.size, 2))
        else:
            return spaces.Box(low=0.0, high=float(self.size), shape=(2,))

    def action_space(self, params):
        return spaces.Discrete(2) # 2 Actions: +1 and +2 steps

@struct.dataclass
class BoyanParams:
    fail_prob: float = 0.0
    max_steps_in_episode: int = 100000