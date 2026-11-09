import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any


class MountainCarExactValue:
    """
    Exact policy evaluation for a discretized Mountain Car environment.
    Discretizes position and velocity into a grid.
    Matches the "reward-on-transition-to-goal" and shifted linear system from FourRoomsExactValue.
    """

    def __init__(
        self,
        n_pos: int = 32,
        n_vel: int = 32,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = False,
        scale_obs: bool = True,
        goal_reward: float = 100.0,
        min_position: float = -1.2,
        max_position: float = 0.6,
        max_speed: float = 0.07,
        goal_position: float = 0.5,
        goal_velocity: float = 0.0,
        force: float = 0.001,
        gravity: float = 0.0025,
    ):
        self.n_pos = int(n_pos)
        self.n_vel = int(n_vel)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs
        self.scale_obs = scale_obs
        self.goal_reward = float(goal_reward)

        # Continuous bounds
        self.min_position = float(min_position)
        self.max_position = float(max_position)
        self.max_speed = float(max_speed)
        self.goal_position = float(goal_position)
        self.goal_velocity = float(goal_velocity)
        self.force = float(force)
        self.gravity = float(gravity)

        self.num_actions = 3

        # Discretize states
        self.pos_bins = np.linspace(self.min_position, self.max_position, self.n_pos)
        self.vel_bins = np.linspace(-self.max_speed, self.max_speed, self.n_vel)

        # Build coordinate arrays
        # coords[s_idx] = [pos, vel]
        self.num_states = self.n_pos * self.n_vel

        # We will add a singular terminal state
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1

        self.coords = np.zeros((self.num_states, 2), dtype=np.float32)
        idx = 0
        for i in range(self.n_pos):
            for j in range(self.n_vel):
                self.coords[idx] = [self.pos_bins[i], self.vel_bins[j]]
                idx += 1

        # Determine default start state (pos around -0.5, vel 0.0)
        start_pos_idx = np.argmin(np.abs(self.pos_bins - (-0.5)))
        start_vel_idx = np.argmin(np.abs(self.vel_bins - 0.0))
        self.start_idx = start_pos_idx * self.n_vel + start_vel_idx
        self.start = np.array([-0.5, 0.0], dtype=np.float32)

        # Goal state representation
        goal_pos_idx = np.argmin(np.abs(self.pos_bins - self.goal_position))
        goal_vel_idx = np.argmin(np.abs(self.vel_bins - self.goal_velocity))
        self.goal_idx = goal_pos_idx * self.n_vel + goal_vel_idx
        self.goal = np.array([self.goal_position, self.goal_velocity], dtype=np.float32)

        self.occupied_map = np.ones((self.n_pos, self.n_vel), dtype=bool)
        self.reset_indices = np.array([self.start_idx], dtype=np.int32)
        self.fail_prob = 0.0

        # Build Observations
        self.obs_stack = self._build_obs_stack()

        # Build Dynamics
        self.P, self.R = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _build_obs_stack(self) -> jax.Array:
        if self.use_visual_obs:
            # Output Shape: (num_states, n_pos, n_vel, 2)
            obs = np.zeros((self.num_states, self.n_pos, self.n_vel, 2), dtype=np.float32)
            # Channel 0: One-hot indicator of position and velocity in phase space
            pos_indices = np.arange(self.num_states) // self.n_vel
            vel_indices = np.arange(self.num_states) % self.n_vel
            obs[np.arange(self.num_states), pos_indices, vel_indices, 0] = 1.0

            # Channel 1: Track height profile (sin(3x) + 1) / 2
            height = (np.sin(3.0 * self.pos_bins) + 1.0) / 2.0  # shape (n_pos,)
            obs[:, :, :, 1] = height[:, None]
            return jnp.asarray(obs, dtype=jnp.float32)
        else:
            obs = np.copy(self.coords)
            if self.scale_obs:
                # Scale position to [-1, 1]
                obs[:, 0] = 2.0 * (obs[:, 0] - self.min_position) / (self.max_position - self.min_position) - 1.0
                # Scale velocity to [-1, 1]
                obs[:, 1] = 2.0 * (obs[:, 1] - (-self.max_speed)) / (2 * self.max_speed) - 1.0
            return jnp.asarray(obs, dtype=jnp.float32)

    def _get_bin_idx(self, pos: float, vel: float) -> int:
        pos_idx = np.argmin(np.abs(self.pos_bins - pos))
        vel_idx = np.argmin(np.abs(self.vel_bins - vel))
        return pos_idx * self.n_vel + vel_idx

    def _step_continuous(self, pos: float, vel: float, action: int) -> Tuple[float, float, bool]:
        # action: 0 (left), 1 (idle), 2 (right)
        new_vel = vel + (action - 1) * self.force - np.cos(3 * pos) * self.gravity
        new_vel = np.clip(new_vel, -self.max_speed, self.max_speed)

        new_pos = pos + new_vel
        new_pos = np.clip(new_pos, self.min_position, self.max_position)

        if new_pos == self.min_position and new_vel < 0:
            new_vel = 0.0

        is_goal = (new_pos >= self.goal_position) and (new_vel >= self.goal_velocity)
        return new_pos, new_vel, is_goal

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)

        for s_idx in range(self.num_states):
            pos, vel = self.coords[s_idx]
            
            # Goal region check
            if pos >= self.goal_position and vel >= self.goal_velocity:
                if not continuing:
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :, self.terminal_idx] = self.goal_reward
                else:
                    P[s_idx, :, self.start_idx] = 1.0
                    R[s_idx, :, self.start_idx] = self.goal_reward
                continue
                
            for a in range(self.num_actions):
                next_pos, next_vel, _ = self._step_continuous(pos, vel, a)
                
                # Replace naive rounding with stochastic bilinear interpolation
                transitions = self._get_interpolated_bins(next_pos, next_vel)
                for next_idx, prob in transitions:
                    P[s_idx, a, next_idx] += prob
                    R[s_idx, a, next_idx] = 0.0

        # --- TERMINAL STATE ---
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :, self.terminal_idx] = 0.0
        
        return jnp.asarray(P), jnp.asarray(R)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        # 1. State-to-state transition matrix under policy
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)

        # 2. Expected reward under policy
        R_pi = jnp.einsum("sa,sam,sam->s", pi, P_env, R_env)

        # 3. Solve linear system
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Map per-state values to (n_pos, n_vel) grid.
        Grid index is [pos_idx, vel_idx].
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        grid = jnp.zeros((self.n_pos, self.n_vel), dtype=values.dtype)
        # indices are simply mapped sequentially
        return values.reshape((self.n_pos, self.n_vel))

    def get_optimal_value_function(self, tol=1e-6, max_iters=1000):
        V = jnp.zeros(self.num_total_states)
        R_expected = jnp.einsum("sam,sam->sa", self.P, self.R)
        
        def body_fun(val):
            i, V, delta = val
            expected_v = jnp.einsum("sam,m->sa", self.P, V)
            Q = R_expected + self.gamma * expected_v
            V_new = jnp.max(Q, axis=-1)
            # Ensure terminal state value remains 0
            V_new = V_new.at[self.terminal_idx].set(0.0)
            delta = jnp.max(jnp.abs(V_new - V))
            return (i + 1, V_new, delta)
        
        def cond_fun(val):
            i, V, delta = val
            return jnp.logical_and(i < max_iters, delta > tol)
        
        _, V_star, _ = jax.lax.while_loop(cond_fun, body_fun, (0, V, 1.0))
        return V_star

    def compute_true_values_raw(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        V_pi = self.solve_linear_system(pi, self.P, self.R)
        return V_pi

    def compute_true_values(self, pi: jax.Array) -> jax.Array:
        V_pi = self.compute_true_values_raw(pi)
        return self.get_value_grid(V_pi)

    def compute_stationary_distribution_raw(self, pi: jax.Array) -> jax.Array:
        P_env = self.P_cont[: self.num_states, :, : self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)

        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)

        b = jnp.zeros(self.num_states).at[-1].set(1.0)

        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum(), P_pi

    def compute_stationary_distribution(self, pi: jax.Array) -> jax.Array:
        mu, _ = self.compute_stationary_distribution_raw(pi)
        return self.get_value_grid(mu)

    def compute_v_error_on_d_pi(self, pi: jax.Array, v_pred: jax.Array, v_true: jax.Array):
        mu = self.compute_stationary_distribution(pi)
        # All states are valid in MountainCar
        return jnp.sum(mu * (v_pred - v_true) ** 2)

    def compute_discounted_visitation_raw(self, pi: jax.Array) -> jax.Array:
        P_env = self.P[: self.num_states, :, : self.num_states]
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

    def _get_interpolated_bins(self, pos: float, vel: float) -> list[Tuple[int, float]]:
        """Distributes a continuous state into the 4 nearest discrete bins."""
        # Clip to bounds
        pos = np.clip(pos, self.min_position, self.max_position)
        vel = np.clip(vel, -self.max_speed, self.max_speed)

        # Calculate exact fractional indices
        pos_idx_exact = (pos - self.min_position) / (self.max_position - self.min_position) * (self.n_pos - 1)
        vel_idx_exact = (vel - (-self.max_speed)) / (2 * self.max_speed) * (self.n_vel - 1)

        # Identify bounding grid coordinates
        p0 = int(np.floor(pos_idx_exact))
        p1 = min(p0 + 1, self.n_pos - 1)
        v0 = int(np.floor(vel_idx_exact))
        v1 = min(v0 + 1, self.n_vel - 1)

        # Calculate distances from the lower bounds
        dp = pos_idx_exact - p0
        dv = vel_idx_exact - v0

        # Bilinear weights (probabilities)
        w_p0_v0 = (1 - dp) * (1 - dv)
        w_p1_v0 = dp * (1 - dv)
        w_p0_v1 = (1 - dp) * dv
        w_p1_v1 = dp * dv

        transitions = []
        if w_p0_v0 > 0:
            transitions.append((p0 * self.n_vel + v0, w_p0_v0))
        if w_p1_v0 > 0:
            transitions.append((p1 * self.n_vel + v0, w_p1_v0))
        if w_p0_v1 > 0:
            transitions.append((p0 * self.n_vel + v1, w_p0_v1))
        if w_p1_v1 > 0:
            transitions.append((p1 * self.n_vel + v1, w_p1_v1))

        return transitions


class MountainCarDenseExactValue(MountainCarExactValue):
    """
    MountainCar with Potential-Based Reward Shaping based on height (Option A: Zero-offset potential).
    
    Potential:
        Phi(s) = potential_scale * (sin(3x) + 1) / 2 in [0, potential_scale].
    
    Shaping:
        F(s, a, s') = gamma * Phi(s') - Phi(s).
        R_dense(s, a, s') = R_sparse(s, a, s') + F(s, a, s').
    
    Guarantees policy invariance while maintaining return scale identical to sparse return
    without shifting the baseline for failing trajectories.
    """

    def __init__(
        self,
        potential_scale: float = 20.0,
        goal_reward: float = 100.0,
        **kwargs,
    ):
        self.potential_scale = float(potential_scale)
        super().__init__(goal_reward=goal_reward, **kwargs)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P, R_sparse = super()._build_env_dynamics(continuing=continuing)

        # Option A: Zero-offset Potential based on height
        # Phi(s) = potential_scale * (sin(3x) + 1) / 2 in [0, potential_scale]
        phi = np.zeros(self.num_total_states, dtype=np.float32)
        phi[: self.num_states] = self.potential_scale * (np.sin(3.0 * self.coords[:, 0]) + 1.0) / 2.0
        
        # Terminal state retains the goal potential Phi(s_goal) so telescoping returns hold
        phi[self.terminal_idx] = phi[self.goal_idx]
        self.potential = jnp.asarray(phi, dtype=jnp.float32)

        gamma = self.gamma
        F = gamma * phi[None, None, :] - phi[:, None, None]
        # Absorbing self-loop in terminal state has zero shaping reward
        F[self.terminal_idx, :, self.terminal_idx] = 0.0

        R_dense = np.array(R_sparse) + F
        R_dense = np.where(np.array(P) > 0, R_dense, 0.0)
        return P, jnp.asarray(R_dense, dtype=jnp.float32)
