from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any


class WhirlpoolExactValue:
    """
    Exact policy evaluation for the Whirlpool Environment using GYMNAX COORDINATES [y, x].
    A highly rotationally-asymmetric MDP designed to stress-test TD learning's K matrix.
    Uses reward shape R[s, a, s'] matching FourRoomsExactValue.
    """

    def __init__(
        self,
        size: int = 20,
        fail_prob: float = 0.9,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = True,
        goal_pos: Tuple[int, int] | None = None,
        start_pos: Tuple[int, int] | None = None,
    ):
        self.N = int(size)
        assert self.N % 2 == 1, "must be of size odd x odd, gave an even number."
        self.fail_prob = float(fail_prob)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs

        # 1. Generate Map (H, W) - Fully open grid for the whirlpool
        self.env_map = jnp.ones((self.N, self.N), dtype=bool)
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)

        # 2. Coordinates: Gymnax uses Matrix Indexing (Row, Col) -> [y, x]
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)

        self.num_states = int(self.coords.shape[0])
        self.num_actions = 4
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1

        # 3. Directions: Match Gymnax (y, x) definition
        # 0: Up (-1, 0), 1: Right (0, 1), 2: Down (1, 0), 3: Left (0, -1)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        # 4. Goal is center by default
        center_idx = self.N // 2
        default_goal = jnp.array([center_idx, center_idx], dtype=jnp.int32)
        if goal_pos is None:
            self.goal = default_goal
        else:
            self.goal = jnp.array(goal_pos, dtype=jnp.int32)
        self.goal_idx = self._coord_to_idx(self.goal)

        # 5. Determine the outer ring (for uniform random reset)
        outer_ring_coords = []
        outer_ring_indices = []
        max_d = self.N // 2

        for i in range(self.num_states):
            py, px = int(self.coords[i][0]), int(self.coords[i][1])
            if max(abs(py - center_idx), abs(px - center_idx)) == max_d:
                outer_ring_coords.append([py, px])
                outer_ring_indices.append(i)

        self.outer_ring_coords = jnp.array(outer_ring_coords, dtype=jnp.int32)
        self.outer_ring_indices = jnp.array(outer_ring_indices, dtype=jnp.int32)

        # 6. Start Position
        if start_pos is None:
            self.start = self.outer_ring_coords[0]
            self.start_idx = int(self.outer_ring_indices[0])
            self.start_pos_provided = False
        else:
            self.start = jnp.array(start_pos, dtype=jnp.int32)
            self.start_idx = self._coord_to_idx(self.start)
            self.start_pos_provided = True

        # 7. Build Observations
        self.obs_stack = self._build_obs_stack()

        # 8. Build Dynamics
        self.P, self.R = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))

    def _build_obs_stack(self) -> jax.Array:
        if self.use_visual_obs:
            agent_maps = np.zeros((self.num_states, self.N, self.N), dtype=np.float32)
            y_coords = self.coords[:, 0]
            x_coords = self.coords[:, 1]
            agent_maps[np.arange(self.num_states), x_coords, y_coords] = 1.0
            wall_stack = np.broadcast_to(self.occupied_map, agent_maps.shape)
            obs = np.stack([wall_stack, agent_maps], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)
        else:
            pos_yx = self.coords
            goal_stack = np.broadcast_to(self.goal[None, :], pos_yx.shape)
            return jnp.asarray(np.concatenate([pos_yx, goal_stack], axis=1), dtype=jnp.float32)

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        proposed = pos + self.directions[action]
        proposed = jnp.clip(proposed, 0, self.N - 1)
        return proposed

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        """
        Builds P and R of shape (num_total_states, num_actions, num_total_states).

        LOGIC:
        1. Rewards: Shifted to 'Exit' to ensure V(Goal) = 1.0 (matching FourRoomsExactValue).
           - V(Goal) = R_exit (1.0) + gamma*0 = 1.0
           - V(Neighbor) = R_entry (0.0) + gamma*V(Goal) = gamma*1.0

        2. Dynamics: Matches Whirlpool physics rules.
        """
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)

        cy, cx = self.N // 2, self.N // 2
        max_d = self.N // 2

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx]
            y, x = int(pos[0]), int(pos[1])

            # --- 1. GOAL STATE (Source of Value 1.0) ---
            if s_idx == self.goal_idx:
                if not continuing:
                    # Episodic: Goal -> Terminal
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :, self.terminal_idx] = 1.0  # Reward 1.0 on Exit
                else:
                    # Continuing: Goal -> Outer Ring
                    for idx in self.outer_ring_indices:
                        P[s_idx, :, idx] = 1.0 / len(self.outer_ring_indices)
                        R[s_idx, :, idx] = 1.0  # Reward 1.0 on Exit
                continue

            # --- 2. DETERMINE RING DISTANCE ---
            d = max(abs(y - cy), abs(x - cx))

            # --- 3. MAP ACTIONS (With exact corner handling) ---
            # 0: Up, 1: Right, 2: Down, 3: Left
            if y == cy - d and x < cx + d:     # Top edge
                cw, ccw, inward, outward = 1, 3, 2, 0
            elif x == cx + d and y < cy + d:   # Right edge
                cw, ccw, inward, outward = 2, 0, 3, 1
            elif y == cy + d and x > cx - d:   # Bottom edge
                cw, ccw, inward, outward = 3, 1, 0, 2
            elif x == cx - d and y > cy - d:   # Left edge
                cw, ccw, inward, outward = 0, 2, 1, 3
            else:  # Fallback (Bottom-Left Corner edge case)
                cw, ccw, inward, outward = 0, 2, 1, 3

            # 4. Resolve Actions based on Physics Rules
            for action in range(self.num_actions):
                transitions = []  # (Probability, Executed Cardinal Action)

                if action == cw:
                    transitions.append((1.0, cw))
                elif action == ccw:
                    transitions.append((1.0 - self.fail_prob, ccw))
                    transitions.append((self.fail_prob, cw))
                elif action == outward:
                    transitions.append((1.0, cw))
                elif action == inward:
                    transitions.append((1.0 - self.fail_prob, inward))
                    transitions.append((self.fail_prob, cw))

                # Apply Transitions
                for prob, executed_a in transitions:
                    if prob <= 0.0:
                        continue
                    next_pos = self._step_pos(pos, executed_a)
                    hits_goal = bool(jnp.all(next_pos == self.goal))

                    if hits_goal:
                        next_idx = self.goal_idx
                        P[s_idx, action, next_idx] += prob
                        R[s_idx, action, next_idx] = 0.0
                    else:
                        next_idx = self._coord_to_idx(next_pos)
                        P[s_idx, action, next_idx] += prob
                        R[s_idx, action, next_idx] = 0.0

        # Terminal state dynamics
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :, self.terminal_idx] = 0.0

        return jnp.asarray(P), jnp.asarray(R)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        # 1. Get the state-to-state transition matrix under the policy
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)

        # 2. Expected reward under policy
        R_pi = jnp.einsum("sa,sam,sam->s", pi, P_env, R_env)

        # 3. Solve the standard linear system
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Map per-state values to N x N grid.
        Uses standard matrix indexing [row, col] (which is [y, x]).
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
        return grid.at[self.coords[:, 0], self.coords[:, 1]].set(values)

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

    def compute_true_values_raw(self, pi: jax.Array) -> jax.Array:
        """Returns vector"""
        V_pi = self.solve_linear_system(pi, self.P, self.R)
        return V_pi

    def compute_true_values(self, pi: jax.Array) -> jax.Array:
        """Returns grid"""
        V_pi = self.compute_true_values_raw(pi)
        return self.get_value_grid(V_pi)

    def compute_stationary_distribution_raw(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns vector and P_pi"""
        P_env = self.P_cont[:self.num_states, :, :self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)

        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)

        b = jnp.zeros(self.num_states).at[-1].set(1.0)

        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum(), P_pi

    def compute_stationary_distribution(self, pi: jax.Array) -> jax.Array:
        """Returns grid"""
        mu, _ = self.compute_stationary_distribution_raw(pi)
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
        if self.start_pos_provided:
            rho_0 = rho_0.at[self.start_idx].set(1.0)
        else:
            rho_0 = rho_0.at[self.outer_ring_indices].set(1.0 / len(self.outer_ring_indices))
        A = jnp.eye(self.num_states) - self.gamma * P_pi.T
        d_gamma = jnp.linalg.solve(A, (1 - self.gamma) * rho_0)
        d_gamma_norm = d_gamma / jnp.sum(d_gamma)
        return d_gamma_norm

    def compute_discounted_visitation(self, pi: jax.Array) -> jax.Array:
        mu = self.compute_discounted_visitation_raw(pi)
        return self.get_value_grid(mu)


class ContinuingWhirlpool(WhirlpoolExactValue):
    """Continuing variant of Whirlpool where goal transitions cycle back to outer ring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P, self.R = self._build_env_dynamics(continuing=True)
        self.P_cont = self.P
