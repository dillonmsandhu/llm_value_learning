from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any
from flax import struct
from gymnax.environments import environment, spaces


class EightRoomsExactValue:
    """
    Exact policy evaluation for Eight Rooms (4 rooms high x 2 rooms wide).
    Constructed by repeating Four Rooms twice vertically with connecting doorways
    along the hallway columns, start in the top left room (3, 1) and goal in the
    bottom right room (23, 11).
    """

    def __init__(
        self,
        height: int = 25,
        width: int = 13,
        fail_prob: float = 0.15,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = True,
        goal_pos: Tuple[int, int] | None = None,
        start_pos: Tuple[int, int] | None = None,
        size: int | None = None,
    ):
        self.height = int(height)
        self.width = int(width)
        self.N = self.height  # For backwards compatibility if self.N is accessed
        self.fail_prob = float(fail_prob)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs

        # 1. Generate Map (H, W)
        self.env_map = self._generate_eight_rooms_map()
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)

        # 2. Coordinates: Matrix Indexing (Row, Col) -> [y, x]
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)

        self.num_states = int(self.coords.shape[0])
        self.num_actions = 4
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1

        # 3. Directions: [dy, dx]
        # 0: Up (-1, 0), 1: Right (0, 1), 2: Down (1, 0), 3: Left (0, -1)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        # 4. Default Goal: Bottom right room (Row 23, Col 11)
        default_goal = jnp.array([23, 11], dtype=jnp.int32)
        if goal_pos is None:
            is_valid = bool(self.env_map[default_goal[0], default_goal[1]])
            self.goal = default_goal if is_valid else self.coords[-1]
        else:
            g = jnp.array(goal_pos, dtype=jnp.int32)
            if not bool(self.env_map[g[0], g[1]]):
                raise ValueError(f"goal_pos={goal_pos} (y,x) is not a free cell.")
            self.goal = g

        # 5. Default Start: Top left room (Row 3, Col 1)
        default_start = jnp.array([3, 1], dtype=jnp.int32)
        if start_pos is None:
            is_valid_start = bool(self.env_map[default_start[0], default_start[1]])
            self.start = default_start if is_valid_start else self.coords[0]
        else:
            self.start = jnp.array(start_pos, dtype=jnp.int32)

        self.start_idx = self._coord_to_idx(self.start)
        self.goal_idx = self._coord_to_idx(self.goal)

        # 6. Build Observations
        self.obs_stack = self._build_obs_stack()

        # 7. Build Dynamics
        self.P, self.R = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    @staticmethod
    def _generate_eight_rooms_map() -> jax.Array:
        eight_rooms_str = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xx xxxxxx xxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""
        lines = eight_rooms_str.strip().split("\n")
        bool_map = []
        for row in lines:
            bool_map.append([r == " " for r in row])
        return jnp.array(bool_map)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))

    def _build_obs_stack(self) -> jax.Array:
        """Constructs observation tensors (visual grid or vector positions)."""
        if self.use_visual_obs:
            agent_maps = np.zeros((self.num_states, self.height, self.width), dtype=np.float32)
            y_coords = self.coords[:, 0]
            x_coords = self.coords[:, 1]
            agent_maps[np.arange(self.num_states), y_coords, x_coords] = 1.0

            wall_stack = np.broadcast_to(self.occupied_map, agent_maps.shape)
            obs = np.stack([wall_stack, agent_maps], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)
        else:
            pos_yx = self.coords
            goal_yx = self.goal
            goal_stack = np.broadcast_to(goal_yx[None, :], pos_yx.shape)
            return jnp.asarray(np.concatenate([pos_yx, goal_stack], axis=1), dtype=jnp.float32)

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        proposed = pos + self.directions[action]
        can_move = self.env_map[proposed[0], proposed[1]]
        return jax.lax.select(can_move, proposed, pos)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)

        p_correct = 1.0 - self.fail_prob
        p_wrong = self.fail_prob / 3.0

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx]

            # --- 1. GOAL STATE ---
            if s_idx == self.goal_idx:
                if not continuing:
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :, self.terminal_idx] = 1.0
                else:
                    P[s_idx, :, self.start_idx] = 1.0
                    R[s_idx, :, self.start_idx] = 1.0
                continue

            # --- 2. STANDARD STATES ---
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    prob = p_correct if executed_a == chosen_a else p_wrong
                    if prob == 0:
                        continue

                    next_pos = self._step_pos(pos, executed_a)
                    hits_goal = bool(jnp.all(next_pos == self.goal))

                    if hits_goal:
                        next_idx = self.goal_idx
                        P[s_idx, chosen_a, next_idx] += prob
                        R[s_idx, chosen_a, next_idx] = 0.0
                    else:
                        next_idx = self._coord_to_idx(next_pos)
                        P[s_idx, chosen_a, next_idx] += prob

        # Terminal state dynamics
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :, self.terminal_idx] = 0.0

        return jnp.asarray(P), jnp.asarray(R)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        R_pi = jnp.einsum("sa,sam,sam->s", pi, P_env, R_env)
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """Map per-state values to (height, width) grid."""
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]
        grid = jnp.zeros((self.height, self.width), dtype=values.dtype)
        return grid.at[self.coords[:, 0], self.coords[:, 1]].set(values)

    def get_optimal_value_function(self, tol=1e-6, max_iters=1000):
        V = jnp.zeros(self.num_total_states)
        R_expected = jnp.einsum("sam,sam->sa", self.P, self.R)

        def body_fun(val):
            i, V, delta = val
            expected_v = jnp.einsum("sam,m->sa", self.P, V)
            Q = R_expected + self.gamma * expected_v
            V_new = jnp.max(Q, axis=-1)
            V_new = V_new.at[self.terminal_idx].set(0.0)
            delta = jnp.max(jnp.abs(V_new - V))
            return (i + 1, V_new, delta)

        def cond_fun(val):
            i, V, delta = val
            return jnp.logical_and(i < max_iters, delta > tol)

        _, V_star, _ = jax.lax.while_loop(cond_fun, body_fun, (0, V, 1.0))
        return V_star

    def compute_true_values_raw(self, pi: jax.Array) -> jax.Array:
        return self.solve_linear_system(pi, self.P, self.R)

    def compute_true_values(self, pi: jax.Array) -> jax.Array:
        V_pi = self.compute_true_values_raw(pi)
        return self.get_value_grid(V_pi)

    def compute_stationary_distribution_raw(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array]:
        P_env = self.P_cont[:self.num_states, :, :self.num_states]
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
        valid_mask = self.occupied_map == 0
        mu_flat = mu[valid_mask]
        v_pred_flat = v_pred[valid_mask]
        v_true_flat = v_true[valid_mask]
        return jnp.sum(mu_flat * (v_pred_flat - v_true_flat) ** 2)

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

    def compute_shortest_path_distances(self) -> np.ndarray:
        """
        Computes the shortest path distance (geodesic distance on the maze grid)
        from every state to the goal state using BFS.
        Returns an array of shape (num_total_states,) where terminal_idx has distance 0.
        """
        from collections import deque

        coords_arr = np.array(self.coords)
        coords_to_idx = {tuple(c): i for i, c in enumerate(coords_arr)}
        adj = [[] for _ in range(self.num_states)]
        for i, (y, x) in enumerate(coords_arr):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = int(y + dy), int(x + dx)
                if (ny, nx) in coords_to_idx:
                    adj[i].append(coords_to_idx[(ny, nx)])

        dist = np.zeros(self.num_total_states, dtype=np.float32)
        dist_states = np.full(self.num_states, -1, dtype=np.int32)
        dist_states[self.goal_idx] = 0
        q = deque([self.goal_idx])
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if dist_states[nbr] == -1:
                    dist_states[nbr] = dist_states[curr] + 1
                    q.append(nbr)

        dist[: self.num_states] = dist_states.astype(np.float32)
        dist[self.terminal_idx] = 0.0
        return dist


class ContinuingEightRooms(EightRoomsExactValue):
    """Continuing variant of Eight Rooms where goal transitions cycle back to start."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P, self.R = self._build_env_dynamics(continuing=True)
        self.P_cont = self.P


class EightRoomsDenseExactValue(EightRoomsExactValue):
    """
    Exact policy evaluation for Eight Rooms with Potential-Based Reward Shaping (PBRS).
    Uses the negative shortest-path geodesic distance to the goal as potential:
        Phi(s) = - potential_scale * D(s, goal), with Phi(terminal) = 0.
        F(s, a, s') = gamma * Phi(s') - Phi(s)
        R_dense(s, a, s') = R_sparse(s, a, s') + F(s, a, s')
    Preserves policy ranking, optimal policy, and action advantages while providing
    a dense reward signal along every transition leading to the goal.
    """

    def __init__(self, *args, potential_scale: float = 0.03125, **kwargs):
        self.potential_scale = float(potential_scale)
        super().__init__(*args, **kwargs)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P, R_sparse = super()._build_env_dynamics(continuing=continuing)

        # Compute shortest-path distance to goal
        self.distances = self.compute_shortest_path_distances()

        # Potential function: Phi(s) = - potential_scale * D(s)
        phi = - self.potential_scale * self.distances
        phi[self.terminal_idx] = 0.0
        self.potential = jnp.asarray(phi, dtype=jnp.float32)

        # Shaping tensor: F(s, a, s') = gamma * Phi(s') - Phi(s)
        gamma = self.gamma
        F = gamma * phi[None, None, :] - phi[:, None, None]
        # Terminal state self-loop has 0 shaping reward
        F[self.terminal_idx, :, self.terminal_idx] = 0.0

        R_dense = np.array(R_sparse) + F
        # Mask where P == 0
        R_dense = np.where(np.array(P) > 0, R_dense, 0.0)

        return P, jnp.asarray(R_dense, dtype=jnp.float32)


class ContinuingEightRoomsDense(EightRoomsDenseExactValue):
    """Continuing variant of Eight Rooms Dense where goal transitions cycle back to start."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P, self.R = self._build_env_dynamics(continuing=True)
        self.P_cont = self.P


# Gymnax environment implementation for EightRooms
@struct.dataclass
class EightRoomsState(environment.EnvState):
    pos: jax.Array
    goal: jax.Array
    time: int


@struct.dataclass
class EightRoomsParams(environment.EnvParams):
    fail_prob: float = 0.01
    max_steps_in_episode: int = 1000


try:
    _EnvBase = environment.Environment[EightRoomsState, EightRoomsParams]
except (TypeError, AttributeError):
    _EnvBase = environment.Environment

class EightRooms(_EnvBase):
    """Gymnax-compatible Environment for EightRooms."""

    def __init__(
        self,
        use_visual_obs: bool = True,
        goal_fixed: Tuple[int, int] = (23, 11),
        pos_fixed: Tuple[int, int] = (3, 1),
    ):
        super().__init__()
        self.height = 25
        self.width = 13
        self.use_visual_obs = use_visual_obs
        self.goal_fixed = jnp.array(goal_fixed, dtype=jnp.int32)
        self.pos_fixed = jnp.array(pos_fixed, dtype=jnp.int32)

        self.env_map = EightRoomsExactValue._generate_eight_rooms_map()
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)

        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        self.num_states = int(self.coords.shape[0])

        # 0: Up, 1: Right, 2: Down, 3: Left
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

    @property
    def default_params(self) -> EightRoomsParams:
        return EightRoomsParams()

    def step_env(
        self,
        key: jax.Array,
        state: EightRoomsState,
        action: int | float | jax.Array,
        params: EightRoomsParams,
    ) -> Tuple[jax.Array, EightRoomsState, jax.Array, jax.Array, dict]:
        key_prob, key_action = jax.random.split(key)
        p_roll = jax.random.uniform(key_prob)
        random_action = jax.random.randint(key_action, (), 0, 4)
        executed_a = jnp.where(p_roll < params.fail_prob, random_action, action)

        proposed_pos = state.pos + self.directions[executed_a]
        can_move = self.env_map[proposed_pos[0], proposed_pos[1]]
        new_pos = jax.lax.select(can_move, proposed_pos, state.pos)

        is_goal = jnp.logical_and(new_pos[0] == state.goal[0], new_pos[1] == state.goal[1])
        reward = is_goal.astype(jnp.float32)

        state = EightRoomsState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EightRoomsParams
    ) -> Tuple[jax.Array, EightRoomsState]:
        state = EightRoomsState(pos=self.pos_fixed, goal=self.goal_fixed, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EightRoomsState, params=None, key=None) -> jax.Array:
        if not self.use_visual_obs:
            return jnp.array([state.pos[0], state.pos[1], state.goal[0], state.goal[1]], dtype=jnp.float32)
        else:
            agent_map = jnp.zeros((self.height, self.width), dtype=jnp.float32)
            agent_map = agent_map.at[state.pos[0], state.pos[1]].set(1.0)
            return jnp.stack([self.occupied_map, agent_map], axis=-1)

    def is_terminal(self, state: EightRoomsState, params: EightRoomsParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        done_goal = jnp.logical_and(state.pos[0] == state.goal[0], state.pos[1] == state.goal[1])
        return jnp.logical_or(done_goal, done_steps)

    @property
    def name(self) -> str:
        return "EightRooms-misc"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EightRoomsParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EightRoomsParams) -> spaces.Box:
        if self.use_visual_obs:
            return spaces.Box(0, 1, (self.height, self.width, 2), jnp.float32)
        else:
            return spaces.Box(0, max(self.height, self.width) - 1, (4,), jnp.float32)

    def state_space(self, params: EightRoomsParams) -> spaces.Dict:
        return spaces.Dict({
            "pos": spaces.Box(0, max(self.height, self.width) - 1, (2,), jnp.float32),
            "goal": spaces.Box(0, max(self.height, self.width) - 1, (2,), jnp.float32),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })


@struct.dataclass
class EightRoomsDenseParams(EightRoomsParams):
    gamma: float = 0.99
    potential_scale: float = 1.0


class EightRoomsDense(EightRooms):
    """Gymnax-compatible Environment for EightRooms with potential-based dense reward."""

    def __init__(
        self,
        use_visual_obs: bool = True,
        goal_fixed: Tuple[int, int] = (23, 11),
        pos_fixed: Tuple[int, int] = (3, 1),
        gamma: float = 0.99,
        potential_scale: float = 1.0,
    ):
        super().__init__(
            use_visual_obs=use_visual_obs,
            goal_fixed=goal_fixed,
            pos_fixed=pos_fixed,
        )
        self.gamma = float(gamma)
        self.potential_scale = float(potential_scale)
        self.dist_grid = self._compute_distance_grid()

    def _compute_distance_grid(self) -> jax.Array:
        from collections import deque

        coords_arr = np.array(self.coords)
        coords_to_idx = {tuple(c): i for i, c in enumerate(coords_arr)}
        goal_tuple = tuple(int(x) for x in self.goal_fixed)
        goal_idx = coords_to_idx[goal_tuple]

        adj = [[] for _ in range(self.num_states)]
        for i, (y, x) in enumerate(coords_arr):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = int(y + dy), int(x + dx)
                if (ny, nx) in coords_to_idx:
                    adj[i].append(coords_to_idx[(ny, nx)])

        dist_states = np.full(self.num_states, -1, dtype=np.int32)
        dist_states[goal_idx] = 0
        q = deque([goal_idx])
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if dist_states[nbr] == -1:
                    dist_states[nbr] = dist_states[curr] + 1
                    q.append(nbr)

        grid = np.zeros((self.height, self.width), dtype=np.float32)
        for i, (y, x) in enumerate(coords_arr):
            grid[y, x] = float(dist_states[i])
        return jnp.asarray(grid, dtype=jnp.float32)

    @property
    def default_params(self) -> EightRoomsDenseParams:
        return EightRoomsDenseParams(gamma=self.gamma, potential_scale=self.potential_scale)

    def step_env(
        self,
        key: jax.Array,
        state: EightRoomsState,
        action: int | float | jax.Array,
        params: EightRoomsDenseParams,
    ) -> Tuple[jax.Array, EightRoomsState, jax.Array, jax.Array, dict]:
        key_prob, key_action = jax.random.split(key)
        p_roll = jax.random.uniform(key_prob)
        random_action = jax.random.randint(key_action, (), 0, 4)
        executed_a = jnp.where(p_roll < params.fail_prob, random_action, action)

        proposed_pos = state.pos + self.directions[executed_a]
        can_move = self.env_map[proposed_pos[0], proposed_pos[1]]
        new_pos = jax.lax.select(can_move, proposed_pos, state.pos)

        is_goal = jnp.logical_and(new_pos[0] == state.goal[0], new_pos[1] == state.goal[1])
        base_reward = is_goal.astype(jnp.float32)

        # Potential-based shaping: F = gamma * Phi(s') - Phi(s)
        # Phi(s) = - potential_scale * distance(s)
        # If transitioning to terminal (goal), Phi(s') = 0.0
        cur_dist = self.dist_grid[state.pos[0], state.pos[1]]
        next_dist = self.dist_grid[new_pos[0], new_pos[1]]

        scale = getattr(params, "potential_scale", self.potential_scale)
        gamma = getattr(params, "gamma", self.gamma)

        phi_s = - scale * cur_dist
        phi_next = jnp.where(is_goal, 0.0, - scale * next_dist)
        shaping = gamma * phi_next - phi_s
        reward = base_reward + shaping

        state = EightRoomsState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    @property
    def name(self) -> str:
        return "EightRooms-dense"


# Re-export Continuous control variants
from envs.continuous_eightrooms import (
    ContinuousEightRooms,
    ContinuousEightRoomsDense,
    ContinuousEightRoomsState,
    ContinuousEightRoomsParams,
    ContinuousEightRoomsDenseParams,
)

