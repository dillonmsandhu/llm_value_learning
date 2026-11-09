from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any

class FourRoomsExactValue:
    """
    Exact policy evaluation for Four Rooms using GYMNAX COORDINATES [y, x].
    Replicates the Gymnax visual observation bug (transposed agent).
    """

    def __init__(
        self,
        size: int = 13,
        fail_prob: float = 0.15,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = True,
        goal_pos: Tuple[int, int] | None = None,
        start_pos: Tuple[int, int] | None = None,
    ):
        self.N = int(size)
        self.fail_prob = float(fail_prob)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs

        # 1. Generate Map (H, W)
        self.env_map = self._generate_four_rooms_map()
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)
        
        # 2. Coordinates: Gymnax uses Matrix Indexing (Row, Col) -> [y, x]
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32) # Coordinates are (y,x)
        
        self.num_states = int(self.coords.shape[0])
        self.num_actions = 4
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1
        
        # 3. Directions: Match Gymnax (y, x) definition
        # 0: Up (-1, 0), 1: Right (0, 1), 2: Down (1, 0), 3: Left (0, -1)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        # 4. Default Goal: Gymnax uses [8, 9] (Row 8, Col 9)
        default_goal = jnp.array([8, 9], dtype=jnp.int32)
        
        if goal_pos is None:
            is_valid = bool(self.env_map[default_goal[0], default_goal[1]])
            self.goal = default_goal if is_valid else self.coords[-1]
        else:
            # Assume user provides [y, x] if they are using this class
            g = jnp.array(goal_pos, dtype=jnp.int32)
            if not bool(self.env_map[g[0], g[1]]):
                raise ValueError(f"goal_pos={goal_pos} (y,x) is not a free cell.")
            self.goal = g
        
        if start_pos is None:
            # 5. Default Start: Gymnax uses [4, 1] (Row 4, Col 1)
            default_start = jnp.array([4, 1], dtype=jnp.int32)
        
            is_valid_start = bool(self.env_map[default_start[0], default_start[1]])
            self.start = default_start if is_valid_start else self.coords[0]
        else:
            self.start = jnp.array(start_pos, dtype=jnp.int32)
            # self.start = start_pos
            
        self.start_idx = self._coord_to_idx(self.start)
        self.goal_idx = self._coord_to_idx(self.goal)

        # 6. Build Observations (Replicating Gymnax Logic + Bug)
        self.obs_stack = self._build_obs_stack()
        
        # 7. Build Dynamics
        self.P, self.R = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _generate_four_rooms_map(self) -> jax.Array:
        four_rooms_str = """
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
xxxxxxxxxxxxx"""
        lines = four_rooms_str.strip().split("\n")
        bool_map = []
        for row in lines:
            bool_map.append([r == " " for r in row])
        return jnp.array(bool_map)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))
    
    def _build_obs_stack(self) -> jax.Array:
        """
        Constructs observations. 
        CRITICAL: Replicates Gymnax 'Mirror World' Bug for Visual Obs.
        """
        if self.use_visual_obs:
            # --- Visual Observation ---
            # Output Shape: (N_states, Height, Width, 2)
            
            # 1. Initialize Agent Maps
            agent_maps = np.zeros((self.num_states, self.N, self.N), dtype=np.float32)
            
            # 2. Set Agent Positions
            # self.coords is [y, x] (Row, Col).
            y_coords = self.coords[:, 0]
            x_coords = self.coords[:, 1]
            
            # GYMNAX LOGIC: agent_map.at[pos[1], pos[0]].set(1)
            # pos[1] is x (Col), pos[0] is y (Row).
            # So it writes to grid[x, y].
            # We replicate this transposition:
            agent_maps[np.arange(self.num_states), x_coords, y_coords] = 1.0

            # 3. Wall Map (Gymnax uses standard map for walls)
            wall_stack = np.broadcast_to(self.occupied_map, agent_maps.shape)

            # 4. Stack
            obs = np.stack([wall_stack, agent_maps], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)

        else:
            # --- Vector Observation ---
            # Gymnax returns: [pos[0], pos[1], goal[0], goal[1]]
            # Which is [y, x, y, x]
            
            pos_yx = self.coords
            
            goal_yx = self.goal
            goal_stack = np.broadcast_to(goal_yx[None, :], pos_yx.shape)
            
            return jnp.asarray(np.concatenate([pos_yx, goal_stack], axis=1), dtype=jnp.float32)

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        # pos is [y, x], directions are [dy, dx]
        proposed = pos + self.directions[action]
        
        # Check collision in map[y, x]
        # proposed[0] is y, proposed[1] is x
        can_move = self.env_map[proposed[0], proposed[1]]
        
        return jax.lax.select(can_move, proposed, pos)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        """
        Builds P and R.
        
        LOGIC:
        1. Rewards: Shifted to 'Exit' to ensure V(Goal) = 1.0.
           - V(Goal) = R_exit (1.0) + gamma*0 = 1.0
           - V(Neighbor) = R_entry (0.0) + gamma*V(Goal) = 0.99
           
        2. Dynamics: MATCHES GYMNAX EXACTLY.
           - P(intended) = 1 - fail_prob
           - P(wrong)    = fail_prob / 3
        """
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)

        p_correct = 1.0 - self.fail_prob
        p_wrong = self.fail_prob / 3.0

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx] # (y, x)

            # --- 1. GOAL STATE (Source of Value 1.0) ---
            if s_idx == self.goal_idx:
                if not continuing:
                    # Episodic: Goal -> Terminal
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :, self.terminal_idx] = 1.0  # Reward 1.0 on Exit
                else:
                    # Continuing: Goal -> Start
                    P[s_idx, :, self.start_idx] = 1.0
                    R[s_idx, :, self.start_idx] = 1.0
                continue

            # --- 2. STANDARD STATES ---
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    # Use the Gymnax Probability Logic
                    prob = p_correct if executed_a == chosen_a else p_wrong
                    
                    if prob == 0: continue
                        
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

    # def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
    #     P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
    #     R_pi = jnp.einsum("sa,sa->s", pi, R_env)
    #     A = jnp.eye(self.num_total_states) - self.gamma * P_pi
    #     return jnp.linalg.solve(A, R_pi)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        # 1. Get the state-to-state transition matrix under the policy
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        
        # 2. Expected reward under policy
        R_pi = jnp.einsum("sa,sam,sam->s", pi, P_env, R_env)
        
        # 3. Solve the standard linear system
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    # def get_value_grid(self, values: jax.Array) -> jax.Array:
    #         """
    #         Map per-state values to N x N grid.
    #         MATCHES VISUAL OBS: Writes to grid[x, y] (Transposed).
    #         """
    #         if values.shape[0] == self.num_total_states:
    #             values = values[: self.num_states]

    #         grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
            
    #         # self.coords is [y, x] (Row, Col)
    #         # Visual Obs puts agent at [x, y].
    #         # match visual obs for comparison.
    #         return grid.at[self.coords[:, 1], self.coords[:, 0]].set(values)


    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Map per-state values to N x N grid.
        Uses standard matrix indexing [row, col] (which is [y, x]).
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
        
        # self.coords is already [y, x] (Row, Col)
        # We now place them correctly into grid[row, col]
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

    def compute_true_values_raw(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        "returns vector"
        V_pi = self.solve_linear_system(pi, self.P, self.R)
        return V_pi
    
    def compute_true_values(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        "Returns grid"
        V_pi = self.compute_true_values_raw(pi)
        return self.get_value_grid(V_pi)

    def compute_stationary_distribution_raw(self, pi: jax.Array) -> jax.Array:
        """Returns vector"""
        P_env = self.P_cont[:self.num_states, :, :self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        
        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)
        
        # Can be written directly:
        b = jnp.zeros(self.num_states).at[-1].set(1.0)
        
        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum(), P_pi

    def compute_stationary_distribution(self, pi: jax.Array) -> jax.Array:
        "Returns grid"
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


class FourRoomsDenseExactValue(FourRoomsExactValue):
    """
    Exact policy evaluation for Four Rooms with Potential-Based Reward Shaping (PBRS).
    Uses the negative shortest-path geodesic distance to the goal as potential:
        Phi(s) = - potential_scale * D(s, goal), with Phi(terminal) = 0.
        F(s, a, s') = gamma * Phi(s') - Phi(s)
        R_dense(s, a, s') = R_sparse(s, a, s') + F(s, a, s')
    """

    def __init__(self, *args, potential_scale: float = 0.03125, **kwargs):
        self.potential_scale = float(potential_scale)
        super().__init__(*args, **kwargs)

    def compute_shortest_path_distances(self) -> np.ndarray:
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

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P, R_sparse = super()._build_env_dynamics(continuing=continuing)
        self.distances = self.compute_shortest_path_distances()
        phi = - self.potential_scale * self.distances
        phi[self.terminal_idx] = 0.0
        self.potential = jnp.asarray(phi, dtype=jnp.float32)

        gamma = self.gamma
        F = gamma * phi[None, None, :] - phi[:, None, None]
        F[self.terminal_idx, :, self.terminal_idx] = 0.0

        R_dense = np.array(R_sparse) + F
        R_dense = np.where(np.array(P) > 0, R_dense, 0.0)
        return P, jnp.asarray(R_dense, dtype=jnp.float32)


class FourRoomsMinesExactValue(FourRoomsExactValue):
    """
    Four Rooms environment with lethal mines placed along corridors.
    Stepping on a mine terminates the agent and resets the trajectory to the start state.
    Mines are strictly forbidden from doorway cells to ensure all rooms remain accessible.
    """
    DEFAULT_MINES = [(3, 3), (5, 8), (8, 2), (9, 9)]
    DOORWAYS = [(3, 6), (6, 2), (7, 9), (10, 6)]

    def __init__(
        self,
        size: int = 13,
        fail_prob: float = 0.15,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = True,
        goal_pos: Tuple[int, int] | None = None,
        start_pos: Tuple[int, int] | None = None,
        mine_locations: list | None = None,
        mine_reward: float = 0.0,
    ):
        if mine_locations is None:
            self.mine_locations = list(self.DEFAULT_MINES)
        else:
            self.mine_locations = [tuple(m) for m in mine_locations]

        self.mine_reward = float(mine_reward)

        # Validate mine locations
        for m in self.mine_locations:
            if m in self.DOORWAYS:
                raise ValueError(f"Mine location {m} is in a doorway cell {self.DOORWAYS}. Doorways must be clear.")

        super().__init__(
            size=size,
            fail_prob=fail_prob,
            gamma=gamma,
            episodic=episodic,
            use_visual_obs=use_visual_obs,
            goal_pos=goal_pos,
            start_pos=start_pos,
        )

        # Ensure mines don't overlap with start or goal
        start_tuple = tuple(int(x) for x in self.start)
        goal_tuple = tuple(int(x) for x in self.goal)
        for m in self.mine_locations:
            if m == start_tuple:
                raise ValueError(f"Mine location {m} cannot overlap with start position {start_tuple}.")
            if m == goal_tuple:
                raise ValueError(f"Mine location {m} cannot overlap with goal position {goal_tuple}.")

        self.mine_indices = [self._coord_to_idx(jnp.array(m)) for m in self.mine_locations]

    def _build_obs_stack(self) -> jax.Array:
        obs = super()._build_obs_stack()
        if self.use_visual_obs:
            mine_map = np.zeros((self.N, self.N), dtype=np.float32)
            for my, mx in self.mine_locations:
                mine_map[my, mx] = 0.5
            obs_np = np.array(obs)
            # Channel 0 has walls (1.0). Overlay mines at 0.5 intensity.
            obs_np[..., 0] = np.maximum(obs_np[..., 0], mine_map[None, ...])
            obs = jnp.asarray(obs_np, dtype=jnp.float32)
        return obs

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)

        p_correct = 1.0 - self.fail_prob
        p_wrong = self.fail_prob / 3.0
        mine_locs_set = set(self.mine_locations)

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx]

            # 1. Goal state
            if s_idx == self.goal_idx:
                if not continuing:
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :, self.terminal_idx] = 1.0
                else:
                    P[s_idx, :, self.start_idx] = 1.0
                    R[s_idx, :, self.start_idx] = 1.0
                continue

            # 2. Mine cells themselves (if ever initialized or queried directly)
            if tuple(int(x) for x in pos) in mine_locs_set:
                if not continuing:
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :, self.terminal_idx] = self.mine_reward
                else:
                    P[s_idx, :, self.start_idx] = 1.0
                    R[s_idx, :, self.start_idx] = self.mine_reward
                continue

            # 3. Standard states
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    prob = p_correct if executed_a == chosen_a else p_wrong
                    if prob == 0:
                        continue

                    next_pos = self._step_pos(pos, executed_a)
                    next_pos_tuple = (int(next_pos[0]), int(next_pos[1]))

                    # Check goal
                    if next_pos_tuple == tuple(int(x) for x in self.goal):
                        P[s_idx, chosen_a, self.goal_idx] += prob
                        R[s_idx, chosen_a, self.goal_idx] = 0.0
                    # Check mine: Stepping on a mine terminates immediately!
                    # Target is purely mine_reward + gamma * 0 = mine_reward (no bootstrapping).
                    elif next_pos_tuple in mine_locs_set:
                        if not continuing:
                            P[s_idx, chosen_a, self.terminal_idx] += prob
                            R[s_idx, chosen_a, self.terminal_idx] = self.mine_reward
                        else:
                            P[s_idx, chosen_a, self.start_idx] += prob
                            R[s_idx, chosen_a, self.start_idx] = self.mine_reward
                    else:
                        next_idx = self._coord_to_idx(next_pos)
                        P[s_idx, chosen_a, next_idx] += prob

        # Terminal state
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :, self.terminal_idx] = 0.0

        return jnp.asarray(P, dtype=jnp.float32), jnp.asarray(R, dtype=jnp.float32)


class FourRoomsMinesDenseExactValue(FourRoomsMinesExactValue):
    """
    Four Rooms with mines and potential-based reward shaping (PBRS).
    """

    def __init__(self, *args, potential_scale: float = 0.03125, **kwargs):
        self.potential_scale = float(potential_scale)
        super().__init__(*args, **kwargs)

    def compute_shortest_path_distances(self) -> np.ndarray:
        from collections import deque

        coords_arr = np.array(self.coords)
        coords_to_idx = {tuple(c): i for i, c in enumerate(coords_arr)}
        mine_locs_set = set(self.mine_locations)
        adj = [[] for _ in range(self.num_states)]
        for i, (y, x) in enumerate(coords_arr):
            if (int(y), int(x)) in mine_locs_set:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = int(y + dy), int(x + dx)
                if (ny, nx) in coords_to_idx and (ny, nx) not in mine_locs_set:
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

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P, R_sparse = super()._build_env_dynamics(continuing=continuing)
        self.distances = self.compute_shortest_path_distances()
        phi = - self.potential_scale * self.distances
        phi[self.terminal_idx] = 0.0
        self.potential = jnp.asarray(phi, dtype=jnp.float32)

        gamma = self.gamma
        phi_start = float(phi[self.start_idx])
        F = gamma * phi[None, None, :] - phi[:, None, None]
        F[self.terminal_idx, :, self.terminal_idx] = 0.0

        R_dense = np.array(R_sparse) + F

        # In episodic mode, transitions into terminal_idx from non-goal states (i.e. mines)
        # represent dying and resetting to the start state.
        # They should receive the potential jump gamma * phi(start) - phi(s),
        # not gamma * phi(goal) - phi(s).
        if not continuing:
            P_arr = np.array(P)
            R_sparse_arr = np.array(R_sparse)
            for s in range(self.num_states):
                if s != self.goal_idx:
                    for a in range(self.num_actions):
                        if P_arr[s, a, self.terminal_idx] > 0:
                            R_dense[s, a, self.terminal_idx] = (
                                R_sparse_arr[s, a, self.terminal_idx] + gamma * phi_start - phi[s]
                            )

        R_dense = np.where(np.array(P) > 0, R_dense, 0.0)
        return P, jnp.asarray(R_dense, dtype=jnp.float32)


