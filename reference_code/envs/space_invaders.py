from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Any, Dict, List


class SpaceInvadersExactValue:
    """
    Exact policy evaluation for Tabular Space Invaders Environment.
    
    Supports:
        - 3-Alien shielded fleet (default):
            - 2 back aliens in a row at row fy (columns col0 and col0 + 1)
            - 1 front alien at row fy + 1 directly in front of col0 (shielding the left back alien)
            - Vertical laser mechanics: shooting at col0 hits the front alien first;
              once the front alien is destroyed, subsequent shots at col0 hit the back alien.
            - Default grid: Width W = 7 columns, Height H = 6 rows.
            - Active states: 1974 states.
        - 2-Alien fleet (legacy):
            - 2 aliens side-by-side at row fy.
            - Active states: 420 states (W=5, H=4) or 1050 states (W=7, H=6).

    Observations:
        4-channel frame-stacked visual representation of shape (H, W, 4):
            Channel 0: Player position at current frame t.
            Channel 1: Living aliens at current frame t.
            Channel 2: Player position at frame t - 1.
            Channel 3: Living aliens at frame t - 1 (shadow channel capturing fleet velocity/direction).
        This guarantees complete full observability with 100% unique observations.
    """

    def __init__(
        self,
        width: int = 7,
        height: int = 6,
        num_aliens: int = 3,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = True,
        endless: bool = True,
        kill_reward: float = 0.5,
        death_penalty: float = 0.0,
    ):
        self.W = int(width)
        self.H = int(height)
        self.num_aliens = int(num_aliens)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs
        self.endless = endless
        self.kill_reward = float(kill_reward)
        self.death_penalty = float(death_penalty)
        self.fail_prob = 0.0

        self.num_actions = 4
        self.action_names = ["No-op", "Left", "Right", "Fire"]
        self.directions = jnp.array(
            [[0, 0], [0, -1], [0, 1], [-1, 0]], dtype=jnp.float32
        )

        # 1. Enumerate all valid active state configurations
        self.states = self._build_states()
        self.num_states = len(self.states)
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1

        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.coords = jnp.array(self.states, dtype=jnp.int32)

        if self.num_aliens == 3:
            start_state = (1, 1, 1, self.W - 1, 0, 0, 1)
        else:
            start_state = (1, 1, self.W - 1, 0, 0, 1, 1)
        self.start_idx = self.state_to_idx[start_state]
        self.goal_idx = self.start_idx
        self.reset_indices = jnp.array([self.start_idx], dtype=jnp.int32)
        self.occupied_map = jnp.zeros((self.H, self.W), dtype=jnp.float32)

        # 2. Build Observations
        self.obs_stack = self._build_obs_stack()

        # 3. Build Dynamics P and R (Episodic and Continuing)
        P_ep, R_ep, P_win, P_death = self._build_env_dynamics(continuing=False)
        P_cnt, R_cnt, _, _ = self._build_env_dynamics(continuing=True)

        self.P = jnp.asarray(P_ep, dtype=jnp.float32)
        self.R = jnp.asarray(R_ep, dtype=jnp.float32)
        self.P_cont = jnp.asarray(P_cnt, dtype=jnp.float32)
        self.P_win = jnp.asarray(P_win, dtype=jnp.float32)
        self.P_death = jnp.asarray(P_death, dtype=jnp.float32)

    def _build_states(self):
        states = []
        if self.num_aliens == 3:
            # 1. (1, 1, 1): All 3 alive (Front at fy+1, Back-0 at fy, Back-1 at fy, col0+1)
            for xp in range(self.W):
                for fy in range(self.H - 2):
                    for col0 in range(self.W - 1):
                        for fdir in [-1, 1]:
                            states.append((1, 1, 1, xp, fy, col0, fdir))

            # 2. (1, 0, 1): Front + Back-1
            for xp in range(self.W):
                for fy in range(self.H - 2):
                    for col0 in range(self.W - 1):
                        for fdir in [-1, 1]:
                            states.append((1, 0, 1, xp, fy, col0, fdir))

            # 3. (0, 1, 1): Both Back
            for xp in range(self.W):
                for fy in range(self.H - 1):
                    for col0 in range(self.W - 1):
                        for fdir in [-1, 1]:
                            states.append((0, 1, 1, xp, fy, col0, fdir))

            # 4. (1, 1, 0): Front + Back-0 (column stack at col)
            for xp in range(self.W):
                for fy in range(self.H - 2):
                    for col in range(self.W):
                        for fdir in [-1, 1]:
                            states.append((1, 1, 0, xp, fy, col, fdir))

            # 5. (1, 0, 0): Canonical Single Alien at col
            for xp in range(self.W):
                for fy in range(self.H - 1):
                    for col in range(self.W):
                        for fdir in [-1, 1]:
                            states.append((1, 0, 0, xp, fy, col, fdir))
        else:
            for xp in range(self.W):
                for fy in range(self.H - 1):
                    for col0 in range(self.W - 1):
                        for fdir in [-1, 1]:
                            states.append((1, 1, xp, fy, col0, col0 + 1, fdir))
            for xp in range(self.W):
                for fy in range(self.H - 1):
                    for col0 in range(self.W):
                        for fdir in [-1, 1]:
                            states.append((1, 0, xp, fy, col0, -1, fdir))
        return states

    def _get_alien_pos_3(self, af: int, ab0: int, ab1: int, fy: int, c0: int) -> List[Tuple[int, int]]:
        aliens = []
        if af == 1 and ab0 == 1 and ab1 == 1:
            aliens = [(fy + 1, c0), (fy, c0), (fy, c0 + 1)]
        elif af == 1 and ab0 == 0 and ab1 == 1:
            aliens = [(fy + 1, c0), (fy, c0 + 1)]
        elif af == 0 and ab0 == 1 and ab1 == 1:
            aliens = [(fy, c0), (fy, c0 + 1)]
        elif af == 1 and ab0 == 1 and ab1 == 0:
            aliens = [(fy + 1, c0), (fy, c0)]
        elif af == 1 and ab0 == 0 and ab1 == 0:
            aliens = [(fy, c0)]
        return aliens

    def _get_prev_alien_pos_3(
        self, af: int, ab0: int, ab1: int, fy: int, c0: int, fdir: int
    ) -> List[Tuple[int, int]]:
        w_span = 2 if (ab1 == 1) else 1
        if fdir == 1:
            if c0 > 0:
                prev_fy, prev_c0 = fy, c0 - 1
            else:
                prev_fy, prev_c0 = max(0, fy - 1), c0
        else:
            if c0 + w_span - 1 < self.W - 1:
                prev_fy, prev_c0 = fy, c0 + 1
            else:
                prev_fy, prev_c0 = max(0, fy - 1), c0

        aliens = []
        if af == 1 and ab0 == 1 and ab1 == 1:
            aliens = [(prev_fy + 1, prev_c0), (prev_fy, prev_c0), (prev_fy, prev_c0 + 1)]
        elif af == 1 and ab0 == 0 and ab1 == 1:
            aliens = [(prev_fy + 1, prev_c0), (prev_fy, prev_c0 + 1)]
        elif af == 0 and ab0 == 1 and ab1 == 1:
            aliens = [(prev_fy, prev_c0), (prev_fy, prev_c0 + 1)]
        elif af == 1 and ab0 == 1 and ab1 == 0:
            aliens = [(prev_fy + 1, prev_c0), (prev_fy, prev_c0)]
        elif af == 1 and ab0 == 0 and ab1 == 0:
            aliens = [(prev_fy, prev_c0)]
        return aliens

    def _get_prev_alien_pos(
        self, a0: int, a1: int, fy: int, col0: int, col1: int, fdir: int
    ) -> List[Tuple[int, int]]:
        prevs = []
        if a0 == 1 and a1 == 1:
            if fdir == 1:
                if col0 > 0:
                    prevs = [(fy, col0 - 1), (fy, col1 - 1)]
                else:
                    prevs = [(max(0, fy - 1), col0), (max(0, fy - 1), col1)]
            else:
                if col1 < self.W - 1:
                    prevs = [(fy, col0 + 1), (fy, col1 + 1)]
                else:
                    prevs = [(max(0, fy - 1), col0), (max(0, fy - 1), col1)]
        elif a0 == 1 and a1 == 0:
            c = col0
            if fdir == 1:
                prevs = [(fy, c - 1)] if c > 0 else [(max(0, fy - 1), c)]
            else:
                prevs = [(fy, c + 1)] if c < self.W - 1 else [(max(0, fy - 1), c)]
        return prevs

    def _build_obs_stack(self) -> jax.Array:
        """
        Builds spatial grid observations of shape (num_states, H, W, 4).
        Frame t (current):
          Channel 0: Player position map (1.0 at [H - 1, x_p]).
          Channel 1: Living alien position map (1.0 at living alien coordinates).
        Frame t - 1 (shadow / frame-stacked past position):
          Channel 2: Player shadow map (1.0 at [H - 1, x_p]).
          Channel 3: Living alien shadow map (1.0 at [prev_fy, prev_col]).
        """
        if self.use_visual_obs:
            obs = np.zeros((self.num_states, self.H, self.W, 4), dtype=np.float32)
            for i, s in enumerate(self.states):
                if self.num_aliens == 3:
                    af, ab0, ab1, xp, fy, c0, fdir = s
                    obs[i, self.H - 1, xp, 0] = 1.0
                    for y, x in self._get_alien_pos_3(af, ab0, ab1, fy, c0):
                        obs[i, y, x, 1] = 1.0
                    obs[i, self.H - 1, xp, 2] = 1.0
                    for y, x in self._get_prev_alien_pos_3(af, ab0, ab1, fy, c0, fdir):
                        obs[i, y, x, 3] = 1.0
                else:
                    a0, a1, xp, fy, col0, col1, fdir = s
                    obs[i, self.H - 1, xp, 0] = 1.0
                    if a0 == 1:
                        obs[i, fy, col0, 1] = 1.0
                    if a1 == 1:
                        obs[i, fy, col1, 1] = 1.0
                    obs[i, self.H - 1, xp, 2] = 1.0
                    for pfy, pcol in self._get_prev_alien_pos(a0, a1, fy, col0, col1, fdir):
                        obs[i, pfy, pcol, 3] = 1.0
            return jnp.asarray(obs, dtype=jnp.float32)
        else:
            obs = np.zeros((self.num_states, 7), dtype=np.float32)
            for i, s in enumerate(self.states):
                if self.num_aliens == 3:
                    af, ab0, ab1, xp, fy, c0, fdir = s
                    obs[i] = [
                        xp / float(self.W),
                        fy / float(self.H),
                        c0 / float(self.W),
                        1.0 if fdir == 1 else 0.0,
                        float(af),
                        float(ab0),
                        float(ab1),
                    ]
                else:
                    a0, a1, xp, fy, col0, col1, fdir = s
                    obs[i] = [
                        xp / float(self.W),
                        fy / float(self.H),
                        max(0, col0) / float(self.W),
                        max(0, col1) / float(self.W),
                        1.0 if fdir == 1 else 0.0,
                        float(a0),
                        float(a1),
                    ]
            return jnp.asarray(obs, dtype=jnp.float32)

    def _build_env_dynamics(
        self, continuing: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P = np.zeros(
            (self.num_total_states, self.num_actions, self.num_total_states),
            dtype=np.float32,
        )
        R = np.zeros(
            (self.num_total_states, self.num_actions, self.num_total_states),
            dtype=np.float32,
        )
        P_win = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        P_death = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        if self.num_aliens == 3:
            for s_idx in range(self.num_states):
                af, ab0, ab1, xp, fy, c0, fdir = self.states[s_idx]

                for a in range(self.num_actions):
                    if a == 0: xp_next = xp
                    elif a == 1: xp_next = max(0, xp - 1)
                    elif a == 2: xp_next = min(self.W - 1, xp + 1)
                    elif a == 3: xp_next = xp

                    reward = 0.0
                    af_next, ab0_next, ab1_next = af, ab0, ab1
                    c0_curr = c0

                    if a == 3:
                        if af == 1 and ab0 == 1 and ab1 == 1:
                            if xp == c0:
                                af_next = 0
                                reward += self.kill_reward
                            elif xp == c0 + 1:
                                ab1_next = 0
                                reward += self.kill_reward
                        elif af == 1 and ab0 == 0 and ab1 == 1:
                            if xp == c0:
                                af_next = 0
                                reward += self.kill_reward
                                c0_curr = c0 + 1
                            elif xp == c0 + 1:
                                ab1_next = 0
                                reward += self.kill_reward
                        elif af == 0 and ab0 == 1 and ab1 == 1:
                            if xp == c0:
                                ab0_next = 0
                                reward += self.kill_reward
                                c0_curr = c0 + 1
                            elif xp == c0 + 1:
                                ab1_next = 0
                                reward += self.kill_reward
                        elif af == 1 and ab0 == 1 and ab1 == 0:
                            if xp == c0:
                                af_next = 0
                                reward += self.kill_reward
                        elif af == 1 and ab0 == 0 and ab1 == 0:
                            if xp == c0:
                                af_next = 0
                                reward += self.kill_reward

                    if (af_next + ab0_next + ab1_next) == 0:
                        P_win[s_idx, a] = 1.0
                        if not self.endless:
                            if not continuing:
                                P[s_idx, a, self.terminal_idx] = 1.0
                                R[s_idx, a, self.terminal_idx] = reward
                            else:
                                P[s_idx, a, self.start_idx] = 1.0
                                R[s_idx, a, self.start_idx] = reward
                        else:
                            if xp_next <= self.W // 2:
                                s_respawn = (1, 1, 1, xp_next, 0, self.W - 2, -1)
                            else:
                                s_respawn = (1, 1, 1, xp_next, 0, 0, 1)
                            respawn_idx = self.state_to_idx[s_respawn]
                            P[s_idx, a, respawn_idx] = 1.0
                            R[s_idx, a, respawn_idx] = reward
                        continue

                    if (af_next + ab0_next + ab1_next) == 1:
                        af_next, ab0_next, ab1_next = 1, 0, 0

                    w_span = 2 if (af_next + ab0_next + ab1_next > 1 and ab1_next == 1) else 1
                    if fdir == 1:
                        if c0_curr + w_span - 1 < self.W - 1:
                            c0_next, fy_next, fdir_next = c0_curr + 1, fy, 1
                        else:
                            c0_next, fy_next, fdir_next = c0_curr, fy + 1, -1
                    else:
                        if c0_curr > 0:
                            c0_next, fy_next, fdir_next = c0_curr - 1, fy, -1
                        else:
                            c0_next, fy_next, fdir_next = c0_curr, fy + 1, 1

                    lowest_row = (fy_next + 1) if af_next == 1 else fy_next
                    is_invaded = lowest_row >= self.H - 1
                    is_bombed = False
                    if lowest_row == self.H - 2:
                        if af_next == 1 and c0_next == xp_next: is_bombed = True
                        if ab0_next == 1 and c0_next == xp_next: is_bombed = True
                        if ab1_next == 1 and (c0_next + 1) == xp_next: is_bombed = True

                    if is_invaded or is_bombed:
                        P_death[s_idx, a] = 1.0
                        death_rew = reward - self.death_penalty
                        if not continuing:
                            P[s_idx, a, self.terminal_idx] = 1.0
                            R[s_idx, a, self.terminal_idx] = death_rew
                        else:
                            P[s_idx, a, self.start_idx] = 1.0
                            R[s_idx, a, self.start_idx] = death_rew
                    else:
                        next_state = (af_next, ab0_next, ab1_next, xp_next, fy_next, c0_next, fdir_next)
                        next_idx = self.state_to_idx[next_state]
                        P[s_idx, a, next_idx] = 1.0
                        R[s_idx, a, next_idx] = reward

        else:
            for s_idx in range(self.num_states):
                a0, a1, xp, fy, col0, col1, fdir = self.states[s_idx]

                for a in range(self.num_actions):
                    if a == 0: xp_next = xp
                    elif a == 1: xp_next = max(0, xp - 1)
                    elif a == 2: xp_next = min(self.W - 1, xp + 1)
                    elif a == 3: xp_next = xp

                    reward = 0.0
                    if a0 == 1 and a1 == 1:
                        if a == 3:
                            hit_left = (xp == col0)
                            hit_right = (xp == col1)
                            if hit_left:
                                a0_next, a1_next, col0_curr, col1_curr = 1, 0, col1, -1
                                reward += self.kill_reward
                            elif hit_right:
                                a0_next, a1_next, col0_curr, col1_curr = 1, 0, col0, -1
                                reward += self.kill_reward
                            else:
                                a0_next, a1_next, col0_curr, col1_curr = 1, 1, col0, col1
                        else:
                            a0_next, a1_next, col0_curr, col1_curr = 1, 1, col0, col1
                    else:
                        if a == 3 and xp == col0:
                            a0_next, a1_next, col0_curr, col1_curr = 0, 0, -1, -1
                            reward += self.kill_reward
                        else:
                            a0_next, a1_next, col0_curr, col1_curr = 1, 0, col0, -1

                    if a0_next == 0 and a1_next == 0:
                        P_win[s_idx, a] = 1.0
                        if not self.endless:
                            if not continuing:
                                P[s_idx, a, self.terminal_idx] = 1.0
                                R[s_idx, a, self.terminal_idx] = reward
                            else:
                                P[s_idx, a, self.start_idx] = 1.0
                                R[s_idx, a, self.start_idx] = reward
                        else:
                            if xp_next <= self.W // 2:
                                s_respawn = (1, 1, xp_next, 0, self.W - 2, self.W - 1, -1)
                            else:
                                s_respawn = (1, 1, xp_next, 0, 0, 1, 1)
                            respawn_idx = self.state_to_idx[s_respawn]
                            P[s_idx, a, respawn_idx] = 1.0
                            R[s_idx, a, respawn_idx] = reward
                        continue

                    if a0_next == 1 and a1_next == 1:
                        if fdir == 1:
                            if col1_curr < self.W - 1:
                                col0_next, col1_next, fy_next, fdir_next = col0_curr + 1, col1_curr + 1, fy, 1
                            else:
                                col0_next, col1_next, fy_next, fdir_next = col0_curr, col1_curr, fy + 1, -1
                        else:
                            if col0_curr > 0:
                                col0_next, col1_next, fy_next, fdir_next = col0_curr - 1, col1_curr - 1, fy, -1
                            else:
                                col0_next, col1_next, fy_next, fdir_next = col0_curr, col1_curr, fy + 1, 1
                    else:
                        col1_next = -1
                        if fdir == 1:
                            if col0_curr < self.W - 1:
                                col0_next, fy_next, fdir_next = col0_curr + 1, fy, 1
                            else:
                                col0_next, fy_next, fdir_next = col0_curr, fy + 1, -1
                        else:
                            if col0_curr > 0:
                                col0_next, fy_next, fdir_next = col0_curr - 1, fy, -1
                            else:
                                col0_next, fy_next, fdir_next = col0_curr, fy + 1, 1

                    is_invaded = fy_next >= self.H - 1
                    is_bombed = (fy_next == self.H - 2) and (
                        (a0_next == 1 and col0_next == xp_next)
                        or (a1_next == 1 and col1_next == xp_next)
                    )

                    if is_invaded or is_bombed:
                        P_death[s_idx, a] = 1.0
                        death_rew = reward - self.death_penalty
                        if not continuing:
                            P[s_idx, a, self.terminal_idx] = 1.0
                            R[s_idx, a, self.terminal_idx] = death_rew
                        else:
                            P[s_idx, a, self.start_idx] = 1.0
                            R[s_idx, a, self.start_idx] = death_rew
                    else:
                        next_state = (a0_next, a1_next, xp_next, fy_next, col0_next, col1_next, fdir_next)
                        next_idx = self.state_to_idx[next_state]
                        P[s_idx, a, next_idx] = 1.0
                        R[s_idx, a, next_idx] = reward

        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :, self.terminal_idx] = 0.0
        return P, R, P_win, P_death

    def solve_linear_system(
        self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array
    ) -> jax.Array:
        """Solves the exact linear system (I - gamma * P_pi) V = R_pi."""
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        R_pi = jnp.einsum("sa,sam,sam->s", pi, P_env, R_env)
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def compute_true_values_raw(self, pi: jax.Array) -> jax.Array:
        """Returns vector of exact state values V(s) of shape (num_total_states,)."""
        return self.solve_linear_system(pi, self.P, self.R)

    def compute_true_values(self, pi: jax.Array) -> jax.Array:
        return self.compute_true_values_raw(pi)

    def compute_stationary_distribution_raw(
        self, pi: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the stationary distribution mu under the resetting chain P_cont.
        Returns (mu, P_pi).
        """
        P_env = self.P_cont[: self.num_states, :, : self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)

        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)
        b = jnp.zeros(self.num_states).at[-1].set(1.0)

        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum(), P_pi

    def compute_discounted_visitation_raw(self, pi: jax.Array) -> jax.Array:
        """
        Computes the normalized discounted occupancy distribution:
            d_gamma = (1 - gamma) * mu_0 * (I - gamma * P_pi_trans)^{-1}
        """
        P_env = self.P[: self.num_states, :, : self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        mu_0 = jnp.zeros(self.num_states).at[self.start_idx].set(1.0)

        A = jnp.eye(self.num_states) - self.gamma * P_pi.T
        d_gamma = jnp.linalg.solve(A, (1.0 - self.gamma) * mu_0)
        return d_gamma / jnp.sum(d_gamma)

    def compute_absorption_probabilities(
        self, pi: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Computes exact closed-form probability of WIN vs DEATH from each state:
            prob_win(s) = [(I - P_pi_trans)^{-1} P_win_pi]_s
            prob_death(s) = [(I - P_pi_trans)^{-1} P_death_pi]_s
        """
        P_trans = self.P[: self.num_states, :, : self.num_states]
        P_pi_trans = jnp.einsum("sa,sam->sm", pi[: self.num_states], P_trans)
        I = jnp.eye(self.num_states)
        fundamental = jnp.linalg.inv(I - P_pi_trans)

        P_win_pi = jnp.einsum("sa,sa->s", pi[: self.num_states], self.P_win)
        P_death_pi = jnp.einsum("sa,sa->s", pi[: self.num_states], self.P_death)

        prob_win = fundamental @ P_win_pi
        prob_death = fundamental @ P_death_pi
        return prob_win, prob_death

    def get_optimal_value_function(
        self, tol: float = 1e-6, max_iters: int = 1000
    ) -> jax.Array:
        """Computes V* via exact Bellman optimality value iteration."""
        V = jnp.zeros(self.num_total_states)
        R_expected = jnp.einsum("sam,sam->sa", self.P, self.R)

        def body_fun(val):
            i, V_curr, delta = val
            expected_v = jnp.einsum("sam,m->sa", self.P, V_curr)
            Q = R_expected + self.gamma * expected_v
            V_new = jnp.max(Q, axis=-1)
            V_new = V_new.at[self.terminal_idx].set(0.0)
            delta = jnp.max(jnp.abs(V_new - V_curr))
            return (i + 1, V_new, delta)

        def cond_fun(val):
            i, V_curr, delta = val
            return jnp.logical_and(i < max_iters, delta > tol)

        _, V_star, _ = jax.lax.while_loop(cond_fun, body_fun, (0, V, 1.0))
        return V_star

    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Projects per-state values onto a 2D (H, W) grid representing average value
        conditioned on the player's horizontal position.
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        # Aggregate values by player position (xp) and lowest alien row (fy)
        grid = jnp.zeros((self.H, self.W), dtype=values.dtype)
        counts = jnp.zeros((self.H, self.W), dtype=values.dtype)

        xp_coords = self.coords[:, 2]
        fy_coords = self.coords[:, 3]

        for i in range(self.num_states):
            grid = grid.at[fy_coords[i], xp_coords[i]].add(values[i])
            counts = counts.at[fy_coords[i], xp_coords[i]].add(1.0)

        return jnp.where(counts > 0, grid / counts, 0.0)
