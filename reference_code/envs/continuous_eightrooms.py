from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any
from flax import struct
from gymnax.environments import environment, spaces
from envs.eightrooms import EightRoomsExactValue, EightRoomsDense


# Robust Environment Base for cross-version compatibility
try:
    _EnvBase = environment.Environment
except (TypeError, AttributeError):
    _EnvBase = object


@struct.dataclass
class ContinuousEightRoomsState(environment.EnvState):
    pos: jax.Array  # Continuous 2D position [y, x] in [0, 25] x [0, 13]
    vel: jax.Array  # Continuous 2D velocity [vy, vx]
    goal: jax.Array  # Continuous 2D goal coordinates [goal_y, goal_x]
    time: int


@struct.dataclass
class ContinuousEightRoomsParams(environment.EnvParams):
    max_steps_in_episode: int = 1000
    step_size: float = 0.5  # Maximum displacement per step (or speed multiplier)
    agent_radius: float = 0.2  # Collision radius of the agent (corridors are 1.0 wide)
    goal_radius: float = 0.6  # Euclidean radius around goal center to register completion
    action_noise: float = 0.0  # Gaussian noise standard deviation added to action
    fail_prob: float = 0.0  # Probability of taking a completely random uniform action
    substeps: int = 4  # Sub-steps for collision resolution to prevent tunneling
    control_mode: int = 0  # 0: Kinematic (velocity / displacement control), 1: Dynamic (acceleration with inertia)
    damping: float = 0.2  # Velocity damping factor per step in dynamic control mode
    dt: float = 0.1  # Time step integration parameter for dynamic mode
    max_vel: float = 1.0  # Maximum speed clamp for dynamic mode


class ContinuousEightRooms(_EnvBase):
    """
    Gymnax-compatible Continuous Control Environment for EightRooms (4 rooms high x 2 rooms wide).

    Features:
    - Continuous 2D state space: pos = (y, x) in [0, 25] x [0, 13].
    - Continuous 2D action space: action = (a_y, a_x) in [-1.0, 1.0]^2.
    - Exact same 8 rooms maze geometry and doorway connectivity as discrete EightRooms.
    - Same sparse reward structure: +1.0 only upon reaching the goal region, 0.0 elsewhere.
    - Vectorized, branch-free JAX collision detection and wall sliding physics (sub-stepped to guarantee no tunneling).
    - Supports both vector observations (MLP, shape (4,) or (6,)) and visual 2D raster observations (CNN, shape (25, 13, 2)).
    - Supports both kinematic (velocity) and dynamic (acceleration + inertia) control modes.
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        include_vel_in_obs: bool = False,
        goal_fixed: Tuple[float, float] = (23.5, 11.5),
        pos_fixed: Tuple[float, float] = (3.5, 1.5),
        height: int = 25,
        width: int = 13,
        substeps: int = 4,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.substeps = int(substeps)
        self.use_visual_obs = use_visual_obs
        self.include_vel_in_obs = include_vel_in_obs

        # Center of default discrete start cell (3, 1) and goal cell (23, 11)
        self.pos_fixed = jnp.array(pos_fixed, dtype=jnp.float32)
        self.goal_fixed = jnp.array(goal_fixed, dtype=jnp.float32)

        # 1. Load exact 8-room binary occupancy map (1.0 = wall, 0.0 = free space)
        self.env_map = EightRoomsExactValue._generate_eight_rooms_map()
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)

        # 2. Free cell coordinates for probing / analysis
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        self.num_cells = int(self.coords.shape[0])

    @property
    def default_params(self) -> ContinuousEightRoomsParams:
        return ContinuousEightRoomsParams()

    def _is_colliding(self, pos: jax.Array, radius: float) -> jax.Array:
        """
        Check if an agent disc of the given radius at continuous pos (y, x)
        overlaps any wall cell or boundary.
        """
        y, x = pos[0], pos[1]
        ymin = jnp.floor(y - radius).astype(jnp.int32)
        ymax = jnp.floor(y + radius).astype(jnp.int32)
        xmin = jnp.floor(x - radius).astype(jnp.int32)
        xmax = jnp.floor(x + radius).astype(jnp.int32)

        def is_wall(cy: jax.Array, cx: jax.Array) -> jax.Array:
            in_bounds = (cy >= 0) & (cy < self.height) & (cx >= 0) & (cx < self.width)
            clamped_y = jnp.clip(cy, 0, self.height - 1)
            clamped_x = jnp.clip(cx, 0, self.width - 1)
            return (~in_bounds) | (self.occupied_map[clamped_y, clamped_x] > 0.5)

        c00 = is_wall(ymin, xmin)
        c01 = is_wall(ymin, xmax)
        c10 = is_wall(ymax, xmin)
        c11 = is_wall(ymax, xmax)
        return c00 | c01 | c10 | c11

    def _resolve_sliding_step(
        self,
        pos: jax.Array,
        delta: jax.Array,
        radius: float,
    ) -> jax.Array:
        """
        Sub-stepped continuous motion with axis-aligned wall sliding.
        If moving diagonally into a wall, the normal component is stopped and
        the parallel component slides smoothly.
        """
        sub_delta = delta / float(self.substeps)

        def sub_step_fn(i: int, p: jax.Array) -> jax.Array:
            # 1. Try Y motion
            cand_y = p.at[0].add(sub_delta[0])
            can_y = ~self._is_colliding(cand_y, radius)
            p_after_y = jax.lax.select(can_y, cand_y, p)

            # 2. Try X motion
            cand_x = p_after_y.at[1].add(sub_delta[1])
            can_x = ~self._is_colliding(cand_x, radius)
            p_after_x = jax.lax.select(can_x, cand_x, p_after_y)

            return p_after_x

        return jax.lax.fori_loop(0, self.substeps, sub_step_fn, pos)

    def step_env(
        self,
        key: jax.Array,
        state: ContinuousEightRoomsState,
        action: jax.Array,
        params: ContinuousEightRoomsParams,
    ) -> Tuple[jax.Array, ContinuousEightRoomsState, jax.Array, jax.Array, dict]:
        # 1. Clip action to unit box [-1.0, 1.0]^2
        action = jnp.clip(action, -1.0, 1.0)

        # 2. Action noise / stochasticity
        key_noise, key_fail = jax.random.split(key)
        noise = jax.random.normal(key_noise, shape=(2,)) * params.action_noise
        action = jnp.clip(action + noise, -1.0, 1.0)

        p_roll = jax.random.uniform(key_fail)
        random_action = jax.random.uniform(key_fail, shape=(2,), minval=-1.0, maxval=1.0)
        executed_action = jnp.where(p_roll < params.fail_prob, random_action, action)

        # 3. Compute proposed displacement based on control mode
        # Mode 0: Kinematic velocity control (action = velocity command)
        # Mode 1: Dynamic acceleration control (action = acceleration / force command)
        is_dynamic = params.control_mode == 1

        v_dynamic = (1.0 - params.damping) * state.vel + executed_action * params.step_size * params.dt
        v_dynamic = jnp.clip(v_dynamic, -params.max_vel, params.max_vel)
        disp_dynamic = v_dynamic * params.dt

        disp_kinematic = executed_action * params.step_size

        disp = jnp.where(is_dynamic, disp_dynamic, disp_kinematic)

        # 4. Integrate displacement with sub-stepped wall collision & sliding
        new_pos = self._resolve_sliding_step(
            state.pos, disp, params.agent_radius
        )

        # 5. Effective velocity achieved
        effective_vel = new_pos - state.pos
        new_vel = jnp.where(is_dynamic, v_dynamic, effective_vel)

        # 6. Sparse Reward: +1.0 when reaching Euclidean goal radius, 0.0 elsewhere
        dist_to_goal = jnp.linalg.norm(new_pos - state.goal)
        is_goal = dist_to_goal <= params.goal_radius
        reward = is_goal.astype(jnp.float32)

        # 7. Check episode termination
        next_time = state.time + 1
        done_steps = next_time >= params.max_steps_in_episode
        done_goal = is_goal
        done = jnp.logical_or(done_goal, done_steps)

        new_state = ContinuousEightRoomsState(
            pos=new_pos,
            vel=new_vel,
            goal=state.goal,
            time=next_time,
        )

        info = {
            "discount": self.discount(new_state, params),
            "is_goal": is_goal,
            "dist_to_goal": dist_to_goal,
            "is_timeout": done_steps & (~done_goal),
        }

        return (
            jax.lax.stop_gradient(self.get_obs(new_state)),
            jax.lax.stop_gradient(new_state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: ContinuousEightRoomsParams
    ) -> Tuple[jax.Array, ContinuousEightRoomsState]:
        state = ContinuousEightRoomsState(
            pos=self.pos_fixed,
            vel=jnp.zeros(2, dtype=jnp.float32),
            goal=self.goal_fixed,
            time=0,
        )
        return self.get_obs(state), state

    def _render_agent_map(self, pos: jax.Array) -> jax.Array:
        """Smooth bilinear splatting of continuous agent position onto grid map."""
        cy = pos[0] - 0.5
        cx = pos[1] - 0.5
        y0 = jnp.clip(jnp.floor(cy).astype(jnp.int32), 0, self.height - 1)
        x0 = jnp.clip(jnp.floor(cx).astype(jnp.int32), 0, self.width - 1)
        y1 = jnp.clip(y0 + 1, 0, self.height - 1)
        x1 = jnp.clip(x0 + 1, 0, self.width - 1)

        fy = jnp.clip(cy - y0, 0.0, 1.0)
        fx = jnp.clip(cx - x0, 0.0, 1.0)

        agent_map = jnp.zeros((self.height, self.width), dtype=jnp.float32)
        agent_map = agent_map.at[y0, x0].add((1.0 - fy) * (1.0 - fx))
        agent_map = agent_map.at[y0, x1].add((1.0 - fy) * fx)
        agent_map = agent_map.at[y1, x0].add(fy * (1.0 - fx))
        agent_map = agent_map.at[y1, x1].add(fy * fx)
        return agent_map

    def get_obs(self, state: ContinuousEightRoomsState, params=None, key=None) -> jax.Array:
        if not self.use_visual_obs:
            if self.include_vel_in_obs:
                return jnp.array([
                    state.pos[0], state.pos[1],
                    state.vel[0], state.vel[1],
                    state.goal[0], state.goal[1],
                ], dtype=jnp.float32)
            else:
                return jnp.array([
                    state.pos[0], state.pos[1],
                    state.goal[0], state.goal[1],
                ], dtype=jnp.float32)
        else:
            agent_map = self._render_agent_map(state.pos)
            return jnp.stack([self.occupied_map, agent_map], axis=-1)

    def is_terminal(self, state: ContinuousEightRoomsState, params: ContinuousEightRoomsParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        dist_to_goal = jnp.linalg.norm(state.pos - state.goal)
        done_goal = dist_to_goal <= params.goal_radius
        return jnp.logical_or(done_goal, done_steps)

    @property
    def name(self) -> str:
        return "ContinuousEightRooms"

    def action_space(self, params: ContinuousEightRoomsParams | None = None) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=jnp.float32)

    def observation_space(self, params: ContinuousEightRoomsParams | None = None) -> spaces.Box:
        if self.use_visual_obs:
            return spaces.Box(0.0, 1.0, (self.height, self.width, 2), jnp.float32)
        else:
            obs_dim = 6 if self.include_vel_in_obs else 4
            high = max(float(self.height), float(self.width))
            return spaces.Box(low=-high, high=high, shape=(obs_dim,), dtype=jnp.float32)

    def state_space(self, params: ContinuousEightRoomsParams) -> spaces.Dict:
        high = max(float(self.height), float(self.width))
        return spaces.Dict({
            "pos": spaces.Box(0.0, high, (2,), jnp.float32),
            "vel": spaces.Box(-params.max_vel, params.max_vel, (2,), jnp.float32),
            "goal": spaces.Box(0.0, high, (2,), jnp.float32),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })


@struct.dataclass
class ContinuousEightRoomsDenseParams(ContinuousEightRoomsParams):
    gamma: float = 0.99
    potential_scale: float = 0.03125


class ContinuousEightRoomsDense(ContinuousEightRooms):
    """
    Continuous EightRooms with Potential-Based Reward Shaping (PBRS).
    Uses continuous bilinear interpolation of the shortest-path geodesic distance grid
    to evaluate smooth potentials Phi(s) = - potential_scale * GeodesicDist(s, goal).
    Preserves the optimal policy while enabling direct 2x2 comparison:
    [Continuous vs Discrete] x [Sparse vs Dense Shaped].
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        include_vel_in_obs: bool = False,
        goal_fixed: Tuple[float, float] = (23.5, 11.5),
        pos_fixed: Tuple[float, float] = (3.5, 1.5),
        gamma: float = 0.99,
        potential_scale: float = 0.03125,
        **kwargs,
    ):
        super().__init__(
            use_visual_obs=use_visual_obs,
            include_vel_in_obs=include_vel_in_obs,
            goal_fixed=goal_fixed,
            pos_fixed=pos_fixed,
            **kwargs,
        )
        self.gamma = float(gamma)
        self.potential_scale = float(potential_scale)

        # Precompute geodesic distance grid
        discrete_env = EightRoomsDense(use_visual_obs=False)
        self.dist_grid = discrete_env.dist_grid

    @property
    def default_params(self) -> ContinuousEightRoomsDenseParams:
        return ContinuousEightRoomsDenseParams(
            gamma=self.gamma,
            potential_scale=self.potential_scale,
        )

    def _get_continuous_geodesic_distance(self, pos: jax.Array) -> jax.Array:
        """Bilinear interpolation of the precomputed geodesic distance grid."""
        cy = pos[0] - 0.5
        cx = pos[1] - 0.5
        y0 = jnp.clip(jnp.floor(cy).astype(jnp.int32), 0, self.height - 1)
        x0 = jnp.clip(jnp.floor(cx).astype(jnp.int32), 0, self.width - 1)
        y1 = jnp.clip(y0 + 1, 0, self.height - 1)
        x1 = jnp.clip(x0 + 1, 0, self.width - 1)

        fy = jnp.clip(cy - y0, 0.0, 1.0)
        fx = jnp.clip(cx - x0, 0.0, 1.0)

        d00 = self.dist_grid[y0, x0]
        d01 = self.dist_grid[y0, x1]
        d10 = self.dist_grid[y1, x0]
        d11 = self.dist_grid[y1, x1]

        return (1.0 - fy) * (1.0 - fx) * d00 + (1.0 - fy) * fx * d01 + fy * (1.0 - fx) * d10 + fy * fx * d11

    def step_env(
        self,
        key: jax.Array,
        state: ContinuousEightRoomsState,
        action: jax.Array,
        params: ContinuousEightRoomsDenseParams,
    ) -> Tuple[jax.Array, ContinuousEightRoomsState, jax.Array, jax.Array, dict]:
        obs, new_state, sparse_reward, done, info = super().step_env(key, state, action, params)

        scale = getattr(params, "potential_scale", self.potential_scale)
        gamma = getattr(params, "gamma", self.gamma)

        cur_dist = self._get_continuous_geodesic_distance(state.pos)
        next_dist = self._get_continuous_geodesic_distance(new_state.pos)

        phi_s = - scale * cur_dist
        phi_next = jnp.where(info["is_goal"], 0.0, - scale * next_dist)
        shaping = gamma * phi_next - phi_s
        dense_reward = sparse_reward + shaping

        info["shaping_reward"] = shaping
        info["dense_reward"] = dense_reward
        info["geodesic_dist"] = next_dist

        return obs, new_state, dense_reward, done, info

    @property
    def name(self) -> str:
        return "ContinuousEightRooms-dense"
