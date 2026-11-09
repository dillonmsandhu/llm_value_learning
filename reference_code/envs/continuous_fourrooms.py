from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple, Any
from flax import struct
from gymnax.environments import environment, spaces
from envs.continuous_eightrooms import (
    ContinuousEightRooms,
    ContinuousEightRoomsParams,
    ContinuousEightRoomsDenseParams,
    ContinuousEightRoomsState,
)
from envs.fourrooms import FourRoomsExactValue, FourRoomsDenseExactValue


@struct.dataclass
class ContinuousFourRoomsState(ContinuousEightRoomsState):
    pass


@struct.dataclass
class ContinuousFourRoomsParams(ContinuousEightRoomsParams):
    max_steps_in_episode: int = 1000
    step_size: float = 0.5
    agent_radius: float = 0.2
    goal_radius: float = 0.6
    action_noise: float = 0.0
    fail_prob: float = 0.0
    substeps: int = 4
    control_mode: int = 0
    damping: float = 0.2
    dt: float = 0.1
    max_vel: float = 1.0


@struct.dataclass
class ContinuousFourRoomsDenseParams(ContinuousFourRoomsParams):
    gamma: float = 0.99
    potential_scale: float = 0.03125


class ContinuousFourRooms(ContinuousEightRooms):
    """
    Gymnax-compatible Continuous Control Environment for FourRooms (2 rooms high x 2 rooms wide, 13x13 grid).

    Features:
    - Continuous 2D state space: pos = (y, x) in [0, 13] x [0, 13].
    - Continuous 2D action space: action = (a_y, a_x) in [-1.0, 1.0]^2.
    - Exact same 4 rooms maze geometry and doorway connectivity as discrete FourRooms.
    - Same sparse reward structure: +1.0 only upon reaching the goal region, 0.0 elsewhere.
    - Vectorized, branch-free JAX collision detection and wall sliding physics.
    - Supports both kinematic (velocity) and dynamic (acceleration) modes.
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        include_vel_in_obs: bool = False,
        goal_fixed: Tuple[float, float] = (11.5, 11.5),
        pos_fixed: Tuple[float, float] = (3.5, 1.5),
        substeps: int = 4,
    ):
        super().__init__(
            use_visual_obs=use_visual_obs,
            include_vel_in_obs=include_vel_in_obs,
            goal_fixed=goal_fixed,
            pos_fixed=pos_fixed,
            height=13,
            width=13,
            substeps=substeps,
        )
        # Load 4-room binary occupancy map
        evaluator = FourRoomsExactValue(use_visual_obs=False)
        self.env_map = evaluator.env_map
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)

        # Free cell coordinates
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        self.num_cells = int(self.coords.shape[0])

    @property
    def default_params(self) -> ContinuousFourRoomsParams:
        return ContinuousFourRoomsParams()

    @property
    def name(self) -> str:
        return "ContinuousFourRooms"


class ContinuousFourRoomsDense(ContinuousFourRooms):
    """
    Continuous FourRooms with Potential-Based Reward Shaping (PBRS).
    Uses continuous bilinear interpolation of the shortest-path geodesic distance grid
    to evaluate smooth potentials Phi(s) = - potential_scale * GeodesicDist(s, goal).
    Preserves the optimal policy while providing dense progress guidance.
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        include_vel_in_obs: bool = False,
        goal_fixed: Tuple[float, float] = (11.5, 11.5),
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

        # Precompute geodesic distance grid from FourRoomsDenseExactValue
        exact_eval = FourRoomsDenseExactValue(
            use_visual_obs=False,
            potential_scale=potential_scale,
            gamma=gamma,
            goal_pos=(11, 11),
        )
        dist_grid = jnp.zeros((self.height, self.width), dtype=jnp.float32)
        coords = exact_eval.coords
        self.dist_grid = dist_grid.at[coords[:, 0], coords[:, 1]].set(
            exact_eval.distances[:exact_eval.num_states]
        )

    @property
    def default_params(self) -> ContinuousFourRoomsDenseParams:
        return ContinuousFourRoomsDenseParams(
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
        state: ContinuousFourRoomsState,
        action: jax.Array,
        params: ContinuousFourRoomsDenseParams,
    ) -> Tuple[jax.Array, ContinuousFourRoomsState, jax.Array, jax.Array, dict]:
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
        return "ContinuousFourRooms-dense"
