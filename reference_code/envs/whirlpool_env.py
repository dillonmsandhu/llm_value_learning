from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from typing import Any, Tuple

@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    goal: jax.Array
    time: int

@struct.dataclass
class EnvParams(environment.EnvParams):
    # fail_prob is the strength of the current. 0.9 = 90% chance to be swept clockwise
    fail_prob: float = 0.9 
    max_steps_in_episode: int = 1e6
try:
    _EnvBase = environment.Environment[EnvState, EnvParams]
except (TypeError, AttributeError):
    _EnvBase = environment.Environment

class Whirlpool(_EnvBase):
    """JAX/Gymnax implementation of the Whirlpool environment."""

    def __init__(
        self,
        size: int = 13,
        use_visual_obs: bool = False,
    ):
        super().__init__()
        self.N = size
        self.use_visual_obs = use_visual_obs

        # 1. Map Generation (Completely open)
        self.env_map = jnp.ones((self.N, self.N), dtype=bool)
        self.occupied_map = jnp.zeros((self.N, self.N), dtype=jnp.float32)
        
        # 2. Coordinates
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32)
        
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

        # 3. Fixed Goal (Center)
        self.center_idx = self.N // 2
        self.goal_fixed = jnp.array([self.center_idx, self.center_idx], dtype=jnp.int32)

        # 4. Precompute Outer Ring for Uniform Resets
        outer_ring = []
        for y_idx in range(self.N):
            for x_idx in range(self.N):
                if max(abs(y_idx - self.center_idx), abs(x_idx - self.center_idx)) == self.center_idx:
                    outer_ring.append([y_idx, x_idx])
        self.outer_ring_coords = jnp.array(outer_ring, dtype=jnp.int32)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        """Perform single timestep state transition using vectorized physics."""
        y, x = state.pos[0], state.pos[1]
        cy, cx = self.center_idx, self.center_idx

        # 1. Determine Ring Distance
        d = jnp.maximum(jnp.abs(y - cy), jnp.abs(x - cx))

        # 2. Vectorized Edge Masks
        is_top = (y == cy - d) & (x < cx + d)
        is_right = (x == cx + d) & (y < cy + d)
        is_bottom = (y == cy + d) & (x > cx - d)
        is_left = (x == cx - d) & (y > cy - d)

        # 3. Map semantic directions based on current edge
        cw = jnp.where(is_top, 1, jnp.where(is_right, 2, jnp.where(is_bottom, 3, 0)))
        ccw = jnp.where(is_top, 3, jnp.where(is_right, 0, jnp.where(is_bottom, 1, 2)))
        inward = jnp.where(is_top, 2, jnp.where(is_right, 3, jnp.where(is_bottom, 0, 1)))
        outward = jnp.where(is_top, 0, jnp.where(is_right, 1, jnp.where(is_bottom, 2, 3)))

        # 4. Roll for current strength
        p_roll = jax.random.uniform(key)
        swept_away = p_roll < params.fail_prob

        # 5. Resolve Physics
        # Determine what semantic action the chosen cardinal action maps to
        tried_cw = (action == cw)
        tried_outward = (action == outward)

        # If they tried CW or Outward, or if they were swept away, force CW.
        # Otherwise, let them execute their intended action (CCW or Inward).
        executed_a = jnp.where(
            tried_cw | tried_outward,
            cw,
            jnp.where(swept_away, cw, action)
        )

        # 6. Execute Movement (with boundary clipping)
        proposed_pos = state.pos + self.directions[executed_a]
        new_pos = jnp.clip(proposed_pos, 0, self.N - 1)

        # 7. Check Goal & Reward (Entry-Reward Style)
        is_goal = jnp.logical_and(new_pos[0] == state.goal[0], new_pos[1] == state.goal[1])
        reward = is_goal.astype(jnp.float32)

        # 8. Update State & Terminate
        state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment uniformly onto the outer ring."""
        # Sample index for outer ring
        idx = jax.random.randint(key, (), 0, self.outer_ring_coords.shape[0])
        pos = self.outer_ring_coords[idx]
        
        state = EnvState(pos=pos, goal=self.goal_fixed, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        if not self.use_visual_obs:
            return jnp.array([state.pos[0], state.pos[1], state.goal[0], state.goal[1]], dtype=jnp.float32)
        else:
            agent_map = jnp.zeros_like(self.occupied_map)
            agent_map = agent_map.at[state.pos[1], state.pos[0]].set(1) # Replicate Gymnax Transposition
            return jnp.stack([self.occupied_map, agent_map], axis=2)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done_steps = state.time >= params.max_steps_in_episode
        done_goal = jnp.logical_and(state.pos[0] == state.goal[0], state.pos[1] == state.goal[1])
        return jnp.logical_or(done_goal, done_steps)

    @property
    def name(self) -> str:
        return "Whirlpool-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        if self.use_visual_obs:
            return spaces.Box(0, 1, (self.N, self.N, 2), jnp.float32)
        else:
            return spaces.Box(0, self.N - 1, (4,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict({
            "pos": spaces.Box(0, self.N - 1, (2,), jnp.float32),
            "goal": spaces.Box(0, self.N - 1, (2,), jnp.float32),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })