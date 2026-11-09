from envs.fourrooms import *
class ContinuingFourRooms(FourRoomsExactValue):
    """
    An infinite-horizon, continuing variant of Four Rooms where the terminal
    state exists in the tensors but is physically isolated/inaccessible.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override standard dynamics to use the non-terminating continuing dynamics
        # This keeps the exact same tensor sizes but alters the transitions
        self.P, self.R = self._build_env_dynamics(continuing=True)
        self.P_cont = self.P

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        """
        Builds P and R where 'continuing=True' forces a macroscopic 
        one-way cycle (Goal -> Start) without entering the terminal state.
        """
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions), dtype=np.float32)

        p_correct = 1.0 - self.fail_prob
        p_wrong = self.fail_prob / 3.0

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx]

            # --- 1. GOAL STATE TRANSITION ---
            if s_idx == self.goal_idx:
                # Teleport back to start instead of entering terminal state
                P[s_idx, :, self.start_idx] = 1.0
                R[s_idx, :] = 1.0  # Award reward on teleportation exit
                continue

            # --- 2. STANDARD STATES ---
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    prob = p_correct if executed_a == chosen_a else p_wrong
                    if prob == 0: continue
                        
                    next_pos = self._step_pos(pos, executed_a)
                    hits_goal = bool(jnp.all(next_pos == self.goal))
                    
                    if hits_goal:
                        next_idx = self.goal_idx
                        P[s_idx, chosen_a, next_idx] += prob
                        R[s_idx, chosen_a] += prob * 0.0
                    else:
                        next_idx = self._coord_to_idx(next_pos)
                        P[s_idx, chosen_a, next_idx] += prob

        # Maintain isolated self-loop for terminal state to preserve row-stochasticity 
        # and keep matrix inversion solvers stable.
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        
        return jnp.asarray(P), jnp.asarray(R)