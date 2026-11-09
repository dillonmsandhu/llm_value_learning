from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict, Any, NamedTuple


class DAGState(NamedTuple):
    """
    Internal Markov state for the procedural reasoning DAG.
    Note: Network observations ONLY see `history_tokens`, never privileged (milestone, is_derailed).
    """
    prompt_idx: jnp.ndarray      # int32 scalar: problem ID in 0 .. num_problems - 1
    milestone: jnp.ndarray       # int32 scalar: milestones completed in 0 .. num_milestones
    is_derailed: jnp.ndarray     # int32 scalar: 0 = on track, 1 = in error/blunder detour
    step_count: jnp.ndarray      # int32 scalar: current step t in 0 .. max_steps
    history_tokens: jnp.ndarray  # int32 array: (max_seq_len,) full sequence of token IDs
    done: jnp.ndarray            # bool scalar: whether trajectory has terminated


@struct.dataclass
class DAGEnvParams:
    p_advance: float = 0.65       # Base policy probability of picking valid advance tokens (split across valid routes)
    p_blunder: float = 0.15       # Base policy probability of taking a blunder / dead-end branch
    p_recover: float = 0.35       # Base policy probability of self-correcting when derailed
    p_neutral: float = 0.15       # Base policy probability of fluff/neutral steps
    gamma: float = 1.0


class ReasoningDAGExactValue:
    """
    Exact analytical dynamic programming solver for the Procedural Reasoning DAG.
    Computes ground-truth value functions V*(prompt_idx, t, milestone, is_derailed)
    via backward induction across all problems and depth layers in milliseconds.
    Supports problem-dependent difficulty spectrum theta in [0, 1] governing:
      - required milestones M(theta)
      - advance probability p_advance(theta)
      - recovery probability p_recover(theta)
      - blunder probability p_blunder(theta)
    """

    def __init__(
        self,
        num_problems: int = 50,
        num_milestones: int = 4,
        max_steps: int = 16,
        num_actions: int = 8,
        difficulties: Any = None,
        gamma: float = 1.0,
        seed: int = 42,
    ):
        self.num_problems = int(num_problems)
        self.num_milestones = int(num_milestones)
        self.max_steps = int(max_steps)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)

        # 1. Unified Difficulty Spectrum Parameterization: theta in [0, 1]
        # theta = 0: trivial (Pass@1 ~ 85%), theta = 1: competition hard (Pass@1 ~ 1%)
        if difficulties is None:
            thetas = jnp.full(self.num_problems, 0.5, dtype=jnp.float32)
        elif isinstance(difficulties, (int, float)):
            thetas = jnp.full(self.num_problems, float(difficulties), dtype=jnp.float32)
        elif str(difficulties) == "spectrum":
            if self.num_problems == 1:
                thetas = jnp.array([0.5], dtype=jnp.float32)
            else:
                thetas = jnp.linspace(0.0, 1.0, self.num_problems, dtype=jnp.float32)
        else:
            thetas = jnp.asarray(difficulties, dtype=jnp.float32)

        self.thetas = thetas
        min_m = min(3, self.num_milestones)
        self.required_milestones = jnp.round(min_m + self.thetas * (self.num_milestones - min_m)).astype(jnp.int32)
        self.p_advance = 0.80 - self.thetas * 0.55
        self.p_recover = 0.50 - self.thetas * 0.40
        self.p_blunder = 0.10 + self.thetas * 0.55
        self.p_neutral = jnp.maximum(0.02, 1.0 - self.p_advance - self.p_blunder)

        # 2. Procedural generation of problem tables
        # Action (num_actions - 1) is universally designated as the Meta Self-Correction / Verification operator
        self.correction_action = self.num_actions - 1
        num_standard_actions = self.num_actions - 1

        # For each problem and milestone, we define 2 valid advance tokens (allowing reconverging paths)
        rng = jax.random.PRNGKey(seed)
        rng, p_rng = jax.random.split(rng)

        def _make_problem_routes(k):
            keys = jax.random.split(k, self.num_milestones)
            def _milestone_routes(mk):
                perm = jax.random.permutation(mk, num_standard_actions)
                valid_acts = perm[:2]
                blunder_acts = perm[2:]
                mask = jnp.zeros(self.num_actions, dtype=jnp.bool_).at[valid_acts].set(True)
                return valid_acts, blunder_acts, mask
            return jax.vmap(_milestone_routes)(keys)

        prob_keys = jax.random.split(p_rng, self.num_problems)
        self.valid_advance_actions, self.blunder_actions, self.valid_advance_mask = jax.vmap(_make_problem_routes)(prob_keys)

        # Vocabulary Constants:
        self.PAD_TOKEN = 0
        self.BOS_TOKEN = 1
        self.SEP_TOKEN = 2
        # Prompt tokens: 3 .. 3 + num_problems - 1
        self.PROMPT_TOKEN_OFFSET = 3
        # Action tokens: PROMPT_TOKEN_OFFSET + num_problems ..
        self.ACTION_TOKEN_OFFSET = self.PROMPT_TOKEN_OFFSET + self.num_problems
        self.vocab_size = self.ACTION_TOKEN_OFFSET + self.num_actions + 4
        self.prompt_len = 3  # [BOS, prompt_token, SEP]
        self.max_seq_len = self.prompt_len + self.max_steps

        # 3. Precompute Exact Backward Induction Table:
        # Table shape: (num_problems, max_steps + 1, num_milestones + 1, 2)
        # where last dim is is_derailed in {0, 1}
        self.V_table = self.solve_backward_induction()

    def solve_backward_induction(self) -> jnp.ndarray:
        """
        Solves exact Bellman backward induction for the base policy across the entire DAG.
        Returns V_table of shape (num_problems, max_steps + 1, num_milestones + 1, 2).
        """
        P = self.num_problems
        T = self.max_steps
        M = self.num_milestones

        v_table = jnp.zeros((P, T + 1, M + 1, 2), dtype=jnp.float32)

        # Terminal boundary at t = T:
        # Reward is 1.0 iff milestone >= required_milestones[p] and is_derailed == 0
        terminal_rewards = jnp.zeros((P, M + 1, 2), dtype=jnp.float32)
        for p in range(P):
            terminal_rewards = terminal_rewards.at[p, int(self.required_milestones[p]):, 0].set(1.0)
        v_table = v_table.at[:, T, :, :].set(terminal_rewards)

        # Backward sweep: t = T - 1 down to 0
        for t in range(T - 1, -1, -1):
            for m in range(M + 1):
                for d in range(2):
                    v_next_layer = v_table[:, t + 1, :, :]  # (P, M+1, 2)

                    if d == 0:  # On track
                        v_advance = v_next_layer[:, min(m + 1, M), 0]
                        v_blunder = v_next_layer[:, m, 1]
                        v_neutral = v_next_layer[:, m, 0]

                        expected_v_not_reached = (
                            self.p_advance * v_advance
                            + self.p_blunder * v_blunder
                            + self.p_neutral * v_neutral
                        )
                        expected_v_reached = (
                            (1.0 - self.p_blunder) * v_neutral
                            + self.p_blunder * v_blunder
                        )
                        expected_v = jnp.where(m < self.required_milestones, expected_v_not_reached, expected_v_reached)
                    else:  # Derailed (d == 1)
                        v_recover = v_next_layer[:, m, 0]
                        v_stay_derailed = v_next_layer[:, m, 1]
                        expected_v = (
                            self.p_recover * v_recover
                            + (1.0 - self.p_recover) * v_stay_derailed
                        )

                    v_table = v_table.at[:, t, m, d].set(self.gamma * expected_v)

        return v_table

    def get_exact_value(self, state: DAGState) -> jnp.ndarray:
        """Looks up the exact ground-truth value V*(s) for any state (scalar or batched)."""
        return self.V_table[state.prompt_idx, state.step_count, state.milestone, state.is_derailed]


class ReasoningDAGEnv:
    """
    Procedural Reasoning Directed Acyclic Graph (DAG) Environment.
    Simulates multi-problem chain-of-thought reasoning with multiple valid solution paths,
    blunder states, self-correction / verification reconvergence, and problem difficulty spectrum.
    """

    def __init__(
        self,
        num_problems: int = 50,
        num_milestones: int = 4,
        max_steps: int = 16,
        num_actions: int = 8,
        difficulties: Any = None,
        seed: int = 42,
    ):
        self.evaluator = ReasoningDAGExactValue(
            num_problems=num_problems,
            num_milestones=num_milestones,
            max_steps=max_steps,
            num_actions=num_actions,
            difficulties=difficulties,
            seed=seed,
        )
        self.num_problems = self.evaluator.num_problems
        self.num_milestones = self.evaluator.num_milestones
        self.max_steps = self.evaluator.max_steps
        self.num_actions = self.evaluator.num_actions
        self.thetas = self.evaluator.thetas
        self.required_milestones = self.evaluator.required_milestones
        self.p_advance = self.evaluator.p_advance
        self.p_recover = self.evaluator.p_recover
        self.p_blunder = self.evaluator.p_blunder
        self.vocab_size = self.evaluator.vocab_size
        self.max_seq_len = self.evaluator.max_seq_len
        self.prompt_len = self.evaluator.prompt_len
        self.correction_action = self.evaluator.correction_action
        self.valid_advance_mask = self.evaluator.valid_advance_mask
        self.valid_advance_actions = self.evaluator.valid_advance_actions
        self.blunder_actions = self.evaluator.blunder_actions

    def reset(self, rng: jax.random.PRNGKey, params: DAGEnvParams = None) -> Tuple[jnp.ndarray, DAGState]:
        """Resets environment by sampling a random problem prompt from the bank."""
        rng, p_rng = jax.random.split(rng)
        prompt_idx = jax.random.randint(p_rng, shape=(), minval=0, maxval=self.num_problems)

        # Build initial token sequence: [BOS, prompt_token, SEP, PAD, PAD, ...]
        history = jnp.full(self.max_seq_len, self.evaluator.PAD_TOKEN, dtype=jnp.int32)
        history = history.at[0].set(self.evaluator.BOS_TOKEN)
        history = history.at[1].set(self.evaluator.PROMPT_TOKEN_OFFSET + prompt_idx)
        history = history.at[2].set(self.evaluator.SEP_TOKEN)

        init_state = DAGState(
            prompt_idx=prompt_idx,
            milestone=jnp.array(0, dtype=jnp.int32),
            is_derailed=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            history_tokens=history,
            done=jnp.array(False, dtype=jnp.bool_),
        )
        return history, init_state

    def sample_base_action(self, rng: jax.random.PRNGKey, state: DAGState, params: DAGEnvParams = None) -> jnp.ndarray:
        """
        Samples an action from the base model policy pi_0(a | s).
        Models realistic base LLM sampling with valid advances, blunders, and self-corrections
        under problem-specific difficulty theta.
        """
        rng, choice_rng, act_rng = jax.random.split(rng, 3)

        curr_m = jnp.minimum(state.milestone, self.num_milestones - 1)
        valid_acts = self.valid_advance_actions[state.prompt_idx, curr_m]      # shape (2,)
        blunder_acts = self.blunder_actions[state.prompt_idx, curr_m]          # shape (num_actions - 3,)

        p_adv = self.p_advance[state.prompt_idx]
        p_blund = self.p_blunder[state.prompt_idx]
        p_rec = self.p_recover[state.prompt_idx]

        r = jax.random.uniform(choice_rng)
        rand_valid_idx = jax.random.randint(act_rng, shape=(), minval=0, maxval=valid_acts.shape[0])
        rand_blunder_idx = jax.random.randint(act_rng, shape=(), minval=0, maxval=blunder_acts.shape[0])

        pick_advance = valid_acts[rand_valid_idx]
        pick_blunder = blunder_acts[rand_blunder_idx]
        pick_correction = jnp.array(self.correction_action, dtype=jnp.int32)

        def _sample_on_track():
            return jnp.where(
                r < p_adv,
                pick_advance,
                jnp.where(r < p_adv + p_blund, pick_blunder, pick_correction),
            )

        def _sample_derailed():
            return jnp.where(r < p_rec, pick_correction, pick_blunder)

        action = jnp.where(state.is_derailed == 0, _sample_on_track(), _sample_derailed())
        return action

    def step(
        self,
        rng: jax.random.PRNGKey,
        state: DAGState,
        action: jnp.ndarray,
        params: DAGEnvParams = None,
    ) -> Tuple[jnp.ndarray, DAGState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Executes one reasoning step.
        action: int32 in 0 .. num_actions - 1.
        """
        # Determine action category for current problem and milestone:
        curr_m = jnp.minimum(state.milestone, self.num_milestones - 1)
        is_advance_action = self.valid_advance_mask[state.prompt_idx, curr_m, action]
        is_correction_action = (action == self.correction_action)

        # Transition logic:
        # 1. If on-track (d == 0):
        #    - advance action -> milestone increments (if not yet at problem-specific target)
        #    - correction action -> neutral verification (milestone stays, stays on track)
        #    - other action -> enters error state (is_derailed becomes 1)
        new_milestone_on_track = jnp.where(
            is_advance_action & (state.milestone < self.required_milestones[state.prompt_idx]),
            state.milestone + 1,
            state.milestone,
        )
        new_derailed_on_track = jnp.where(
            is_advance_action | is_correction_action,
            0,
            1,  # Blunder committed
        )

        # 2. If derailed (d == 1):
        #    - correction action -> successfully recovers to d = 0 (milestone preserved!)
        #    - any other action -> remains derailed (d = 1, milestone stays)
        new_milestone_derailed = state.milestone
        new_derailed_derailed = jnp.where(is_correction_action, 0, 1)

        # Select new milestone and derail state:
        new_milestone = jnp.where(state.is_derailed == 0, new_milestone_on_track, new_milestone_derailed)
        new_derailed = jnp.where(state.is_derailed == 0, new_derailed_on_track, new_derailed_derailed)
        new_step_count = state.step_count + 1

        # Append emitted action token to sequence observation:
        token_id = self.evaluator.ACTION_TOKEN_OFFSET + action
        insert_pos = self.prompt_len + state.step_count
        new_history = state.history_tokens.at[insert_pos].set(token_id)

        # Check termination:
        done = (new_step_count >= self.max_steps)

        # Terminal reward at t = T: 1.0 iff required milestones completed and on-track!
        reward = jnp.where(
            done & (new_milestone >= self.required_milestones[state.prompt_idx]) & (new_derailed == 0),
            1.0,
            0.0,
        )

        next_state = DAGState(
            prompt_idx=state.prompt_idx,
            milestone=new_milestone,
            is_derailed=new_derailed,
            step_count=new_step_count,
            history_tokens=new_history,
            done=done,
        )

        info = {
            "milestone": new_milestone,
            "is_derailed": new_derailed,
            "step": new_step_count,
            "exact_v": self.evaluator.get_exact_value(next_state),
        }

        return new_history, next_state, reward, done, info
