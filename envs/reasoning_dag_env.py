from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict, Any, NamedTuple


class DAGState(NamedTuple):
    """
    Internal Markov state shared by the binary and multi-level quality DAGs.
    Note: Network observations ONLY see `history_tokens`, never privileged (milestone, path_quality).
    """
    prompt_idx: jnp.ndarray      # int32 scalar: problem ID in 0 .. num_problems - 1
    milestone: jnp.ndarray       # int32 scalar: milestones completed in 0 .. num_milestones
    path_quality: jnp.ndarray    # Binary legacy: 0=on track, 1=derailed. Quality DAG: higher=better.
    step_count: jnp.ndarray      # int32 scalar: current step t in 0 .. max_steps
    history_tokens: jnp.ndarray  # int32 array: (max_seq_len,) full sequence of token IDs
    done: jnp.ndarray            # bool scalar: whether trajectory has terminated


@struct.dataclass
class DAGEnvParams:
    p_advance: float = 0.65       # Base policy probability of picking valid advance tokens (split across valid routes)
    p_blunder: float = 0.15       # Base policy probability of taking a blunder / dead-end branch
    p_recover: float = 0.35       # Base policy probability of self-correcting when derailed
    p_neutral: float = 0.15       # Base policy probability of fluff/neutral steps
    p_eos: float = 0.80           # Base policy probability of emitting EOS once milestones are complete
    gamma: float = 1.0


class ReasoningDAGExactValue:
    """
    Exact analytical dynamic programming solver for the Procedural Reasoning DAG.
    Computes ground-truth value functions V*(prompt_idx, t, milestone, path_quality)
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
        variable_depth: bool = True,
        gamma: float = 1.0,
        seed: int = 42,
    ):
        self.num_problems = int(num_problems)
        self.num_milestones = int(num_milestones)
        self.max_steps = int(max_steps)
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)

        # 1. Unified Difficulty Spectrum Parameterization: theta in [0, 1]
        # Calibrated to LLM reasoning: higher step-advance success, varied depths, low error recovery
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
        rng = jax.random.PRNGKey(seed)
        rng, depth_rng, mile_rng, p_rng = jax.random.split(rng, 4)

        if variable_depth and self.num_problems > 1:
            min_h = max(12, min(16, self.max_steps))
            base_h = min_h + (self.max_steps - min_h) * (0.15 + 0.70 * self.thetas)
            jitter_h = jax.random.randint(depth_rng, shape=(self.num_problems,), minval=-4, maxval=5)
            self.problem_max_steps = jnp.clip(jnp.round(base_h).astype(jnp.int32) + jitter_h, min_h, self.max_steps)
            min_m = min(4, self.num_milestones)
            max_m_cap = jnp.maximum(min_m, jnp.minimum(self.num_milestones, self.problem_max_steps // 2 - 1))
            base_m = min_m + (max_m_cap - min_m) * self.thetas
            jitter_m = jax.random.randint(mile_rng, shape=(self.num_problems,), minval=-1, maxval=2)
            self.required_milestones = jnp.clip(jnp.round(base_m).astype(jnp.int32) + jitter_m, min_m, max_m_cap)
        else:
            self.problem_max_steps = jnp.full(self.num_problems, self.max_steps, dtype=jnp.int32)
            min_m = min(4, self.num_milestones)
            max_m_cap = min(self.num_milestones, max(min_m, self.max_steps // 2))
            self.required_milestones = jnp.round(min_m + self.thetas * (max_m_cap - min_m)).astype(jnp.int32)

        # Higher step success prob, moderate blunder rate, and low recovery probability in general
        self.p_advance = 0.85 - self.thetas * 0.30
        self.p_recover = 0.15 - self.thetas * 0.10
        self.p_blunder = 0.07 + self.thetas * 0.23
        self.p_neutral = jnp.maximum(0.02, 1.0 - self.p_advance - self.p_blunder)
        # Base policy probability of emitting EOS once milestones are complete:
        self.p_eos = 0.85 - self.thetas * 0.15

        # 2. Procedural generation of problem tables
        # Action (num_actions - 2) is designated as the Meta Self-Correction / Verification operator
        # Action (num_actions - 1) is designated as the EOS (Terminal Submission) operator
        assert self.num_actions >= 4, "ReasoningDAGEnv requires at least 4 actions (>=2 advance, 1 correction, 1 EOS)"
        self.correction_action = self.num_actions - 2
        self.eos_action = self.num_actions - 1
        num_standard_actions = self.num_actions - 2

        # For each problem and milestone, we define 2 valid advance tokens (allowing reconverging paths)

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
        # where last dim is path_quality in {0, 1}
        self.V_table = self.solve_backward_induction()

    def solve_backward_induction(self) -> jnp.ndarray:
        """
        Solves exact Bellman backward induction for the base policy across the entire DAG.
        Returns V_table of shape (num_problems, max_steps + 1, num_milestones + 1, 2).
        Includes EOS termination semantics:
          - If m >= required_milestones and d == 0, emitting EOS yields immediate reward 1.0 (done=True).
          - If d == 1 or m < required_milestones, emitting EOS yields immediate reward 0.0 (done=True).
        """
        P = self.num_problems
        T = self.max_steps
        M = self.num_milestones

        v_table = jnp.zeros((P, T + 1, M + 1, 2), dtype=jnp.float32)

        # Terminal boundary at t = T (horizon cutoff fallback):
        # Reward is 1.0 iff milestone >= required_milestones[p] and path_quality == 0
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

                        # When milestones not yet reached:
                        expected_v_not_reached = (
                            self.p_advance * v_advance
                            + self.p_blunder * v_blunder
                            + self.p_neutral * v_neutral
                        )

                        # When milestones reached:
                        # Model emits EOS with p_eos -> immediate reward 1.0, done.
                        # Otherwise can verify (v_neutral) or blunder (v_blunder).
                        p_blund_reached = jnp.minimum(0.10, self.p_blunder)
                        p_neut_reached = jnp.maximum(0.02, 1.0 - self.p_eos - p_blund_reached)
                        expected_v_reached = (
                            self.p_eos * 1.0
                            + self.gamma * (p_neut_reached * v_neutral + p_blund_reached * v_blunder)
                        )
                        expected_v = jnp.where(m < self.required_milestones, self.gamma * expected_v_not_reached, expected_v_reached)
                    else:  # Derailed (d == 1)
                        v_recover = v_next_layer[:, m, 0]
                        v_stay_derailed = v_next_layer[:, m, 1]

                        # When milestones not yet reached:
                        expected_v_not_reached = (
                            self.p_recover * v_recover
                            + (1.0 - self.p_recover) * v_stay_derailed
                        )

                        # When milestones reached but derailed:
                        # Emitting EOS yields immediate reward 0.0 (done=True).
                        # Can recover (v_recover) or stay derailed (v_stay_derailed).
                        p_rec_reached = jnp.minimum(0.40, self.p_recover)
                        p_stay_reached = jnp.maximum(0.0, 1.0 - self.p_eos - p_rec_reached)
                        expected_v_reached = (
                            self.p_eos * 0.0
                            + self.gamma * (p_rec_reached * v_recover + p_stay_reached * v_stay_derailed)
                        )
                        expected_v = jnp.where(m < self.required_milestones, self.gamma * expected_v_not_reached, expected_v_reached)

                    val_t = jnp.where(t >= self.problem_max_steps, terminal_rewards[:, m, d], expected_v)
                    v_table = v_table.at[:, t, m, d].set(val_t)

        return v_table

    def get_exact_value(self, state: DAGState) -> jnp.ndarray:
        """Looks up the exact ground-truth value V*(s) for any state (scalar or batched)."""
        v = self.V_table[state.prompt_idx, state.step_count, state.milestone, state.path_quality]
        return jnp.where(state.done, 0.0, v)


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
        variable_depth: bool = True,
        seed: int = 42,
        eval_seed: int = 12345,
    ):
        self.eval_seed = int(eval_seed)
        self.evaluator = ReasoningDAGExactValue(
            num_problems=num_problems,
            num_milestones=num_milestones,
            max_steps=max_steps,
            num_actions=num_actions,
            difficulties=difficulties,
            variable_depth=variable_depth,
            seed=seed,
        )
        self.num_problems = self.evaluator.num_problems
        self.num_milestones = self.evaluator.num_milestones
        self.max_steps = self.evaluator.max_steps
        self.problem_max_steps = self.evaluator.problem_max_steps
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
        self.eos_action = self.evaluator.eos_action
        self.p_eos = self.evaluator.p_eos
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
            path_quality=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            history_tokens=history,
            done=jnp.array(False, dtype=jnp.bool_),
        )
        return history, init_state

    def sample_base_action(self, rng: jax.random.PRNGKey, state: DAGState, params: DAGEnvParams = None) -> jnp.ndarray:
        """
        Samples an action from the base model policy pi_0(a | s).
        Models realistic base LLM sampling with valid advances, blunders, self-corrections,
        and EOS (terminal conclusion) once milestones are achieved.
        """
        rng, choice_rng, act_rng = jax.random.split(rng, 3)

        curr_m = jnp.minimum(state.milestone, self.num_milestones - 1)
        valid_acts = self.valid_advance_actions[state.prompt_idx, curr_m]      # shape (2,)
        blunder_acts = self.blunder_actions[state.prompt_idx, curr_m]          # shape (num_actions - 4,)

        p_adv = self.p_advance[state.prompt_idx]
        p_blund = self.p_blunder[state.prompt_idx]
        p_rec = self.p_recover[state.prompt_idx]
        p_eos = self.p_eos[state.prompt_idx]

        r = jax.random.uniform(choice_rng)
        rand_valid_idx = jax.random.randint(act_rng, shape=(), minval=0, maxval=valid_acts.shape[0])
        rand_blunder_idx = jax.random.randint(act_rng, shape=(), minval=0, maxval=blunder_acts.shape[0])

        pick_advance = valid_acts[rand_valid_idx]
        pick_blunder = blunder_acts[rand_blunder_idx]
        pick_correction = jnp.array(self.correction_action, dtype=jnp.int32)
        pick_eos = jnp.array(self.eos_action, dtype=jnp.int32)

        m_reached = (state.milestone >= self.required_milestones[state.prompt_idx])

        # 1. Action selection when milestones NOT yet reached:
        act_on_track_not_reached = jnp.where(
            r < p_adv,
            pick_advance,
            jnp.where(r < p_adv + p_blund, pick_blunder, pick_correction),
        )
        act_derailed_not_reached = jnp.where(
            r < p_rec,
            pick_correction,
            pick_blunder,
        )

        # 2. Action selection when milestones ARE reached:
        # Model favors emitting EOS (submitting final solution)
        p_blund_reached = jnp.minimum(0.10, p_blund)
        act_on_track_reached = jnp.where(
            r < p_eos,
            pick_eos,
            jnp.where(r < p_eos + p_blund_reached, pick_blunder, pick_correction),
        )
        p_rec_reached = jnp.minimum(0.40, p_rec)
        act_derailed_reached = jnp.where(
            r < p_eos,
            pick_eos,
            jnp.where(r < p_eos + p_rec_reached, pick_correction, pick_blunder),
        )

        act_on_track = jnp.where(m_reached, act_on_track_reached, act_on_track_not_reached)
        act_derailed = jnp.where(m_reached, act_derailed_reached, act_derailed_not_reached)

        action = jnp.where(state.path_quality == 0, act_on_track, act_derailed)
        return jnp.where(state.done, pick_eos, action)

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
        Supports early termination via EOS operator (action == self.eos_action).
        """
        # Determine action category for current problem and milestone:
        is_eos = (action == self.eos_action)
        is_correction = (action == self.correction_action)
        curr_m = jnp.minimum(state.milestone, self.num_milestones - 1)
        is_advance = self.valid_advance_mask[state.prompt_idx, curr_m, action]

        # 1. State updates for standard (non-EOS) reasoning steps:
        new_milestone_on_track = jnp.where(
            is_advance & (state.milestone < self.required_milestones[state.prompt_idx]),
            state.milestone + 1,
            state.milestone,
        )
        new_derailed_on_track = jnp.where(
            is_advance | is_correction,
            0,
            1,  # Blunder committed
        )
        new_milestone_derailed = state.milestone
        new_derailed_derailed = jnp.where(is_correction, 0, 1)

        step_milestone = jnp.where(state.path_quality == 0, new_milestone_on_track, new_milestone_derailed)
        step_derailed = jnp.where(state.path_quality == 0, new_derailed_on_track, new_derailed_derailed)
        step_count = state.step_count + 1

        # 2. EOS termination logic:
        # If EOS emitted, episode terminates immediately.
        # Immediate reward is 1.0 iff required milestones reached and currently on-track!
        eos_success = (state.milestone >= self.required_milestones[state.prompt_idx]) & (state.path_quality == 0)
        eos_reward = jnp.where(eos_success, 1.0, 0.0)

        # 3. Horizon termination fallback (reached problem horizon without EOS):
        max_step_done = (step_count >= self.problem_max_steps[state.prompt_idx])
        max_step_success = (step_milestone >= self.required_milestones[state.prompt_idx]) & (step_derailed == 0)
        max_step_reward = jnp.where(max_step_done & max_step_success, 1.0, 0.0)

        # Overall termination and reward:
        new_done = is_eos | max_step_done | state.done
        raw_reward = jnp.where(is_eos, eos_reward, max_step_reward)
        reward = jnp.where(state.done, 0.0, raw_reward)

        # Final state attributes (frozen if previously done):
        final_milestone = jnp.where(state.done, state.milestone, step_milestone)
        final_derailed = jnp.where(state.done, state.path_quality, step_derailed)
        final_step_count = jnp.where(state.done, state.step_count, step_count)

        # Append emitted action token to sequence observation:
        token_id = self.evaluator.ACTION_TOKEN_OFFSET + action
        insert_pos = self.prompt_len + state.step_count
        new_history = jnp.where(
            state.done,
            state.history_tokens,
            state.history_tokens.at[insert_pos].set(token_id),
        )

        next_state = DAGState(
            prompt_idx=state.prompt_idx,
            milestone=final_milestone,
            path_quality=final_derailed,
            step_count=final_step_count,
            history_tokens=new_history,
            done=new_done,
        )

        info = {
            "milestone": final_milestone,
            "path_quality": final_derailed,
            "step": final_step_count,
            "exact_v": self.evaluator.get_exact_value(next_state),
        }

        return new_history, next_state, reward, new_done, info
