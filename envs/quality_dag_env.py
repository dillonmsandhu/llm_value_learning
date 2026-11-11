"""Critic-only reasoning DAG with persistent, history-reconstructable path quality."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from .reasoning_dag_env import DAGState, ReasoningDAGEnv


class QualityDAGEnv(ReasoningDAGEnv):
    """Fixed-policy DAG. Quality is privileged state, never an observation token.

    Six action classes combine {up, down, stay} with {advance, no advance}.
    Their token assignments depend on problem and milestone, not random events
    inside step(). Extra action IDs alias these six classes; EOS is last.
    """
    VERSION = "path_quality_v1"

    def __init__(self, num_problems=20, num_milestones=12, max_steps=40,
                 num_actions=8, difficulties="spectrum", variable_depth=True,
                 seed=42, eval_seed=12345, quality_levels=7, quality_change_rate=0.35,
                 quality_up_bias=0.6, quality_jump=1, absorbing_dead=False,
                 initial_quality=None, success_quality=None, gamma=1.0):
        if quality_levels < 2 or num_actions < 7 or num_milestones < 1 or max_steps < 1 or num_problems < 1:
            raise ValueError("Need Q>=2, actions>=7, and positive problem/milestone/horizon counts")
        if not 0 <= quality_change_rate <= 1 or not 0 <= quality_up_bias <= 1:
            raise ValueError("Quality change rate and up bias must lie in [0,1]")
        if not 1 <= quality_jump < quality_levels or not 0 <= gamma <= 1:
            raise ValueError("Require 1 <= jump < Q and gamma in [0,1]")
        self.num_problems, self.num_milestones = int(num_problems), int(num_milestones)
        self.max_steps, self.num_actions = int(max_steps), int(num_actions)
        self.quality_levels, self.gamma = int(quality_levels), float(gamma)
        self.eval_seed = int(eval_seed)
        self.initial_quality = quality_levels // 2 if initial_quality is None else int(initial_quality)
        self.success_quality = quality_levels // 2 if success_quality is None else int(success_quality)
        if not 0 <= self.initial_quality < quality_levels or not 1 <= self.success_quality < quality_levels:
            raise ValueError("Initial quality must be in [0,Q); success quality in [1,Q)")
        if isinstance(difficulties, str) and difficulties == "spectrum":
            theta = np.linspace(0, 1, num_problems) if num_problems > 1 else np.array([0.5])
        else:
            theta = np.broadcast_to(0.5 if difficulties is None else difficulties, (num_problems,)).copy()
        if not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > 1)):
            raise ValueError("Difficulty must lie in [0,1]")
        rng = np.random.default_rng(seed)
        horizons = np.full(num_problems, max_steps, dtype=np.int32)
        if variable_depth and num_problems > 1:
            lo = min(16, max_steps)
            horizons = np.clip(np.rint(lo + (max_steps-lo)*theta).astype(int) + rng.integers(-2, 3, num_problems), lo, max_steps)
        required = np.minimum(num_milestones, np.maximum(1, horizons // 2))
        if variable_depth:
            required = np.maximum(1, np.rint(required * (0.6 + 0.4*theta)).astype(int))
        self.thetas = jnp.asarray(theta, dtype=jnp.float32)
        self.problem_max_steps, self.required_milestones = jnp.asarray(horizons), jnp.asarray(required)
        self.PAD_TOKEN, self.BOS_TOKEN, self.SEP_TOKEN = 0, 1, 2
        self.PROMPT_TOKEN_OFFSET, self.prompt_len = 3, 3
        self.ACTION_TOKEN_OFFSET = 3 + num_problems
        self.vocab_size = self.ACTION_TOKEN_OFFSET + num_actions + 4
        self.max_seq_len = self.prompt_len + max_steps
        self.eos_action = num_actions - 1
        P, M, Q, A = num_problems, num_milestones, quality_levels, num_actions
        # Every milestone has all six joint effects, even when extra IDs alias them.
        classes = np.empty((P, M+1, A), dtype=np.int32)
        for p in range(P):
            for m in range(M+1):
                classes[p,m,:-1] = rng.permutation(np.arange(A-1) % 6)
                classes[p,m,-1] = 6
        c = classes[:,:,None,:]
        p, m, q, a = np.indices((P,M+1,Q,A))
        x = q / (Q-1)
        up_share = np.clip(quality_up_bias + 0.2*(x-0.5) - 0.25*(theta[p]-0.5), 0.02, 0.98)
        p_up = quality_change_rate * up_share
        p_down = quality_change_rate * (1-up_share)
        quality_mass = np.where(c//2 == 0, p_up, np.where(c//2 == 1, p_down, 1-quality_change_rate))
        advance = np.clip((0.08 + 0.87*x)*(1-0.25*theta[p]), 0.02, 0.98)
        mass = quality_mass * np.where(c%2 == 0, advance, 1-advance)
        counts = np.stack([(classes==i).sum(-1) for i in range(6)], -1)
        alias_count = np.take_along_axis(counts[:,:,None,:], np.broadcast_to(np.minimum(c,5), (P,M+1,Q,A)), axis=-1)
        mass = mass / alias_count
        eos_mass = np.where(m >= required[p], np.where(q >= self.success_quality, 0.85, 0.10), 0.)
        probs = np.where(a == self.eos_action, eos_mass, mass*(1-eos_mass))
        delta = np.where(c//2 == 0, quality_jump, np.where(c//2 == 1, -quality_jump, 0))
        next_q = np.clip(q+delta, 0, Q-1)
        next_m = np.minimum(m+(c%2 == 0), required[p])
        if absorbing_dead:
            next_q = np.where(q==0, 0, next_q)
            next_m = np.where(q==0, m, next_m)
        # Effective quality-change probabilities conditional on not choosing EOS.
        non_eos = np.where(a == self.eos_action, 0., probs)
        denom = np.maximum(non_eos.sum(-1), 1e-12)
        self.p_up = jnp.asarray((non_eos*(next_q > q)).sum(-1)/denom, dtype=jnp.float32)
        self.p_down = jnp.asarray((non_eos*(next_q < q)).sum(-1)/denom, dtype=jnp.float32)
        self.action_prob_table = jnp.asarray(probs, dtype=jnp.float32)
        self.next_quality_table, self.next_milestone_table = jnp.asarray(next_q), jnp.asarray(next_m)
        self.evaluator = self  # Same transition tables drive sampling, stepping, and exact DP.
        self.config = dict(version=self.VERSION, num_problems=P, num_milestones=M,
                           max_steps=max_steps, num_actions=A, quality_levels=Q,
                           quality_change_rate=quality_change_rate, quality_up_bias=quality_up_bias,
                           quality_jump=quality_jump, absorbing_dead=absorbing_dead,
                           initial_quality=self.initial_quality, success_quality=self.success_quality,
                           variable_depth=variable_depth, seed=seed, eval_seed=eval_seed,
                           gamma=gamma, difficulties=theta.tolist())
        self.V_table = self.solve_backward_induction()

    def _outcomes(self, p, t, m, q, a, xp=jnp):
        nm = xp.asarray(self.next_milestone_table)[p,m,q,a]
        nq = xp.asarray(self.next_quality_table)[p,m,q,a]
        eos = a == self.eos_action
        nm, nq = xp.where(eos,m,nm), xp.where(eos,q,nq)
        done = eos | (t+1 >= xp.asarray(self.problem_max_steps)[p])
        reward = (done & (nm >= xp.asarray(self.required_milestones)[p]) & (nq >= self.success_quality)).astype(xp.float32)
        return nm, nq, reward, done

    def solve_backward_induction(self):
        P,M,Q,A = self.num_problems,self.num_milestones,self.quality_levels,self.num_actions
        p,m,q,a = np.indices((P,M+1,Q,A))
        values = np.zeros((P,self.max_steps+1,M+1,Q), dtype=np.float32)
        probs = np.asarray(self.action_prob_table)
        for t in range(self.max_steps-1,-1,-1):
            nm,nq,r,d = self._outcomes(p,t,m,q,a,np)
            backup = (probs*(r+self.gamma*(~d)*values[p,t+1,nm,nq])).sum(-1)
            values[:,t] = np.where(t < np.asarray(self.problem_max_steps)[:,None,None],backup,0.)
        return jnp.asarray(values)

    def get_exact_value(self, state):
        v = self.V_table[state.prompt_idx,state.step_count,state.milestone,state.path_quality]
        return jnp.where(state.done,0.,v)

    def reset(self, rng, params=None):
        obs, st = super().reset(rng,params)
        return obs, st._replace(path_quality=jnp.array(self.initial_quality,jnp.int32))

    def action_probabilities(self, state):
        probs = self.action_prob_table[state.prompt_idx,state.milestone,state.path_quality]
        return jnp.where(state.done, jax.nn.one_hot(self.eos_action,self.num_actions),probs)

    def sample_base_action(self, rng, state, params=None):
        return jax.random.categorical(rng,jnp.log(self.action_probabilities(state))).astype(jnp.int32)

    def step(self, rng, state, action, params=None):
        nm,nq,r,d = self._outcomes(state.prompt_idx,state.step_count,state.milestone,state.path_quality,action)
        obs = state.history_tokens.at[self.prompt_len+state.step_count].set(self.ACTION_TOKEN_OFFSET+action)
        st = DAGState(state.prompt_idx,jnp.where(state.done,state.milestone,nm),
                      jnp.where(state.done,state.path_quality,nq),
                      jnp.where(state.done,state.step_count,state.step_count+1),
                      jnp.where(state.done,state.history_tokens,obs),state.done|d)
        return st.history_tokens,st,jnp.where(state.done,0.,r),st.done,{
            "milestone":st.milestone,"path_quality":st.path_quality,
            "step":st.step_count,"exact_v":self.get_exact_value(st)}

    def diagnostics(self, trajectories=512, seed=2026):
        """On-policy diagnostics; no privileged information enters critic inputs."""
        reset = jax.random.split(jax.random.PRNGKey(seed), trajectories)
        _, states = jax.vmap(self.reset)(reset)
        def scan_step(carry, _):
            st, key = carry
            key, ak, sk = jax.random.split(key,3)
            probs = jax.vmap(self.action_probabilities)(st)
            actions = jax.vmap(self.sample_base_action)(jax.random.split(ak,trajectories),st)
            _, ns, r, _, _ = jax.vmap(self.step)(jax.random.split(sk,trajectories),st,actions)
            p,m,q = st.prompt_idx,st.milestone,st.path_quality
            a = jnp.arange(self.num_actions)[None,:]
            nm,nq,ar,ad = self._outcomes(p[:,None],st.step_count[:,None],m[:,None],q[:,None],a)
            av = ar+self.gamma*(1-ad)*self.V_table[p[:,None],jnp.minimum(st.step_count[:,None]+1,self.max_steps),nm,nq]
            mean = (probs*av).sum(-1)
            variance = (probs*(av-mean[:,None])**2).sum(-1)
            return (ns,key),(~st.done,q,ns.path_quality,r,variance)
        run = jax.jit(lambda: jax.lax.scan(scan_step,(states,jax.random.PRNGKey(seed+1)),None,length=self.max_steps)[1])
        active,q,nq,r,var = map(np.asarray,run())
        counts = np.bincount(q[active],minlength=self.quality_levels)
        transitions = int(active.sum())
        unchanged = float(np.mean(q[active]==nq[active]))
        result = dict(trajectories=trajectories, active_transitions=transitions,
                      success_rate=float(r.sum()/trajectories),
                      quality_occupancy=(counts/max(1,counts.sum())).tolist(),
                      quality_unchanged_fraction=unchanged,
                      quality_up_fraction=float(np.mean(nq[active]>q[active])),
                      quality_down_fraction=float(np.mean(nq[active]<q[active])),
                      mean_successor_value_variance=float(var[active].mean()))
        pairs = np.stack([q[active],nq[active]])
        result['quality_lag1_correlation'] = float(np.corrcoef(pairs)[0,1]) if np.all(pairs.std(axis=1)>0) else None
        return result
