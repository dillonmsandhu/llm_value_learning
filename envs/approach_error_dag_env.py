"""Fixed-policy reasoning DAG with solution approaches and persistent errors."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from .quality_dag_env import QualityDAGEnv


class ApproachErrorDAGEnv(QualityDAGEnv):
    """A finite-horizon MRP; actions expose the fixed policy's transition choices.

    Shared action roles: three transformations, wait, check, reconsider, shortcut,
    and EOS. A transformation's appropriateness depends on the problem family,
    current approach, and milestone. Ordinary progress never silently repairs an
    error. A check repairs a slip; reconsideration repairs misconceptions at a
    progress cost and changes approach.

    For compatibility with the benchmark, DAGState.path_quality packs the latent
    pair as 3*approach+error_status. This is a categorical code, NOT ordered quality.
    Neither component is included in critic observations.
    """
    VERSION = "approach_error_v1"
    SOUND, SLIP, MISCONCEPTION = 0, 1, 2
    ERROR_NAMES = ("sound", "slip", "misconception")
    ACTION_NAMES = ("transform_0", "transform_1", "transform_2", "wait", "check", "reconsider", "shortcut")

    def __init__(self, num_problems=20, num_milestones=12, max_steps=40,
                 num_actions=8, difficulties="spectrum", variable_depth=True,
                 seed=42, eval_seed=12345, num_approaches=3, error_rate=1.0,
                 recovery_rate=1.0, shortcut_weight=0.04, wait_weight=0.10,
                 gamma=1.0):
        for name, value, minimum in (("num_problems",num_problems,1), ("num_milestones",num_milestones,1),
                                     ("max_steps",max_steps,1), ("num_actions",num_actions,8),
                                     ("num_approaches",num_approaches,2)):
            if int(value) != value or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name,value in (("error_rate",error_rate),("recovery_rate",recovery_rate),
                           ("shortcut_weight",shortcut_weight),("wait_weight",wait_weight)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        self.num_problems, self.num_milestones = int(num_problems),int(num_milestones)
        self.max_steps,self.num_actions = int(max_steps),int(num_actions)
        self.num_approaches,self.quality_levels = int(num_approaches),3*int(num_approaches)
        self.gamma,self.eval_seed = float(gamma),int(eval_seed)
        self.initial_quality = 0  # approach zero, sound; fixed and reconstructable
        if isinstance(difficulties,str) and difficulties == "spectrum":
            theta = np.linspace(0,1,num_problems) if num_problems > 1 else np.array([0.5])
        else:
            theta = np.broadcast_to(0.5 if difficulties is None else difficulties,(num_problems,)).copy()
        if not np.all(np.isfinite(theta)) or np.any((theta<0)|(theta>1)):
            raise ValueError("Difficulty must be in [0,1]")
        rng = np.random.default_rng(seed)
        horizons = np.full(num_problems,max_steps,dtype=np.int32)
        if variable_depth and num_problems > 1:
            low = min(16,max_steps)
            horizons = np.clip(np.rint(low+(max_steps-low)*theta).astype(int)+rng.integers(-2,3,num_problems),low,max_steps)
        required = np.minimum(num_milestones,np.maximum(1,horizons//2))
        if variable_depth:
            required = np.maximum(1,np.rint(required*(0.6+0.4*theta)).astype(int))
        self.thetas = jnp.asarray(theta,dtype=jnp.float32)
        self.problem_max_steps,self.required_milestones = jnp.asarray(horizons),jnp.asarray(required)
        self.problem_family = jnp.asarray(rng.integers(0,3,num_problems),dtype=jnp.int32)
        self.preferred_approach = jnp.asarray(rng.integers(0,num_approaches,num_problems),dtype=jnp.int32)
        self.PAD_TOKEN,self.BOS_TOKEN,self.SEP_TOKEN = 0,1,2
        self.PROMPT_TOKEN_OFFSET,self.prompt_len = 3,3
        self.ACTION_TOKEN_OFFSET = 3+num_problems
        self.vocab_size = self.ACTION_TOKEN_OFFSET+num_actions+4
        self.max_seq_len = 3+max_steps
        self.eos_action = num_actions-1
        P,M,L,A = num_problems,num_milestones,self.quality_levels,num_actions
        p,m,latent,a = np.indices((P,M+1,L,A))
        approach,error = latent//3,latent%3
        # Shared roles across tasks; only transformation appropriateness changes.
        role = np.where(a == self.eos_action,7,a%7)
        appropriate = (np.asarray(self.problem_family)[p]+approach+m)%3
        is_transform = role<3
        correct = is_transform & (role==appropriate)
        slip_action = is_transform & (role==(appropriate+1)%3)
        misconception_action = is_transform & (role==(appropriate+2)%3)
        check,reconsider,shortcut,eos = (role==i for i in (4,5,6,7))
        valid_shortcut = ((np.asarray(self.problem_family)[p]+approach+m)%3==0)
        # Apparent progress can continue even when a previous step was wrong.
        nm = np.minimum(m+is_transform,required[p])
        ne = np.where((error==self.SOUND)&slip_action,self.SLIP,error)
        ne = np.where((error==self.SOUND)&misconception_action,self.MISCONCEPTION,ne)
        # Repair a local slip by rolling back one apparent milestone.
        nm = np.where(check & (error==self.SLIP),np.maximum(m-1,0),nm)
        ne = np.where(check & (error==self.SLIP),self.SOUND,ne)
        # Changing approach discards more work; misconceptions require a restart.
        nr = np.where(reconsider,(approach+1)%num_approaches,approach)
        nm = np.where(reconsider,np.where(error==self.MISCONCEPTION,0,np.maximum(m-2,0)),nm)
        ne = np.where(reconsider,self.SOUND,ne)
        nm = np.where(shortcut,np.minimum(m+np.where(valid_shortcut,2,1),required[p]),nm)
        ne = np.where(shortcut & ~valid_shortcut,self.MISCONCEPTION,ne)
        next_latent = 3*nr+ne
        # EOS changes no semantic state. Termination/reward handled in _outcomes.
        nm,next_latent = np.where(eos,m,nm),np.where(eos,latent,next_latent)
        difficulty = theta[p]
        preferred = approach==np.asarray(self.preferred_approach)[p]
        weights = np.zeros((P,M+1,L,A),dtype=np.float64)
        weights += correct*np.clip(0.60-0.25*difficulty+0.10*preferred,0.05,0.95)
        weights += slip_action*error_rate*(0.07+0.10*difficulty)
        weights += misconception_action*error_rate*(0.03+0.07*difficulty)
        weights += (role==3)*wait_weight
        # The fixed base policy only imperfectly recognizes its errors.
        weights += check*recovery_rate*np.where(error==self.SLIP,0.16,0.04)*(1-0.4*difficulty)
        weights += reconsider*recovery_rate*np.where(error==self.MISCONCEPTION,0.10,0.015)*(1-0.5*difficulty)
        weights += shortcut*shortcut_weight*(1-0.3*difficulty)
        weights += eos*np.where(m>=required[p],0.8,0.003+0.007*difficulty)
        # Adding vocabulary aliases must not change semantic action probabilities.
        roles = np.where(np.arange(A)==self.eos_action,7,np.arange(A)%7)
        weights /= np.bincount(roles,minlength=8)[role]
        probs = weights/weights.sum(-1,keepdims=True)
        self.action_prob_table = jnp.asarray(probs,dtype=jnp.float32)
        self.next_milestone_table = jnp.asarray(nm,dtype=jnp.int32)
        self.next_quality_table = jnp.asarray(next_latent,dtype=jnp.int32)
        self.evaluator = self
        self.config = dict(version=self.VERSION,num_problems=P,num_milestones=M,max_steps=max_steps,
                           num_actions=A,num_approaches=num_approaches,error_rate=error_rate,
                           recovery_rate=recovery_rate,shortcut_weight=shortcut_weight,wait_weight=wait_weight,
                           gamma=gamma,seed=seed,eval_seed=eval_seed,variable_depth=variable_depth,
                           difficulties=theta.tolist(),problem_family=np.asarray(self.problem_family).tolist(),
                           preferred_approach=np.asarray(self.preferred_approach).tolist(),
                           latent_encoding="3 * approach + error_status",error_names=list(self.ERROR_NAMES),
                           action_roles=list(self.ACTION_NAMES)+["EOS"])
        self.V_table = self.solve_backward_induction()

    def _outcomes(self,p,t,m,q,a,xp=jnp):
        nm = xp.asarray(self.next_milestone_table)[p,m,q,a]
        nq = xp.asarray(self.next_quality_table)[p,m,q,a]
        done = (a==self.eos_action)|(t+1>=xp.asarray(self.problem_max_steps)[p])
        reward = (done & (nm>=xp.asarray(self.required_milestones)[p]) & (nq%3==self.SOUND)).astype(xp.float32)
        return nm,nq,reward,done

    def step(self,rng,state,action,params=None):
        obs,st,r,d,info = super().step(rng,state,action,params)
        info.update(approach=st.path_quality//3,error_status=st.path_quality%3)
        return obs,st,r,d,info

    def evaluation_groups(self,state):
        return (("approach",state.path_quality//3,self.num_approaches),
                ("error_status",state.path_quality%3,3))

    def diagnostics(self,trajectories=512,seed=2026):
        _,states = jax.vmap(self.reset)(jax.random.split(jax.random.PRNGKey(seed),trajectories))
        def scan_step(carry,_):
            st,key = carry
            key,ak,sk = jax.random.split(key,3)
            actions = jax.vmap(self.sample_base_action)(jax.random.split(ak,trajectories),st)
            _,ns,r,_,_ = jax.vmap(self.step)(jax.random.split(sk,trajectories),st,actions)
            p,m,q = st.prompt_idx,st.milestone,st.path_quality
            a = jnp.arange(self.num_actions)[None,:]
            nm,nq,ar,ad = self._outcomes(p[:,None],st.step_count[:,None],m[:,None],q[:,None],a)
            av = ar+self.gamma*(1-ad)*self.V_table[p[:,None],jnp.minimum(st.step_count[:,None]+1,self.max_steps),nm,nq]
            probs = jax.vmap(self.action_probabilities)(st)
            var = (probs*(av-(probs*av).sum(-1,keepdims=True))**2).sum(-1)
            return (ns,key),(~st.done,q,ns.path_quality,r,var)
        active,q,nq,r,var = map(np.asarray,jax.jit(lambda:jax.lax.scan(scan_step,(states,jax.random.PRNGKey(seed+1)),None,length=self.max_steps)[1])())
        err,nerr = q%3,nq%3
        faulty = active & (err!=self.SOUND)
        matrix = np.zeros((3,3),dtype=int)
        np.add.at(matrix,(err[active],nerr[active]),1)
        return dict(trajectories=trajectories,active_transitions=int(active.sum()),
                    success_rate=float(r.sum()/trajectories),
                    approach_occupancy=(np.bincount(q[active]//3,minlength=self.num_approaches)/active.sum()).tolist(),
                    error_occupancy=(np.bincount(err[active],minlength=3)/active.sum()).tolist(),
                    error_transition_counts=matrix.tolist(),
                    error_persistence=float(np.mean(err[faulty]==nerr[faulty])) if faulty.any() else None,
                    recovery_fraction=float(np.mean(nerr[faulty]==self.SOUND)) if faulty.any() else None,
                    approach_switch_fraction=float(np.mean(q[active]//3!=nq[active]//3)),
                    mean_successor_value_variance=float(var[active].mean()))
