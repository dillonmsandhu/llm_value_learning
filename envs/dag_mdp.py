"""Controlled DAG interfaces: the trainer, rather than the MRP sampler, chooses actions."""
from .reasoning_dag_env import ReasoningDAGEnv
from .quality_dag_env import QualityDAGEnv
from .approach_error_dag_env import ApproachErrorDAGEnv
from .program_synthesis import ProgramSynthesisEnv


class DAGMDP:
    """Preserve DAG dynamics while separating the fixed reference policy from control.

    No auto-reset: EOS and the task horizon are true terminals. The trainer resets
    complete episodes and masks padded transitions. Privileged state is available
    for diagnostics/branching, never passed to the actor or critic networks.
    """
    def __init__(self, base_env):
        self.base_env = base_env
        self.num_actions = base_env.num_actions
        self.vocab_size = base_env.vocab_size
        self.max_seq_len = base_env.max_seq_len
        self.max_steps = base_env.max_steps
        self.num_problems = base_env.num_problems
        self.eos_action = base_env.eos_action

    def reset(self, rng):
        return self.base_env.reset(rng)

    def step(self, rng, state, action):
        obs, state, reward, done, info = self.base_env.step(rng, state, action)
        # This is V of the original fixed policy, not of the learned policy.
        info = {k:v for k,v in info.items() if k != "exact_v"}
        return obs, state, reward, done, info

    def sample_reference_action(self, rng, state):
        """Optional behavior-cloning teacher only; not used for PPO rollouts."""
        return self.base_env.sample_base_action(rng, state)


def make_dag_mdp(env_type="dag_approach", **kwargs):
    classes = {"dag":ReasoningDAGEnv, "dag_quality":QualityDAGEnv,
               "dag_approach":ApproachErrorDAGEnv,"program":ProgramSynthesisEnv}
    if env_type not in classes:
        raise ValueError(f"Unsupported DAG MDP: {env_type}")
    return DAGMDP(classes[env_type](**kwargs))
