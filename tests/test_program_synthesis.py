"""Environment, importer, branching and abstract-wiring checks; no training runs."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from envs.program_tasks import compile_task,polynomial_op,task_split
from envs.program_synthesis import ProgramSynthesisEnv


def equal_states(test,a,b):
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(x,y)


class TaskCompilerTests(unittest.TestCase):
    def test_exact_polynomials_and_reusable_subexpressions(self):
        task=compile_task('(x+2)*(x+2)',degree=3,max_ops=3,answer='x*x+4*x+4')
        self.assertEqual(task['target'],[4,4,1,0])
        self.assertEqual(task['reference_ops'],2)
        self.assertEqual(task['plan'][-1][1],task['plan'][-1][2])
        with self.assertRaisesRegex(ValueError,'provided answer'):
            compile_task('2+3',answer='6')
        for expression in ('1/2','x*x*x*x','10000','__import__("os")','x**100'):
            with self.assertRaises((ValueError,SyntaxError)):
                compile_task(expression,degree=3)

    def test_importer_validates_gold_and_reports_rejections(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'arithmetic__mixed.txt';out=Path(tmp)/'tasks.jsonl'
            source.write_text('Calculate (2+3)*4.\n20\nExpand (z+2)*(z-3).\nz*z-z-6\nCalculate 2+3.\n99\nWhat is seven halves?\n3.5\n')
            subprocess.run([sys.executable,'scripts/prepare_program_tasks.py','--input',str(source),'--output',str(out)],cwd=root,check=True,stdout=subprocess.DEVNULL)
            rows=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows),2)
            report=json.loads(Path(str(out)+'.report.json').read_text())
            self.assertEqual(report['rejected'],2)
            self.assertTrue(all(r['source'].startswith('DeepMind') for r in rows))


class ProgramEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=ProgramSynthesisEnv(num_problems=4,max_steps=8,program_min_ops=2,program_max_ops=3,
                                    program_degree=2,program_max_inputs=4,program_coefficient_limit=64)

    def test_reference_programs_solve_and_terminal_is_absorbing(self):
        e=self.env
        @jax.jit
        def solve(index):
            _,st=e.reset_problem(index)
            def step(st,_):
                _,ns,r,_,_=e.step(jax.random.PRNGKey(0),st,e.reference_action(st))
                return ns,r
            return jax.lax.scan(step,st,None,length=e.max_steps)
        for i in range(e.num_problems):
            st,reward=solve(i)
            self.assertEqual(float(reward.sum()),1.)
            self.assertTrue(bool(st.done))
            _,again,r,_,_=e.step(jax.random.PRNGKey(5),st,jnp.array(e.reset_action))
            equal_states(self,st,again);self.assertEqual(float(r),0.)

    def test_exact_integer_transition_arithmetic_and_bounds(self):
        e=self.env;_,st=e.reset_problem(0)
        # Set a public fixture ledger, preserving the functional state contract.
        registers=jnp.zeros_like(st.registers).at[0].set(jnp.array([3,2,0])).at[1].set(jnp.array([-1,1,0]))
        st=st._replace(registers=registers,register_count=jnp.array(2,jnp.int32))
        for op in (0,1,2):
            _,next_state,r,d,_=e.step(jax.random.PRNGKey(0),st,jnp.array(e.encode_binary(op,0,1)))
            expected=polynomial_op(op,[3,2,0],[-1,1,0],2,64)
            np.testing.assert_array_equal(next_state.registers[2],expected)
            self.assertEqual(next_state.registers.dtype,jnp.int32)
            self.assertEqual(float(r),0.)
        # Non-exact division and references to unavailable registers are no-ops.
        for action in (e.encode_binary(3,0,1),e.encode_binary(0,e.capacity-1,0)):
            _,ns,r,d,info=e.step(jax.random.PRNGKey(0),st,jnp.array(action))
            np.testing.assert_array_equal(ns.registers,st.registers)
            self.assertEqual(int(ns.step_count),int(st.step_count)+1)
            self.assertTrue(bool(info['invalid_action']))

    def test_undo_reset_and_only_submitted_answer_earns_reward(self):
        e=self.env;_,st=e.reset_problem(0)
        a=e.encode_binary(0,0,0)
        _,computed,_,_,_=e.step(jax.random.PRNGKey(0),st,jnp.array(a))
        self.assertEqual(int(computed.register_count),int(st.register_count)+1)
        _,undone,_,_,_=e.step(jax.random.PRNGKey(0),computed,jnp.array(e.undo_action))
        np.testing.assert_array_equal(undone.registers,st.registers)
        self.assertEqual(int(undone.step_count),2)
        _,reset,_,_,_=e.step(jax.random.PRNGKey(0),computed,jnp.array(e.reset_action))
        np.testing.assert_array_equal(reset.registers,st.registers)
        # Answer validity, not agreement with a reference program, defines reward.
        changed=st._replace(registers=st.registers.at[0].set(e.targets[0]))
        _,_,reward,done,_=e.step(jax.random.PRNGKey(0),changed,jnp.array(e.submit_offset))
        self.assertEqual(float(reward),1.);self.assertTrue(bool(done))

    def test_independent_forks_and_no_hidden_target_observation(self):
        e=self.env;_,st=e.reset_problem(0)
        before=copy.deepcopy(st)
        _,left,_,_,_=e.step(jax.random.PRNGKey(0),st,jnp.array(e.encode_binary(0,0,0)))
        _,right,_,_,_=e.step(jax.random.PRNGKey(0),st,jnp.array(e.encode_binary(1,0,0)))
        equal_states(self,st,before)
        self.assertFalse(np.array_equal(left.registers,right.registers))
        other=copy.copy(e);other.targets=e.targets+1
        obs,_=other.reset_problem(0)
        np.testing.assert_array_equal(obs,st.history_tokens)
        # Changing a hidden target cannot change a nonterminal observation.
        obs1=e.step(jax.random.PRNGKey(0),st,jnp.array(e.encode_binary(0,0,0)))[0]
        obs2=other.step(jax.random.PRNGKey(0),st,jnp.array(e.encode_binary(0,0,0)))[0]
        np.testing.assert_array_equal(obs1,obs2)

    def test_disjoint_reproducible_problem_banks_and_shape_contract(self):
        e=self.env
        kwargs=dict(num_problems=4,max_steps=8,program_min_ops=2,program_max_ops=3,
                    program_degree=2,program_max_inputs=4,program_coefficient_limit=64)
        same=ProgramSynthesisEnv(**kwargs);heldout=ProgramSynthesisEnv(**kwargs,split='test')
        self.assertEqual(e.config['task_bank_hash'],same.config['task_bank_hash'])
        self.assertFalse(set(e.config['fingerprints'])&set(heldout.config['fingerprints']))
        self.assertEqual((e.max_seq_len,e.vocab_size,e.num_actions),(heldout.max_seq_len,heldout.vocab_size,heldout.num_actions))
        self.assertTrue(all(task_split(f)=='test' for f in heldout.config['fingerprints']))
        obs,states=jax.vmap(e.reset)(jax.random.split(jax.random.PRNGKey(0),3))
        self.assertEqual(obs.shape,(3,e.max_seq_len))
        self.assertTrue(np.all(np.asarray(obs)>=0));self.assertTrue(np.all(np.asarray(obs)<e.vocab_size))


class ProgramIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(ProgramEnvironmentTests,'env'):ProgramEnvironmentTests.setUpClass()

    def test_branch_batch_size_preserves_samples(self):
        from vine_mc import one_step_branches,prefix_values
        e=ProgramEnvironmentTests.env
        _,states=jax.vmap(e.reset)(jax.random.split(jax.random.PRNGKey(0),2))
        states=jax.tree_util.tree_map(lambda x:x[None],states)
        class ConstantActor:
            def apply(self,params,obs):return jnp.zeros((len(obs),e.num_actions))
        actor=ConstantActor();key=jax.random.PRNGKey(2)
        for function in (one_step_branches,prefix_values):
            def evaluate(size):
                args=(e,actor,None,states,key,2,size)
                return jax.jit(lambda:function(*args,1.) if function is prefix_values else function(*args))()
            a=evaluate(2);b=evaluate(3)
            for aa,bb in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
                np.testing.assert_array_equal(aa,bb)

    def test_all_critic_methods_trace_without_training(self):
        from envs.dag_mdp import DAGMDP
        from train_dag_ppo import make_train
        from program_evaluation import build_program_eval_set
        from run_benchmark import make_train_loop
        e=ProgramEnvironmentTests.env;mdp=DAGMDP(e)
        cfg=dict(num_updates=1,num_envs=2,actor_epochs=1,critic_epochs=1,minibatch_size=3,
                 eval_episodes=2,d_model=8,n_heads=2,n_layers=1,final_hidden_dim=8,vine_batch_size=2)
        for method in ('mc','td','fbi','vine','vine_mc','e_min'):
            shapes=jax.eval_shape(make_train(mdp,dict(cfg,critic_method=method,k=2),evaluation_env=mdp),jax.random.PRNGKey(1))
            self.assertEqual(shapes['metrics']['success_rate'].shape,(1,))
            self.assertEqual(shapes['final_heldout_success'].shape,())
        ev=build_program_eval_set(e,samples_per_problem=1,reference_k=2,action_samples=2)
        self.assertFalse(ev['reference_is_exact'])
        bc=dict(NUM_ENVS=2,NUM_STEPS=e.max_steps,NUM_UPDATES=2,GRAD_STEPS_PER_UPDATE=2,
                MINIBATCH_SIZE=2,D_MODEL=8,N_HEADS=2,N_LAYERS=1,FINAL_HIDDEN_DIM=8)
        for method in ('mc','dirichlet','td_lam0.8','vine_k2','fbi_trunk_k2_lam0.8','fbi_trunk_k1_lam0.8','fbi_staged_lam0.8'):
            shapes=jax.eval_shape(make_train_loop(method,'program',e,ev,bc),jax.random.PRNGKey(1))
            self.assertEqual(shapes['metrics']['msve'].shape,(2,))


if __name__=='__main__':unittest.main()
