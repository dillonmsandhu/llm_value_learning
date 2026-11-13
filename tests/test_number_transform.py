"""Arithmetic/search and abstract trainer checks. No policy training."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from envs.number_transform import NumberTransformEnv,search_graph,test_membership as is_test_pair


class NumberTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kwargs=dict(number_max_value=99,number_min_distance=2,number_max_distance=3,max_steps=4)
        cls.env=NumberTransformEnv(**kwargs);cls.test=NumberTransformEnv(**kwargs,split='test')

    def example(self):
        for e in (self.env,self.test):
            index=np.flatnonzero((np.asarray(e.starts)==63)&(np.asarray(e.targets)==4))
            if len(index):return e,e.reset_problem(index[0])[1]
        self.fail('63->4 must exist in one partition')

    def test_deadline_no_stop_and_no_early_success(self):
        e,st=self.example();rewards=[]
        for a in (35,11,18,18):  # /9, -3, x1, x1
            _,st,r,d,_=e.step(None,st,jnp.array(a));rewards.append(float(r))
            if int(st.step_count)<4:self.assertFalse(bool(d))
        self.assertEqual(rewards,[0.,0.,0.,1.]);self.assertEqual(int(st.current_value),4)
        before=st
        _,after,r,_,_=e.step(None,st,jnp.array(0))
        for x,y in zip(jax.tree_util.tree_leaves(before),jax.tree_util.tree_leaves(after)):np.testing.assert_array_equal(x,y)
        self.assertEqual(float(r),0.)
        e,st=self.example()
        for a in (35,11,18,0):_,st,r,_,_=e.step(None,st,jnp.array(a))  # Reach 4, but finish at 5.
        self.assertEqual(float(r),0.)

    def test_exact_division_invalid_bounds_and_signed_arithmetic(self):
        e,st=self.example();_,st,_,_,_=e.step(None,st,jnp.array(35))
        _,bad,_,_,info=e.step(None,st,jnp.array(28))  # 7 / 2 is not integer division.
        self.assertEqual(int(bad.current_value),7);self.assertTrue(bool(info['invalid_action']))
        self.assertEqual(int(bad.step_count),2)
        _,bad,_,_,info=e.step(None,st,jnp.array(26))  # 7*9=63 is valid here.
        self.assertEqual(int(bad.current_value),63);self.assertFalse(bool(info['invalid_action']))
        following,valid,_=search_graph(-50,50)
        self.assertTrue(valid[-6+50,28]);self.assertEqual(following[-6+50,28],-3)
        self.assertFalse(valid[-7+50,28]);self.assertEqual(following[-7+50,28],-7)
        self.assertFalse(valid[50+50,0]);self.assertEqual(following[50+50,0],50)

    def test_search_distances_and_reference_padding(self):
        e,initial=self.example()
        self.assertEqual(int(e.distances[63,4]),2)
        @jax.jit
        def demonstrate(index):
            _,state=e.reset_problem(index)
            def step(st,_):
                _,ns,r,_,_=e.step(None,st,e.reference_action(st))
                return ns,r
            return jax.lax.scan(step,state,None,length=e.max_steps)
        for index in e.evaluation_problem_ids(12,42):
            _,rewards=demonstrate(index)
            self.assertEqual(float(rewards.sum()),1.)
        # Bellman optimal-distance identity for every non-goal bounded state.
        following,valid,distance=search_graph(0,99)
        for goal in (0,4,57,99):
            expected=np.min(np.where(valid,distance[following,goal]+1,65535),axis=1)
            non_goal=np.arange(100)!=goal
            np.testing.assert_array_equal(distance[non_goal,goal],expected[non_goal])

    def test_digit_vocabulary_and_large_disjoint_pair_pools(self):
        e=self.env
        self.assertEqual((e.num_actions,e.vocab_size),(36,53))
        self.assertGreater(e.num_problems,1000)
        self.assertTrue(np.all(~is_test_pair(np.asarray(e.starts),np.asarray(e.targets),'pair')))
        self.assertTrue(np.all(is_test_pair(np.asarray(self.test.starts),np.asarray(self.test.targets),'pair')))
        np.testing.assert_array_equal(e.encode_number(jnp.array(63))[-2:]-e.digit_offset,[6,3])
        held=NumberTransformEnv(number_max_value=99,number_max_distance=3,max_steps=4,split='test',number_split_mode='target')
        train=NumberTransformEnv(number_max_value=99,number_max_distance=3,max_steps=4,number_split_mode='target')
        self.assertFalse(set(np.asarray(held.targets))&set(np.asarray(train.targets)))
        self.assertEqual(e.max_seq_len,self.test.max_seq_len)

    def test_all_critic_modes_and_stratified_evaluations_trace(self):
        from envs.dag_mdp import DAGMDP
        from train_dag_ppo import make_train
        from program_evaluation import build_program_eval_set
        from run_benchmark import make_train_loop
        cfg=dict(num_updates=1,num_envs=2,actor_epochs=1,critic_epochs=1,minibatch_size=3,eval_episodes=4,
                 d_model=8,n_heads=2,n_layers=1,final_hidden_dim=8,vine_batch_size=2)
        for method in ('mc','td','fbi','vine','vine_mc','e_min'):
            shape=jax.eval_shape(make_train(DAGMDP(self.env),dict(cfg,critic_method=method,k=2),DAGMDP(self.test)),jax.random.PRNGKey(0))
            self.assertIn('final_heldout',shape['evaluation_by_distance'])
        ev=build_program_eval_set(self.env,samples_per_problem=1,reference_k=2,action_samples=2,max_eval_problems=4)
        self.assertEqual(ev['num_problems'],4);self.assertFalse(ev['reference_is_exact'])
        bc=dict(NUM_ENVS=2,NUM_STEPS=4,NUM_UPDATES=2,GRAD_STEPS_PER_UPDATE=2,MINIBATCH_SIZE=2,
                D_MODEL=8,N_HEADS=2,N_LAYERS=1,FINAL_HIDDEN_DIM=8)
        for method in ('mc','dirichlet','td_lam0.8','vine_k2','fbi_trunk_k2_lam0.8'):
            shape=jax.eval_shape(make_train_loop(method,'numbers',self.env,ev,bc),jax.random.PRNGKey(0))
            self.assertEqual(shape['metrics']['msve'].shape,(2,))

    def test_comparison_preview_has_36_action_family_and_fixed_budget(self):
        env=dict(os.environ,DRY_RUN='1')
        output=subprocess.check_output(['bash','scripts/submit_number_comparison.sh','results/no_submit'],env=env,text=True)
        lines=[line for line in output.splitlines() if line.startswith('bash ')]
        self.assertEqual(len(lines),6)
        self.assertTrue(all(' numbers ' in line and '--max_steps 8 ' in line for line in lines))


if __name__=='__main__':unittest.main()
