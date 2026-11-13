"""Exact-enumeration/gradient tests. No policy optimization or training runs."""
import ast
import os
from pathlib import Path
import shlex
import subprocess
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from dirichlet_value import bilinear_double_sampled_loss, double_sampled_dirichlet_loss


class BilinearEstimatorTests(unittest.TestCase):
    def test_exact_gradient_for_independent_trunk_probe_pairs(self):
        # Root goes to a rewarding child or terminal failure, equally likely.
        # Root occupancy 1, child occupancy 1/2; enumerate independent pairs.
        source=jnp.array([0,0,0,0,1,1])
        child_1=jnp.array([1,1,0,0,0,0])
        child_2=jnp.array([1,0,1,0,0,0])
        r1=jnp.array([0.,0.,0.,0.,1.,1.])
        r2=r1
        d1=jnp.array([False,False,True,True,True,True])
        d2=jnp.array([False,True,False,True,True,True])
        P=jnp.array([[0.,.5],[0.,0.]])
        D=jnp.diag(jnp.array([1.,.5]))
        for gamma in (1.,.8):
            G=jnp.array([gamma,gamma,0.,0.,1.,1.])
            truth=jnp.array([gamma*.5,1.])
            def surrogate(v):
                return bilinear_double_sampled_loss(v[source],v[child_1],v[child_2],G,
                                                     r1,d1,r2,d2,jnp.ones(6,bool),gamma)
            def exact(v):
                e=truth-v
                return .5*e@D@(jnp.eye(2)-gamma*P)@e/1.5
            for v in (jnp.array([.2,.3]),jnp.array([-.2,1.7]),truth):
                np.testing.assert_allclose(jax.grad(surrogate)(v),jax.grad(exact)(v),atol=1e-7)
            np.testing.assert_allclose(jax.grad(surrogate)(truth),0.,atol=1e-7)

    def test_terminal_masks_and_stop_gradient_placement(self):
        # Grad(v1) is zero; grad(v2) must carry the independent MC-error term.
        def loss(values,active=True,terminal=False):
            v,v1,v2=values
            return bilinear_double_sampled_loss(v[None],v1[None],v2[None],
                jnp.array([1.]),jnp.array([0.]),jnp.array([False]),
                jnp.array([0.]),jnp.array([terminal]),jnp.array([active]),1.)
        v=jnp.array([.2,.4,.6])
        grad=jax.grad(loss)(v)
        np.testing.assert_allclose(grad,[-.55,0.,.4],atol=1e-7)
        np.testing.assert_array_equal(jax.grad(lambda v:loss(v,active=False))(v),0.)
        self.assertEqual(float(jax.grad(lambda v:loss(v,terminal=True))(v)[2]),0.)

    def test_fixed_padding_normalization(self):
        def loss(v,mask):
            return bilinear_double_sampled_loss(v,jnp.array([.4,.4]),jnp.array([.6,.6]),
                jnp.ones(2),jnp.zeros(2),jnp.zeros(2,bool),jnp.zeros(2),jnp.zeros(2,bool),mask)
        v=jnp.array([.2,.2])
        full=jax.grad(lambda v:loss(v,jnp.array([True,True])))(v)
        masked=jax.grad(lambda v:loss(v,jnp.array([True,False])))(v)
        # Masking a second slot must not renormalize the first slot's gradient.
        self.assertEqual(float(full[0]),float(masked[0]))
        self.assertEqual(float(masked[1]),0.)

    def test_td_coefficient_variance_and_full_gradient_covariance(self):
        b1=jnp.array([0.,0.,1.,1.]);b2=jnp.array([0.,1.,0.,1.])
        delta1=.3*b1-.2;delta2=.3*b2-.2
        np.testing.assert_allclose(jnp.var((delta1+delta2)/2),jnp.var(delta2)/2,atol=1e-8)
        # Same legitimate MRP provides a counterexample to a blanket claim of
        # lower *whole-gradient* variance: Cov(G,delta_trunk) can dominate.
        def one(v,a,b,averaged):
            v0,v1=v
            if averaged:
                return bilinear_double_sampled_loss(v0[None],v1[None],v1[None],a[None],
                    jnp.zeros(1),jnp.array([a==0]),jnp.zeros(1),jnp.array([b==0]),jnp.ones(1,bool))
            return .5*double_sampled_dirichlet_loss(v0[None],v1[None],a[None],
                    jnp.zeros(1),jnp.array([b==0]),jnp.ones(1,bool))
        v=jnp.array([.2,.3])
        averaged=jax.vmap(lambda a,b:jax.grad(lambda v:one(v,a,b,True))(v))(b1,b2)
        product=jax.vmap(lambda a,b:jax.grad(lambda v:one(v,a,b,False))(v))(b1,b2)
        np.testing.assert_allclose(averaged.mean(0),product.mean(0),atol=1e-7)
        self.assertGreater(float(jnp.var(averaged,axis=0).sum()),float(jnp.var(product,axis=0).sum()))

    def test_coupled_successor_bias_is_not_reintroduced(self):
        # A naive product with a trunk-inclusive delta mean is biased.
        a=jnp.array([0.,1.]);v=jnp.array([.2,.3])
        naive=lambda v:.5*jnp.mean((a-v[0])*(v[1]*a-v[0]))
        true=lambda v:.5*(.5-v[0])*(.5*v[1]-v[0])
        self.assertGreater(float(jnp.linalg.norm(jax.grad(naive)(v)-jax.grad(true)(v))),.01)


class PPOWiringTests(unittest.TestCase):
    def test_comparison_submission_has_six_matched_actor_configs(self):
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,DRY_RUN='1')
        output=subprocess.check_output(['bash','scripts/submit_ppo_comparison.sh'],cwd=root,env=env,text=True)
        commands=[shlex.split(x) for x in output.splitlines() if x.startswith('bash ')]
        self.assertEqual(len(commands),6)
        self.assertEqual(sum(x[3]=='e_min' for x in commands),1)
        flags=[dict(zip(x[4::2],x[5::2])) for x in commands]
        for key in ('--actor_lr','--actor_epochs','--actor_lambda','--clip_eps','--entropy_coef','--bc_lr'):
            self.assertEqual(len({f[key] for f in flags}),1)

    def test_ppo_and_fixed_policy_paths_trace_with_shared_estimator(self):
        # Abstract tracing catches shape/wiring issues without running optimizer
        # updates. The rollout and critic graphs are constructed, not executed.
        from train_dag_ppo import make_train
        from envs.dag_mdp import make_dag_mdp
        from envs.approach_error_dag_env import ApproachErrorDAGEnv
        from run_benchmark import make_train_loop,build_dag_eval_set
        cfg=dict(num_updates=1,num_envs=2,actor_epochs=1,critic_epochs=1,minibatch_size=5,
                 eval_episodes=2,d_model=8,n_heads=2,n_layers=1,final_hidden_dim=8)
        env=make_dag_mdp('dag_approach',num_problems=1,num_milestones=2,max_steps=4)
        for method in ('e_min','mc','td','fbi','vine','vine_mc'):
            abstract=jax.eval_shape(make_train(env,dict(cfg,critic_method=method,k=2)),jax.random.PRNGKey(0))
            self.assertEqual(abstract['metrics']['critic_loss'].shape,(1,))
        base=ApproachErrorDAGEnv(num_problems=1,num_milestones=2,max_steps=4)
        ev=build_dag_eval_set(base,samples_per_problem=2)
        bc=dict(NUM_ENVS=2,NUM_STEPS=4,NUM_UPDATES=2,GRAD_STEPS_PER_UPDATE=2,
                MINIBATCH_SIZE=4,D_MODEL=8,N_HEADS=2,N_LAYERS=1,FINAL_HIDDEN_DIM=8)
        abstract=jax.eval_shape(make_train_loop('dirichlet','dag',base,ev,bc),jax.random.PRNGKey(0))
        self.assertEqual(abstract['metrics']['dirichlet_loss'].shape,(2,))
        with self.assertRaisesRegex(ValueError,'complete trunk'):
            make_train_loop('e_min','dag',base,ev,dict(bc,NUM_STEPS=2))


if __name__=='__main__':unittest.main()
