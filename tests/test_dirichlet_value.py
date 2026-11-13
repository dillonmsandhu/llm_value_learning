import unittest
import jax
import jax.numpy as jnp
import numpy as np
from dirichlet_value import double_sampled_dirichlet_loss


class DirichletObjectiveTests(unittest.TestCase):
    def test_double_sampling_matches_exact_energy_and_gradient(self):
        # Root: half the time reaches a rewarding child, half terminal failure.
        P=jnp.array([[0.,.5],[0.,0.]])
        V=jnp.array([.5,1.])
        D=jnp.diag(jnp.array([1.,.5]))
        for gamma in (1.,.8):
            true=jnp.array([gamma*.5,1.])
            # Enumerate every independent MC-return / one-step-branch pair.
            # Six equally weighted entries implement occupancy [1,.5]/1.5.
            source=jnp.array([0,0,0,0,1,1])
            child=jnp.array([1,0,1,0,0,0])
            G=jnp.array([0.,0.,gamma,gamma,1.,1.])
            reward=jnp.array([0.,0.,0.,0.,1.,1.])
            done=jnp.array([False,True,False,True,True,True])
            def sampled(v):
                return double_sampled_dirichlet_loss(v[source],v[child],G,reward,done,jnp.ones(6,bool),gamma)
            def exact(v):
                e=true-v
                return e@D@(jnp.eye(2)-gamma*P)@e/1.5
            v=jnp.array([.2,.3])
            np.testing.assert_allclose(sampled(v),exact(v),atol=1e-7)
            np.testing.assert_allclose(jax.grad(sampled)(v),jax.grad(exact)(v),atol=1e-7)
            np.testing.assert_allclose(jax.grad(sampled)(true),0.,atol=1e-7)

    def test_reusing_trunk_successor_is_biased(self):
        # Same toy root: coupling its return to its branch creates covariance.
        v=jnp.array([.2,.3]);V=jnp.array([.5,1.])
        independent=(V[0]-v[0])*(.5*v[1]-v[0])
        coupled=double_sampled_dirichlet_loss(
            jnp.array([v[0],v[0]]),jnp.array([v[1],0.]),
            jnp.array([1.,0.]),jnp.zeros(2),jnp.array([False,True]),jnp.ones(2,bool))
        self.assertGreater(abs(float(coupled-independent)),.01)

    def test_terminal_padding_and_live_successor_gradients(self):
        def loss(value):
            return double_sampled_dirichlet_loss(
                jnp.array([.2,100.]),value,jnp.array([1.,999.]),
                jnp.array([0.,999.]),jnp.array([False,False]),jnp.array([True,False]))
        gradient=jax.grad(loss)(jnp.array([.3,100.]))
        self.assertAlmostEqual(float(gradient[0]),.4,places=6)
        self.assertEqual(float(gradient[1]),0.)
        # Terminal successors cannot contribute a value or a gradient.
        terminal=lambda v:double_sampled_dirichlet_loss(jnp.array([.2]),v,jnp.array([1.]),jnp.array([1.]),jnp.array([True]),jnp.array([True]))
        np.testing.assert_array_equal(jax.grad(terminal)(jnp.array([999.])),0.)

    def test_finite_episode_boundary_identity(self):
        # Flow occupancy, with terminal error fixed to zero.
        P=np.array([[0.,.5],[0.,0.]])
        d=np.array([1.,.5]);e=np.array([.3,.7])
        energy=e@np.diag(d)@(np.eye(2)-P)@e
        edge=.5*(e[0]-e[1])**2+.5*e[0]**2+.5*e[1]**2
        self.assertAlmostEqual(energy,.5*edge+.5*e[0]**2)


class DirichletBenchmarkTests(unittest.TestCase):
    def test_all_environment_families_and_costs(self):
        from run_benchmark import make_train_loop,build_dag_eval_set,build_tree_eval_set
        from envs.reasoning_dag_env import ReasoningDAGEnv
        from envs.quality_dag_env import QualityDAGEnv
        from envs.approach_error_dag_env import ApproachErrorDAGEnv
        from envs.tree_mrp import ProceduralTreeEnv
        cfg=dict(NUM_ENVS=2,NUM_STEPS=4,NUM_UPDATES=2,GRAD_STEPS_PER_UPDATE=2,
                 MINIBATCH_SIZE=4,D_MODEL=8,N_HEADS=2,N_LAYERS=1,FINAL_HIDDEN_DIM=8)
        for cls in (ReasoningDAGEnv,QualityDAGEnv,ApproachErrorDAGEnv,ProceduralTreeEnv):
            tree=cls is ProceduralTreeEnv
            env=cls(num_problems=1,depth=4) if tree else cls(num_problems=1,num_milestones=2,max_steps=4)
            ev=build_tree_eval_set(env,samples_per_problem=4) if tree else build_dag_eval_set(env,samples_per_problem=4)
            fn=make_train_loop('dirichlet','tree' if tree else 'dag',env,ev,cfg)
            out=jax.device_get(jax.jit(fn)(jax.random.PRNGKey(42)))
            self.assertTrue(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(out)))
            self.assertIn('dirichlet_loss',out['metrics'])
            # Parent and successor both receive gradient-bearing evaluations.
            np.testing.assert_array_equal(out['metrics']['trained_steps'],16.)
            # Additional detached trunk-successor inference in the bilinear surrogate.
            np.testing.assert_array_equal(out['metrics']['critic_eval_steps'],8.)
            self.assertTrue(np.all(out['metrics']['active_infer_steps']>0))


if __name__=='__main__': unittest.main()
