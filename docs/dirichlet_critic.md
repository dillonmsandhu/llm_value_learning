# E_min: bilinear double-sampled critic

The fixed-policy benchmark (`--algos dirichlet`, or alias `e_min`) and PPO
(`--critic_method e_min`) use the same `bilinear_double_sampled_loss` in
`dirichlet_value.py`. It matches `double_sample_e_inner_critic_loss_fn` in the
E_min Gymnax and Brax implementations, apart from external loss multipliers such
as Gymnax's `VF_COEF`. No auxiliary prediction head is needed.

## Objective and exact gradient estimator

For a frozen policy, define `e = V^pi - v` and
`bar_delta = E[r + gamma*v(s_next) | s] - v(s) = (I-gamma*P)e`.
The objective convention is `0.5 * <e, bar_delta>_D`.

The collected trunk supplies its complete suffix return `G`, reward `r1`, and
successor `s1`. One independent action/transition from the same prefix supplies
`r2, s2`. Both actions follow the same frozen policy. Define:

```
e_hat = G - v(s)
delta1 = r1 + gamma*(1-done1)*v(s1) - v(s)
delta2 = r2 + gamma*(1-done2)*v(s2) - v(s)
delta_bar = (delta1 + delta2)/2
```

The implemented gradient surrogate is:

```
L = -0.5 * stop_gradient(delta_bar + e_hat) * v(s)
    +0.5 * gamma * stop_gradient(e_hat) * (1-done2) * v(s2)
```

The first coefficient is allowed to reuse the trunk transition: `grad(v(s))`
depends only on the conditioned prefix. The successor gradient is multiplied by
the trunk error **only on the independent probe**. Using the trunk-inclusive
`delta_bar` in a naive product `e_hat * delta_bar` would generally be biased.
The surrogate scalar is not itself a numerical estimate of the energy.

For any fixed critic, unbiased full returns and conditional independence give
an unbiased stochastic gradient of the stated objective. The original independent
product remains as `double_sampled_dirichlet_loss` for reference/diagnostic tests;
it is no longer the training loss. Earlier results using that product have a
different estimator and loss scale; metadata now records estimator version
`bilinear_double_sampled_trunk_probe_v1`.

## Variance scope

Two independent TD samples halve the conditional variance of `delta_bar`
relative to a single TD sample. This does **not** guarantee lower total gradient
variance: `e_hat` and `delta1` share the trunk. For the current-state coefficient,
the variance change relative to `delta2 + e_hat` is
`-Var(delta)/2 + Cov(e_hat, delta1)`. Exact-enumeration tests cover both the TD
variance reduction and a counterexample to a universal whole-gradient claim.

## Boundaries, weighting, and optimization

Both DAG paths collect full episodes, use actual MC suffix returns, and fix
terminal continuation values to zero. The fixed-policy benchmark rejects an
E_min rollout horizon shorter than `env.max_steps`. The Gymnax/Brax loss-function
match does not by itself certify their rollout targets: unbiasedness for the true
energy also requires unbiased returns; truncated bootstrapped targets generally
do not meet that assumption.

The state weighting uses expected nonterminal occupancy divided by the fixed
maximum number of rollout slots. Post-terminal and minibatch padding have zero
weight; the loss is never divided by the random active count. PPO compensates
for extra fixed minibatch padding so its objective retains the same fixed-slot
normalization.

E_min bypasses critic value clipping and adds no MSE or anchor penalty. Gradients
are estimated before the existing shared optimizer and global gradient clipping;
Adam, clipping, and repeated fitting of a finite sampled batch do not inherit a
guarantee of unbiased optimization updates or population descent.

PPO actor data, GAE, clipping, learning rate, and epochs follow the shared comparison
configuration. Probes never enter the actor minibatches. There is no actor entropy
bonus, KL penalty, or KL stopping added by this method.

## Costs and outputs

Each active trunk prefix adds exactly one active probe transition, the same
sampling footprint as trunk-included FBI K=2. The critic-only compute estimate
counts two gradient-bearing value passes (current and probe successor) plus one
detached value pass (trunk successor) per optimization slot. Standard value,
advantage-alignment, and budget metrics remain available. `critic_loss` in PPO and
`dirichlet_loss` in the fixed-policy benchmark record the signed surrogate.

The PPO comparison now includes six methods and both comparison plotters support
E_min. Existing five-method archives remain readable.

## Cluster commands

```sh
# All six PPO methods, using the tuned shared actor settings:
bash scripts/submit_ppo_comparison.sh dag_approach results/ppo_with_emin

# Only the new method, matching those comparison defaults:
bash scripts/submit_dag_ppo.sh dag_approach e_min \
  --actor_lr 0.00003 --clip_eps 0.2 --entropy_coef 0 --actor_epochs 4 \
  --actor_lambda 1 --bc_updates 100 --bc_lr 0.0003 \
  --num_problems 20 --num_milestones 12 --max_steps 40 --num_envs 64 \
  --d_model 64 --n_layers 2 --eval_episodes 1024 --n_seeds 5 \
  --save_dir results/ppo_with_emin/e_min

# Plot when completed:
python3 plot_ppo_comparison.py --results_dir results/ppo_with_emin
```

`DRY_RUN=1` is available on the six-method launcher. Verification for this update
uses exact finite-MRP gradient enumeration, masking/variance checks, loss-function
audits against the actual Gymnax/Brax source, and abstract tracing of trainer
wiring. No local policy training or cluster submission is needed for these checks.
