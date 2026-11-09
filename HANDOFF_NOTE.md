# Project Handoff Note: Step-Level Value Estimation (`VinePPO` Testbed)

Open `/Users/dillonsandhu/Documents/Research/VinePPO` directly as your workspace root so all files and JAX scripts run inside the workspace without cross-directory symlinks.

---

## 1. Overview & Proposal (`fbi_ppo.tex` / `/Users/dillonsandhu/Documents/Research/rl_ms_project/docs/fbi_ppo.pdf`)

We are investigating scalable step-level value function estimation for LLM reasoning (modeled in a tabular/transformer testbed via `TreeMRP`), focusing strictly on **critic estimation** (used for base-trajectory PPO advantages, without training the actor on auxiliary branches):

1. **Variant 1: 1-Step Fitted Backward Induction ($\text{FBI}(\lambda)$, $K' = 1$)**
   - **Cost:** $\mathcal{O}(K \cdot T)$ auxiliary transitions per trajectory ($K$ 1-step branches per visited state).
   - **Anchoring ($t = T - 1$):** Samples $K$ 1-step completions to leaves and averages their exact rewards: $y(s_{T-1}) = \hat{V}_{\text{MC}}(s_{T-1}) = \frac{1}{K}\sum_{k=1}^K R^{(k)}$. Also supervises $V_\phi$ directly on all $K$ sampled leaf states.
   - **Backward Induction ($t < T - 1$):** Computes 1-step branched expectation $\hat{V}_{\text{branch}}(s_t) = \frac{1}{K}\sum_{k=1}^K \text{sg}(V_\phi(s_{t+1}^{(k)}))$ using the *just-updated* critic parameters from level $t + 1$, blended with downstream trajectory targets via $\lambda \in [0, 1]$:
     $$y_t^{(\lambda)} = (1 - \lambda)\,\hat{V}_{\text{branch}}(s_t) + \lambda\,\gamma\,y_{t+1}^{(\lambda)}.$$
   - **Interleaved Fitting & Replay Anchoring:** At each backward level $t$, we append $(s_t, y_t^{(\lambda)})$ to a cumulative buffer $\mathcal{D}_{\text{fit}}$ (retaining deeper levels $t' > t$) and run `STAGE_STEPS` inner gradient updates so $V_\phi$ fits level $t$ before level $t - 1$ queries it, without overwriting terminal anchors.

2. **Variant 2: Fitted Vine Interpolation & Subsampling ($V_\phi \leftrightarrow \hat{V}_{\text{MC}}$)**
   - **Direct Distillation:** Fits $V_\phi(s_t)$ directly to unbiased $K$-rollout Monte Carlo estimates $\hat{V}_{\text{MC}}(s_t) \in [0, 1]$ (zero multi-step bootstrapping drift).
   - **Online Out-of-Sample Accuracy Audit:** Before updating $V_\phi$ on each batch, measures excess prediction error variance $\sigma^2_{\text{critic}}$ via an EMA of $\max\!\big(0, (V_\phi(s_t) - \hat{V}_{\text{MC}}(s_t))^2 - \hat{\sigma}^2_{\text{MC}}(s_t)\big)$, where $\hat{\sigma}^2_{\text{MC}}(s_t) = \frac{\hat{V}_{\text{MC}}(s_t)(1 - \hat{V}_{\text{MC}}(s_t))}{K - 1}$.
   - **Precision-Weighted Shrinkage:** Combines noisy small-$K$ Monte Carlo targets with $V_\phi$:
     $$\hat{V}_{\text{interp}}(s_t) = (1 - \alpha_t)\,\hat{V}_{\text{MC}}(s_t) + \alpha_t\,V_\phi(s_t), \qquad \alpha_t = \frac{\hat{\sigma}^2_{\text{MC}}(s_t)}{\hat{\sigma}^2_{\text{MC}}(s_t) + \sigma^2_{\text{critic}} + \epsilon}.$$

---

## 2. Key Files in This Folder (`/Users/dillonsandhu/Documents/Research/VinePPO`)

- `envs/tree_mrp.py` & `envs/tabular_matrix_env.py`: Binary tree Markov Reward Process (`TreeMRP` and exact solver `TreeMRPExactValue.solve_backward_induction()`).
- `transformer.py`: `TransformerCritic` (LLaMA-style pre-RMSNorm + RoPE + SwiGLU + MLP value head).
- `fbi_value.py`: Pure-JAX implementation of **Variant 1 ($\text{FBI}(\lambda)$)**:
  - Supports `--lmbda` ($\lambda \in [0, 1]$), `--stage-steps` (default `4`), `--k-branches` (default `4`), and `--lr` (default `1e-3`).
  - Logs `value_error` (all states MSE), `internal_mse` (non-leaf decision states $0 \dots 2^D-2$), `leaf_mse`, `root_abs_error`, and `sibling_ranking_acc` (advantage sign accuracy on sibling branches).
- `vine_target_value.py`: Pure-JAX implementation of **Vine-Target Critic + Variant 2 Online Shrinkage Audit**:
  - Tracks online `sigma2_critic`, `mean_alpha`, `mc_batch_mse`, `critic_batch_mse`, `interp_batch_mse`, `internal_mse`, and `sibling_ranking_acc`.
- `mc_value.py`: Standard single-trajectory Monte Carlo critic baseline ($K = 1$).
- `vine.py` & `compare_vine_mc.py`: Online non-parametric Vine estimator sweep and Figure 3 scatter comparisons.
- `learning_curves_and_variance.py`: Multi-seed benchmark comparing `Standard MC (K=1)`, `Vine-Target (K=4)`, `FBI (K=4, lam=0.0)`, and `FBI (K=4, lam=0.5)` at matched transition budgets, producing learning curves and exact bias-variance decompositions.

---

## 3. Quick Commands to Run

Using the local JAX Python environment (`/Users/dillonsandhu/Documents/Projects/myenv/bin/python`):

```bash
# 1. Run single-method FBI(lambda) test (compare lambda=0.0 vs lambda=0.5)
/Users/dillonsandhu/Documents/Projects/myenv/bin/python fbi_value.py --lmbda 0.0
/Users/dillonsandhu/Documents/Projects/myenv/bin/python fbi_value.py --lmbda 0.5

# 2. Run full multi-seed comparison & bias-variance decomposition
/Users/dillonsandhu/Documents/Projects/myenv/bin/python learning_curves_and_variance.py --fbi-lambdas 0.0 0.5 --seeds 10
```
