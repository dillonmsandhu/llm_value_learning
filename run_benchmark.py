from __future__ import annotations
import os
import sys
import time
import argparse
import pickle
import datetime
from typing import Tuple, Dict, Any, List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from transformer import TransformerCritic
from envs.reasoning_dag_env import ReasoningDAGEnv, DAGState, DAGEnvParams
from envs.tree_mrp import TreeMRP, TreeMRPExactValue, TabularState, ProceduralTreeEnv, ProceduralTreeExactValue, TreeState
import plot_results


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
class StepData(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray


# -----------------------------------------------------------------------------
# Evaluation Dataset Construction
# -----------------------------------------------------------------------------
def build_dag_eval_set(env: ReasoningDAGEnv, samples_per_problem: int = 32, seed: int = 12345):
    """
    Constructs a balanced evaluation dataset for ReasoningDAGEnv covering
    each problem index (difficulty tier) equally, spanning steps, milestones, and derailment conditions.
    Filters out dummy post-termination states.
    """
    P = env.num_problems
    num_rollouts = max(16, (samples_per_problem // 6) + 4)

    all_prompts, all_m, all_d, all_steps, all_hist, all_done = [], [], [], [], [], []

    for p in range(P):
        rng = jax.random.PRNGKey(seed + p * 1000)
        def reset_prob(k):
            history, s = env.reset(k)
            history = history.at[1].set(env.evaluator.PROMPT_TOKEN_OFFSET + p)
            s = s._replace(prompt_idx=jnp.array(p, dtype=jnp.int32), history_tokens=history)
            return history, s

        obs, state = jax.vmap(reset_prob)(jax.random.split(rng, num_rollouts))
        p_states = [state]
        curr_state = state
        for _ in range(env.max_steps):
            rng, k1, k2 = jax.random.split(rng, 3)
            actions = jax.vmap(env.sample_base_action)(jax.random.split(k1, num_rollouts), curr_state)
            next_obs, next_state, r, d, _ = jax.vmap(env.step)(jax.random.split(k2, num_rollouts), curr_state, actions)
            p_states.append(next_state)
            curr_state = next_state

        flat_p_prompt = jnp.concatenate([st.prompt_idx for st in p_states], axis=0)
        flat_p_m = jnp.concatenate([st.milestone for st in p_states], axis=0)
        flat_p_d = jnp.concatenate([st.is_derailed for st in p_states], axis=0)
        flat_p_s = jnp.concatenate([st.step_count for st in p_states], axis=0)
        flat_p_h = jnp.concatenate([st.history_tokens for st in p_states], axis=0)
        flat_p_done = jnp.concatenate([st.done for st in p_states], axis=0)

        # Strictly select active (non-done) states uniformly across all timesteps 0..max_steps
        active_idx = jnp.where(~flat_p_done)[0]
        n_sel = min(samples_per_problem, int(active_idx.shape[0]))
        rng, sel_rng = jax.random.split(rng)
        sel_idx = jax.random.choice(sel_rng, active_idx, shape=(n_sel,), replace=False)

        all_prompts.append(flat_p_prompt[sel_idx])
        all_m.append(flat_p_m[sel_idx])
        all_d.append(flat_p_d[sel_idx])
        all_steps.append(flat_p_s[sel_idx])
        all_hist.append(flat_p_h[sel_idx])
        all_done.append(flat_p_done[sel_idx])

    flat_prompt = jnp.concatenate(all_prompts, axis=0)
    flat_milestone = jnp.concatenate(all_m, axis=0)
    flat_derailed = jnp.concatenate(all_d, axis=0)
    flat_steps = jnp.concatenate(all_steps, axis=0)
    flat_obs = jnp.concatenate(all_hist, axis=0)
    flat_done = jnp.concatenate(all_done, axis=0)
    num_samples = flat_obs.shape[0]

    eval_state = DAGState(
        prompt_idx=flat_prompt,
        milestone=flat_milestone,
        is_derailed=flat_derailed,
        step_count=flat_steps,
        history_tokens=flat_obs,
        done=flat_done,
    )

    exact_v = jax.vmap(env.evaluator.get_exact_value)(eval_state)

    def _action_target(a):
        act_arr = jnp.full(num_samples, a, dtype=jnp.int32)
        step_keys = jax.random.split(jax.random.PRNGKey(a), num_samples)
        next_o, next_st, r, d, _ = jax.vmap(env.step)(step_keys, eval_state, act_arr)
        v_next = jax.vmap(env.evaluator.get_exact_value)(next_st)
        return next_o, r, v_next, d

    all_acts = jnp.arange(env.num_actions)
    next_obs_all, r_all, v_next_all, d_all = jax.vmap(_action_target)(all_acts)
    true_adv = r_all + env.evaluator.gamma * v_next_all - exact_v[None, :]

    all_eval_tokens = jnp.concatenate([flat_obs, next_obs_all.reshape(-1, env.max_seq_len)], axis=0)

    return {
        "eval_obs": flat_obs,
        "eval_state": eval_state,
        "exact_v": exact_v,
        "true_adv": true_adv,
        "next_obs_all": next_obs_all,
        "all_eval_tokens": all_eval_tokens,
        "r_all": r_all,
        "d_all": d_all,
        "num_actions": env.num_actions,
        "prompts": flat_prompt,
        "num_problems": P,
    }


def build_tree_eval_set(env: Any, samples_per_problem: int = 32, seed: int = 12345):
    """
    Constructs a balanced evaluation dataset for ProceduralTreeEnv covering
    each problem index (difficulty tier) equally, spanning steps, branches, and outcomes.
    """
    if hasattr(env, "evaluator") and hasattr(env.evaluator, "target_depths"):
        P = env.num_problems
        num_rollouts = max(16, (samples_per_problem // max(1, env.max_depth)) + 4)

        all_prompts, all_nodes, all_depths, all_hist, all_done = [], [], [], [], []

        for p in range(P):
            rng = jax.random.PRNGKey(seed + p * 1000)
            def reset_prob(k):
                history, s = env.reset(k)
                history = history.at[1].set(env.evaluator.PROMPT_TOKEN_OFFSET + p)
                s = s._replace(prompt_idx=jnp.array(p, dtype=jnp.int32), history_tokens=history)
                return history, s

            obs, state = jax.vmap(reset_prob)(jax.random.split(rng, num_rollouts))
            p_states = [state]
            curr_state = state
            for _ in range(env.max_depth):
                rng, k1, k2 = jax.random.split(rng, 3)
                actions = jax.vmap(env.sample_base_action)(jax.random.split(k1, num_rollouts), curr_state)
                next_obs, next_state, r, d, _ = jax.vmap(env.step)(jax.random.split(k2, num_rollouts), curr_state, actions)
                p_states.append(next_state)
                curr_state = next_state

            flat_p_prompt = jnp.concatenate([st.prompt_idx for st in p_states], axis=0)
            flat_p_node = jnp.concatenate([st.node_idx for st in p_states], axis=0)
            flat_p_depth = jnp.concatenate([st.depth_count for st in p_states], axis=0)
            flat_p_hist = jnp.concatenate([st.history_tokens for st in p_states], axis=0)
            flat_p_done = jnp.concatenate([st.done for st in p_states], axis=0)

            active_idx = jnp.where(~flat_p_done)[0]
            n_sel = min(samples_per_problem, int(active_idx.shape[0]))
            rng, sel_rng = jax.random.split(rng)
            sel_idx = jax.random.choice(sel_rng, active_idx, shape=(n_sel,), replace=False)

            all_prompts.append(flat_p_prompt[sel_idx])
            all_nodes.append(flat_p_node[sel_idx])
            all_depths.append(flat_p_depth[sel_idx])
            all_hist.append(flat_p_hist[sel_idx])
            all_done.append(flat_p_done[sel_idx])

        flat_prompt = jnp.concatenate(all_prompts, axis=0)
        flat_node = jnp.concatenate(all_nodes, axis=0)
        flat_depth = jnp.concatenate(all_depths, axis=0)
        flat_obs = jnp.concatenate(all_hist, axis=0)
        flat_done = jnp.concatenate(all_done, axis=0)
        num_samples = flat_obs.shape[0]

        eval_state = TreeState(
            prompt_idx=flat_prompt,
            node_idx=flat_node,
            depth_count=flat_depth,
            history_tokens=flat_obs,
            done=flat_done,
        )

        exact_v = jax.vmap(env.evaluator.get_exact_value)(eval_state)

        def _action_target(a):
            act_arr = jnp.full(num_samples, a, dtype=jnp.int32)
            step_keys = jax.random.split(jax.random.PRNGKey(a), num_samples)
            next_o, next_st, r, d, _ = jax.vmap(env.step)(step_keys, eval_state, act_arr)
            v_next = jax.vmap(env.evaluator.get_exact_value)(next_st)
            return next_o, r, v_next, d

        all_acts = jnp.arange(env.num_actions)
        next_obs_all, r_all, v_next_all, d_all = jax.vmap(_action_target)(all_acts)
        true_adv = r_all + env.evaluator.gamma * (1.0 - d_all.astype(jnp.float32)) * v_next_all - exact_v[None, :]

        all_eval_tokens = jnp.concatenate([flat_obs, next_obs_all.reshape(-1, env.max_seq_len)], axis=0)

        return {
            "eval_obs": flat_obs,
            "eval_state": eval_state,
            "exact_v": exact_v,
            "true_adv": true_adv,
            "next_obs_all": next_obs_all,
            "all_eval_tokens": all_eval_tokens,
            "r_all": r_all,
            "d_all": d_all,
            "num_actions": env.num_actions,
            "prompts": flat_prompt,
            "num_problems": P,
        }
    else:
        evaluator = env.evaluator
        num_states = evaluator.num_states
        v_true_all = evaluator.solve_backward_induction()[:num_states]
        eval_obs = evaluator.obs_stack[:num_states]

        num_internal = evaluator.leaf_offset
        parents = jnp.arange(num_internal, dtype=jnp.int32)
        left_c = 2 * parents + 1
        right_c = 2 * parents + 2

        true_adv_left = v_true_all[left_c] - v_true_all[parents]
        true_adv_right = v_true_all[right_c] - v_true_all[parents]
        true_adv = jnp.stack([true_adv_left, true_adv_right], axis=0)

        return {
            "eval_obs": eval_obs,
            "exact_v": v_true_all,
            "true_adv": true_adv,
            "num_internal": num_internal,
            "left_c": left_c,
            "right_c": right_c,
            "parents": parents,
            "num_problems": 1,
        }


# -----------------------------------------------------------------------------
# Metric Evaluation Functions
# -----------------------------------------------------------------------------
def evaluate_dag_metrics(network: TransformerCritic, params: Any, eval_set: Dict[str, Any], gamma: float = 1.0):
    """Evaluates MSVE and Advantage Cosine Similarity on ReasoningDAGEnv, globally and per-problem."""
    all_eval_tokens = eval_set["all_eval_tokens"]
    exact_v = eval_set["exact_v"]
    true_adv = eval_set["true_adv"]
    r_all = eval_set["r_all"]
    N = eval_set["eval_obs"].shape[0]
    num_actions = eval_set["num_actions"]
    prompts = eval_set.get("prompts", None)
    P = eval_set.get("num_problems", 1)

    # Single unified forward pass for both current and next state observations
    all_v = network.apply(params, all_eval_tokens, method=network.value)
    v_pred = all_v[:N]
    v_next_pred = all_v[N:].reshape(num_actions, N)

    msve = jnp.mean(jnp.square(v_pred - exact_v))
    d_all = eval_set.get("d_all", None)
    if d_all is not None:
        pred_adv = r_all + gamma * (1.0 - d_all.astype(jnp.float32)) * v_next_pred - v_pred[None, :]
    else:
        pred_adv = r_all + gamma * v_next_pred - v_pred[None, :]

    # Advantage Cosine Similarity across all (action, state) pairs
    dot = jnp.sum(true_adv * pred_adv)
    norm_true = jnp.linalg.norm(true_adv)
    norm_pred = jnp.linalg.norm(pred_adv)
    adv_cos_sim = dot / (norm_true * norm_pred + 1e-8)

    # Per-problem breakdown across difficulty tiers
    if prompts is not None and P > 1:
        def _prob_metrics(p):
            mask = (prompts == p)
            msve_p = jnp.sum(jnp.square(v_pred - exact_v) * mask) / jnp.maximum(1.0, jnp.sum(mask))
            true_p = jnp.where(mask[None, :], true_adv, 0.0)
            pred_p = jnp.where(mask[None, :], pred_adv, 0.0)
            dot_p = jnp.sum(true_p * pred_p)
            norm_t = jnp.linalg.norm(true_p)
            norm_pr = jnp.linalg.norm(pred_p)
            adv_p = dot_p / (norm_t * norm_pr + 1e-8)
            return msve_p, adv_p

        msve_per_prob, adv_per_prob = jax.vmap(_prob_metrics)(jnp.arange(P))
    else:
        msve_per_prob = msve[None]
        adv_per_prob = adv_cos_sim[None]

    return msve, adv_cos_sim, v_pred, msve_per_prob, adv_per_prob


def evaluate_tree_metrics(network: TransformerCritic, params: Any, eval_set: Dict[str, Any]):
    """Evaluates MSVE and Advantage Cosine Similarity on TreeMRP."""
    eval_obs = eval_set["eval_obs"]
    exact_v = eval_set["exact_v"]
    true_adv = eval_set["true_adv"] # (2, num_internal)
    parents = eval_set["parents"]
    left_c = eval_set["left_c"]
    right_c = eval_set["right_c"]

    v_pred = network.apply(params, eval_obs, method=network.value)
    msve = jnp.mean(jnp.square(v_pred - exact_v))

    v_p = v_pred[parents]
    pred_adv_l = v_pred[left_c] - v_p
    pred_adv_r = v_pred[right_c] - v_p
    pred_adv = jnp.stack([pred_adv_l, pred_adv_r], axis=0)

    dot = jnp.sum(true_adv * pred_adv)
    norm_true = jnp.linalg.norm(true_adv)
    norm_pred = jnp.linalg.norm(pred_adv)
    adv_cos_sim = dot / (norm_true * norm_pred + 1e-8)

    msve_per_prob = msve[None]
    adv_per_prob = adv_cos_sim[None]

    return msve, adv_cos_sim, v_pred, msve_per_prob, adv_per_prob


# -----------------------------------------------------------------------------
# Training Functions for Algorithms
# -----------------------------------------------------------------------------
def make_train_loop(
    algo: str,
    env_type: str,
    env: Any,
    eval_set: Dict[str, Any],
    cfg: Dict[str, Any],
):
    """
    Constructs a compiled training function for:
      - algo in ['mc', 'vine_k2', 'vine_k4', 'fbi_lam0.0', 'fbi_lam0.5']
      - env_type in ['dag', 'tree']
    """
    gamma = float(cfg.get("GAMMA", 1.0))
    num_envs = int(cfg.get("NUM_ENVS", 64))
    num_steps = int(cfg.get("NUM_STEPS", 24))
    num_updates = int(cfg.get("NUM_UPDATES", 100))
    grad_steps_per_update = int(cfg.get("GRAD_STEPS_PER_UPDATE", num_steps))
    minibatch_size = int(cfg.get("MINIBATCH_SIZE", 64))
    lr = float(cfg.get("LR", 5e-4))
    weight_decay = float(cfg.get("WEIGHT_DECAY", 1e-4))
    warmup_ratio = float(cfg.get("WARMUP_RATIO", 0.10))
    max_grad_norm = float(cfg.get("MAX_GRAD_NORM", 1.0))

    # Parse algo parameters
    is_mc = (algo == "mc")
    is_vine = algo.startswith("vine")
    is_fbi_staged = algo.startswith("fbi_staged")
    is_fbi_batched = algo.startswith("fbi") and not is_fbi_staged
    is_fbi = is_fbi_staged or is_fbi_batched
    k_branches = 4
    lmbda = 0.0
    # For staged FBI: distribute grad steps across reverse levels
    stage_steps = max(1, grad_steps_per_update // num_steps)

    if is_vine:
        k_branches = int(algo.split("_k")[1])
    elif is_fbi:
        lmbda = float(algo.split("_lam")[1])

    is_dag = (env_type in ("dag", "single_dag", "dag_spectrum"))
    vocab_size = env.vocab_size
    max_seq_len = env.max_seq_len
    pad_token_id = 0
    add_bos = False

    network = TransformerCritic(
        vocab_size=vocab_size,
        d_model=cfg.get("D_MODEL", 64),
        n_heads=cfg.get("N_HEADS", 4),
        n_layers=cfg.get("N_LAYERS", 2),
        final_hidden_dim=cfg.get("FINAL_HIDDEN_DIM", 32),
        max_len=max_seq_len,
        pad_token_id=pad_token_id,
        add_bos=add_bos,
    )

    # RLVR Best Practice Optimizer: Warmup-Cosine Schedule + Gradient Clipping + Calibrated Weight Decay
    total_grad_steps = num_updates * grad_steps_per_update
    warmup_steps = max(1, int(warmup_ratio * total_grad_steps))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_grad_steps,
        end_value=0.05 * lr,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    # Snapshots indices
    u_snap_1 = max(1, int(round(0.2 * num_updates)))
    u_snap_4 = max(1, int(round(0.8 * num_updates)))

    def train_seed(rng: jax.random.PRNGKey):
        rng, init_rng, reset_rng = jax.random.split(rng, 3)

        # 1. Initialize Network & Optimizer State
        dummy_obs = eval_set["eval_obs"][:1]
        params = network.init(init_rng, dummy_obs)
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)

        # 2. Reset Environments
        reset_keys = jax.random.split(reset_rng, num_envs)
        last_obs, env_state = jax.vmap(env.reset)(reset_keys)

        def _update_step(runner_state, update_idx):
            train_state, _, _, rng, snap_1_val, snap_4_val = runner_state

            # Freshly reset all environments at the start of every update (episodes are bounded by num_steps)
            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            last_obs, env_state = jax.vmap(env.reset)(reset_keys)

            # --- A. Collect Primary Base Trajectory (Pure Environment Rollout, Zero Network Calls) ---
            def _step_fn(carry, _):
                st, o, k = carry
                k, act_k, step_k = jax.random.split(k, 3)

                a = jax.vmap(env.sample_base_action)(jax.random.split(act_k, num_envs), st)
                next_o, next_st, r, d, _ = jax.vmap(env.step)(jax.random.split(step_k, num_envs), st, a)

                step_data = StepData(obs=o, action=a, reward=r, done=d, value=jnp.zeros(num_envs, dtype=jnp.float32))
                return (next_st, next_o, k), (step_data, st)

            (env_state, last_obs, rng), (traj_batch, visited_states) = jax.lax.scan(
                _step_fn, (env_state, last_obs, rng), None, length=num_steps
            )
            trunk_active_steps = jnp.sum(~visited_states.done).astype(jnp.float32)

            # --- B. Compute Targets ---
            if is_mc:
                # Backward Monte Carlo return: G_t = r_t + gamma * (1 - d_{t+1}) * G_{t+1}
                def _mc_back(next_g, data):
                    r, d = data
                    g = r + gamma * (1.0 - d.astype(jnp.float32)) * next_g
                    return g, g

                _, targets = jax.lax.scan(
                    _mc_back, jnp.zeros(num_envs, dtype=jnp.float32),
                    (traj_batch.reward, traj_batch.done), reverse=True
                )
                active_infer_steps = trunk_active_steps
                critic_eval_steps = jnp.array(0.0, dtype=jnp.float32)
            elif is_vine:
                # Fork K rollouts to horizon T for each visited state
                def _vine_rollout(state, r_key):
                    def _fork_step(carry, _):
                        st, k = carry
                        k, ak, sk = jax.random.split(k, 3)
                        a = jax.vmap(env.sample_base_action)(jax.random.split(ak, k_branches), st)
                        _, n_st, r, d, _ = jax.vmap(env.step)(jax.random.split(sk, k_branches), st, a)
                        active_f = jnp.sum(~st.done).astype(jnp.float32)
                        return (n_st, k), (r, d, active_f)

                    # Repeat state k_branches times
                    if is_dag:
                        fork_st = DAGState(
                            prompt_idx=jnp.repeat(state.prompt_idx, k_branches),
                            milestone=jnp.repeat(state.milestone, k_branches),
                            is_derailed=jnp.repeat(state.is_derailed, k_branches),
                            step_count=jnp.repeat(state.step_count, k_branches),
                            history_tokens=jnp.repeat(state.history_tokens[None, :], k_branches, axis=0),
                            done=jnp.repeat(state.done, k_branches),
                        )
                    else:
                        fork_st = TreeState(
                            prompt_idx=jnp.repeat(state.prompt_idx, k_branches),
                            node_idx=jnp.repeat(state.node_idx, k_branches),
                            depth_count=jnp.repeat(state.depth_count, k_branches),
                            history_tokens=jnp.repeat(state.history_tokens[None, :], k_branches, axis=0),
                            done=jnp.repeat(state.done, k_branches),
                        )

                    _, (r_seq, d_seq, f_active_seq) = jax.lax.scan(_fork_step, (fork_st, r_key), None, length=num_steps)

                    def _back_return(next_g, rd):
                        r, d = rd
                        g = r + gamma * (1.0 - d.astype(jnp.float32)) * next_g
                        return g, g

                    _, ret_seq = jax.lax.scan(_back_return, jnp.zeros(k_branches, dtype=jnp.float32), (r_seq, d_seq), reverse=True)
                    return jnp.mean(ret_seq[0]), jnp.sum(f_active_seq)

                rng, v_rng = jax.random.split(rng)
                v_keys = jax.random.split(v_rng, num_steps * num_envs).reshape(num_steps, num_envs, 2)
                targets, fork_active_counts = jax.vmap(jax.vmap(_vine_rollout))(visited_states, v_keys)
                active_infer_steps = trunk_active_steps + jnp.sum(fork_active_counts)
                critic_eval_steps = jnp.array(0.0, dtype=jnp.float32)

            elif is_fbi_batched:
                # Synchronous / Batched 1-step branching target:
                # 1. Sample K 1-step branches across all (T, B) states in parallel
                def _sample_branch_1step(state, r_key):
                    ak, sk = jax.random.split(r_key)
                    if is_dag:
                        fork_st = DAGState(
                            prompt_idx=jnp.repeat(state.prompt_idx, k_branches),
                            milestone=jnp.repeat(state.milestone, k_branches),
                            is_derailed=jnp.repeat(state.is_derailed, k_branches),
                            step_count=jnp.repeat(state.step_count, k_branches),
                            history_tokens=jnp.repeat(state.history_tokens[None, :], k_branches, axis=0),
                            done=jnp.repeat(state.done, k_branches),
                        )
                    else:
                        fork_st = TreeState(
                            prompt_idx=jnp.repeat(state.prompt_idx, k_branches),
                            node_idx=jnp.repeat(state.node_idx, k_branches),
                            depth_count=jnp.repeat(state.depth_count, k_branches),
                            history_tokens=jnp.repeat(state.history_tokens[None, :], k_branches, axis=0),
                            done=jnp.repeat(state.done, k_branches),
                        )
                    a = jax.vmap(env.sample_base_action)(jax.random.split(ak, k_branches), fork_st)
                    next_o, _, r, d, _ = jax.vmap(env.step)(jax.random.split(sk, k_branches), fork_st, a)
                    return next_o, r, d

                rng, b_rng = jax.random.split(rng)
                b_keys = jax.random.split(b_rng, num_steps * num_envs).reshape(num_steps, num_envs, 2)
                b_next_obs, b_rew, b_done = jax.vmap(jax.vmap(_sample_branch_1step))(visited_states, b_keys)

                # 2. Single fused forward pass over all T * B trunk observations + T * B * K branch observations
                n_trunk = num_steps * num_envs
                flat_trunk_obs = traj_batch.obs.reshape(n_trunk, -1)
                flat_b_next_obs = b_next_obs.reshape(n_trunk * k_branches, -1)
                all_fbi_obs = jnp.concatenate([flat_trunk_obs, flat_b_next_obs], axis=0)

                all_fbi_v = jax.lax.stop_gradient(
                    network.apply(train_state.params, all_fbi_obs, method=network.value)
                )
                v_base = all_fbi_v[:n_trunk].reshape(num_steps, num_envs)
                v_next = all_fbi_v[n_trunk:].reshape(num_steps, num_envs, k_branches)
                branch_ret = b_rew + gamma * (1.0 - b_done.astype(jnp.float32)) * v_next
                v_branch = jnp.mean(branch_ret, axis=-1)

                # Formulation B: GAE over branched residuals
                delta_branch = v_branch - v_base

                def _gae_back(next_adv, data):
                    delta, d = data
                    adv = delta + (gamma * lmbda) * (1.0 - d.astype(jnp.float32)) * next_adv
                    return adv, adv

                _, gae_adv = jax.lax.scan(
                    _gae_back, jnp.zeros(num_envs, dtype=jnp.float32),
                    (delta_branch, traj_batch.done), reverse=True
                )
                targets = v_base + gae_adv
                active_infer_steps = trunk_active_steps * (1.0 + float(k_branches))
                critic_eval_steps = trunk_active_steps * (1.0 + float(k_branches))

            elif is_fbi_staged:
                # Staged / Sequential FBI (XLA-optimized with jax.lax.scan & active-pool minibatching):
                # 1. Sample K 1-step branches from all visited states in parallel
                def _sample_branch_env_step(state, r_key):
                    ak, sk = jax.random.split(r_key)
                    if is_dag:
                        fork_st = DAGState(
                            prompt_idx=jnp.repeat(state.prompt_idx, k_branches),
                            milestone=jnp.repeat(state.milestone, k_branches),
                            is_derailed=jnp.repeat(state.is_derailed, k_branches),
                            step_count=jnp.repeat(state.step_count, k_branches),
                            history_tokens=jnp.repeat(state.history_tokens[None, :], k_branches, axis=0),
                            done=jnp.repeat(state.done, k_branches),
                        )
                    else:
                        fork_st = TreeState(
                            prompt_idx=jnp.repeat(state.prompt_idx, k_branches),
                            node_idx=jnp.repeat(state.node_idx, k_branches),
                            depth_count=jnp.repeat(state.depth_count, k_branches),
                            history_tokens=jnp.repeat(state.history_tokens[None, :], k_branches, axis=0),
                            done=jnp.repeat(state.done, k_branches),
                        )
                    a = jax.vmap(env.sample_base_action)(jax.random.split(ak, k_branches), fork_st)
                    next_o, _, r, d, _ = jax.vmap(env.step)(jax.random.split(sk, k_branches), fork_st, a)
                    return next_o, r, d

                rng, b_rng, stage_scan_rng = jax.random.split(rng, 3)
                b_keys = jax.random.split(b_rng, num_steps * num_envs).reshape(num_steps, num_envs, 2)
                # Shapes: b_next_obs (T, B, K, obs_dim), b_rew (T, B, K), b_done (T, B, K)
                b_next_obs, b_rew, b_done = jax.vmap(jax.vmap(_sample_branch_env_step))(visited_states, b_keys)

                # 2. Reverse backward induction scan (t = num_steps - 1 down to 0) with cumulative active-pool fitting
                def _staged_backward_step(carry, t):
                    curr_ts, curr_targets, next_adv, k_rng = carry
                    k_rng, stage_k = jax.random.split(k_rng)

                    # Single fused forward pass for current trunk state and K 1-step children using just-updated critic
                    flat_next_o = b_next_obs[t].reshape(num_envs * k_branches, -1)
                    eval_obs_t = jnp.concatenate([traj_batch.obs[t], flat_next_o], axis=0)
                    eval_v_t = jax.lax.stop_gradient(
                        network.apply(curr_ts.params, eval_obs_t, method=network.value)
                    )
                    v_curr_t = eval_v_t[:num_envs]
                    v_next = eval_v_t[num_envs:].reshape(num_envs, k_branches)
                    r_t = b_rew[t]
                    d_t = b_done[t].astype(jnp.float32)
                    v_branch_t = jnp.mean(r_t + gamma * (1.0 - d_t) * v_next, axis=-1)
                    delta_t = v_branch_t - v_curr_t
                    trunk_d_t = traj_batch.done[t].astype(jnp.float32)
                    curr_adv = delta_t + (gamma * lmbda) * (1.0 - trunk_d_t) * next_adv
                    y_t = v_curr_t + curr_adv
                    curr_targets = curr_targets.at[t].set(y_t)

                    # Run `stage_steps` minibatch gradient steps sampled strictly from active levels [t .. num_steps - 1]
                    def _stage_mb_step(mb_carry, _):
                        ts_inner, k_in = mb_carry
                        k_in, k_lvl, k_env = jax.random.split(k_in, 3)
                        mb_levels = jax.random.randint(k_lvl, (minibatch_size,), minval=t, maxval=num_steps)
                        mb_envs = jax.random.randint(k_env, (minibatch_size,), minval=0, maxval=num_envs)
                        m_obs = traj_batch.obs[mb_levels, mb_envs]
                        m_targ = curr_targets[mb_levels, mb_envs]
                        m_mask = (~visited_states.done)[mb_levels, mb_envs].astype(jnp.float32)

                        def _loss(p):
                            preds = network.apply(p, m_obs, method=network.value)
                            sq_err = jnp.square(preds - m_targ)
                            return 0.5 * jnp.sum(sq_err * m_mask) / jnp.maximum(1.0, jnp.sum(m_mask))

                        loss_val, grads = jax.value_and_grad(_loss)(ts_inner.params)
                        ts_inner = ts_inner.apply_gradients(grads=grads)
                        return (ts_inner, k_in), loss_val

                    (curr_ts, _), _ = jax.lax.scan(_stage_mb_step, (curr_ts, stage_k), None, length=stage_steps)
                    return (curr_ts, curr_targets, curr_adv, k_rng), None

                init_targets = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
                init_adv = jnp.zeros(num_envs, dtype=jnp.float32)
                rev_levels = jnp.arange(num_steps - 1, -1, -1, dtype=jnp.int32)
                (train_state, targets, _, _), _ = jax.lax.scan(
                    _staged_backward_step,
                    (train_state, init_targets, init_adv, stage_scan_rng),
                    rev_levels,
                )
                active_infer_steps = trunk_active_steps * (1.0 + float(k_branches))
                critic_eval_steps = trunk_active_steps * (1.0 + float(k_branches))

            # --- C. Critic Network Update (MSE Loss for non-staged methods on active states) ---
            if not is_fbi_staged:
                obs_flat = traj_batch.obs.reshape(-1, *traj_batch.obs.shape[2:])
                targets_flat = targets.reshape(-1)
                active_mask_flat = (~visited_states.done).reshape(-1)
                num_active = jnp.sum(active_mask_flat)
                active_order = jnp.argsort(~active_mask_flat)
                obs_active = obs_flat[active_order]
                targets_active = targets_flat[active_order]

                def _step_mb(carry, _):
                    ts, k = carry
                    k, mb_k = jax.random.split(k)
                    mb_idx = jax.random.randint(mb_k, (minibatch_size,), minval=0, maxval=jnp.maximum(1, num_active))
                    m_obs = obs_active[mb_idx]
                    m_targ = targets_active[mb_idx]

                    def _loss(p):
                        preds = network.apply(p, m_obs, method=network.value)
                        return 0.5 * jnp.mean(jnp.square(preds - m_targ))

                    loss, grads = jax.value_and_grad(_loss)(ts.params)
                    ts = ts.apply_gradients(grads=grads)
                    return (ts, k), loss

                (train_state, rng), _ = jax.lax.scan(_step_mb, (train_state, rng), None, length=grad_steps_per_update)

            # Estimated LLM FLOPs (units of N = model parameters):
            # 2N per inference / target-eval step, 6N per trained state-step (forward + backward)
            actual_step_grads = (stage_steps * num_steps) if is_fbi_staged else grad_steps_per_update
            trained_steps = jnp.array(float(actual_step_grads * minibatch_size), dtype=jnp.float32)
            llm_flops_per_n = 2.0 * (active_infer_steps + critic_eval_steps) + 6.0 * trained_steps

            # --- D. Evaluation across Fixed Eval Set ---
            if "all_eval_tokens" in eval_set:
                msve, adv_cos, v_eval_preds, msve_per_prob, adv_per_prob = evaluate_dag_metrics(network, train_state.params, eval_set, gamma=gamma)
            else:
                msve, adv_cos, v_eval_preds, msve_per_prob, adv_per_prob = evaluate_tree_metrics(network, train_state.params, eval_set)

            # Record snapshots at 1/5 and 4/5
            snap_1_val = jnp.where(update_idx == u_snap_1, v_eval_preds, snap_1_val)
            snap_4_val = jnp.where(update_idx == u_snap_4, v_eval_preds, snap_4_val)

            metrics_step = {
                "msve": msve,
                "advantage_cosine_sim": adv_cos,
                "msve_per_problem": msve_per_prob,
                "adv_cos_per_problem": adv_per_prob,
                "active_infer_steps": active_infer_steps,
                "critic_eval_steps": critic_eval_steps,
                "trained_steps": trained_steps,
                "llm_flops_per_n": llm_flops_per_n,
            }

            return (train_state, env_state, last_obs, rng, snap_1_val, snap_4_val), metrics_step

        num_eval_states = eval_set["eval_obs"].shape[0]
        init_snap_1 = jnp.zeros(num_eval_states, dtype=jnp.float32)
        init_snap_4 = jnp.zeros(num_eval_states, dtype=jnp.float32)

        init_runner = (train_state, env_state, last_obs, rng, init_snap_1, init_snap_4)
        update_indices = jnp.arange(1, num_updates + 1)
        (final_train_state, _, _, _, snap_1, snap_4), metrics = jax.lax.scan(
            _update_step, init_runner, update_indices
        )

        return {
            "metrics": metrics,
            "snap_1_5": snap_1,
            "snap_4_5": snap_4,
        }

    return train_seed


# -----------------------------------------------------------------------------
# Main Benchmark Suite Runner
# -----------------------------------------------------------------------------
def run_benchmark(
    env_type: str = "dag",
    total_transitions_budget: int = 200000,
    n_seeds: int = 5,
    master_seed: int = 42,
    algos: List[str] = None,
    save_dir: str = "results/benchmark",
    dag_params: Dict[str, Any] = None,
    tree_params: Dict[str, Any] = None,
    trajectories_per_update: int = 64,
    epochs_per_update: int = 2,
    minibatch_size: int = 64,
    num_updates: int = None,
    grad_steps_per_update: int = None,
    compute_matched_mc: bool = True,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.10,
    max_grad_norm: float = 1.0,
):
    """
    Executes the comprehensive Value Learning & Advantage Alignment benchmark suite.
    Runs all algorithms with identical trunk batch size (B trajectories/update),
    identical gradient steps per update, and identical buffer replay ratio, then
    compares optimization per gradient step alongside active rollout transitions
    and Estimated LLM FLOPs (in units of N parameters), plus an optional
    Compute-Matched MC baseline.
    """
    os.makedirs(save_dir, exist_ok=True)
    algos = algos or ["mc", "vine_k2", "vine_k4", "fbi_lam0.0", "fbi_lam0.8", "fbi_staged_lam0.0", "fbi_staged_lam0.8"]
    resolved_updates = int(num_updates) if num_updates is not None else (25 if env_type == "single_dag" else 100)

    print("=" * 85)
    print(f"BENCHMARK SUITE: VALUE LEARNING & ADVANTAGE ALIGNMENT")
    print(f"  Target Environment: {env_type.upper()}")
    print(f"  Updates / Trunk B:  {resolved_updates} updates x {trajectories_per_update} traj/update")
    print(f"  Seeds:              {n_seeds} (Master Seed: {master_seed})")
    print(f"  Algorithms:         {algos}")
    print(f"  JAX Platform:       {jax.default_backend().upper()} | Devices: {jax.devices()}")
    print(f"  Save Directory:     {save_dir}")
    print("=" * 85)

    # 1. Initialize Environment & Precompute Exact Evaluator
    is_dag_family = (env_type in ("dag", "single_dag", "dag_spectrum"))
    if is_dag_family:
        if env_type == "single_dag":
            dag_params = dag_params or {"num_problems": 1, "num_milestones": 8, "max_steps": 32, "num_actions": 8, "difficulty": 0.5}
            diff_val = dag_params.get("difficulty", 0.5)
            env_title = f"Single Large Reasoning DAG (Difficulty theta={diff_val}, M={dag_params['num_milestones']}, T={dag_params['max_steps']}, |A|={dag_params['num_actions']})"
            difficulties = diff_val
            num_probs = 1
        else:
            dag_params = dag_params or {"num_problems": 10, "num_milestones": 6, "max_steps": 24, "num_actions": 8}
            num_probs = dag_params.get("num_problems", 10)
            env_title = f"Difficulty Spectrum Reasoning DAG (P={num_probs}, M={dag_params['num_milestones']}, T={dag_params['max_steps']}, |A|={dag_params['num_actions']})"
            difficulties = "spectrum"

        env = ReasoningDAGEnv(
            num_problems=num_probs,
            num_milestones=dag_params["num_milestones"],
            max_steps=dag_params["max_steps"],
            num_actions=dag_params["num_actions"],
            difficulties=difficulties,
            seed=master_seed,
        )
        samples_per_p = max(8, 256 // num_probs)
        eval_set = build_dag_eval_set(env, samples_per_problem=samples_per_p, seed=env.eval_seed)
        num_steps = env.max_steps
    else:
        # Procedural Tree Family
        is_single_tree = (env_type == "single_tree")
        if is_single_tree:
            tree_params = tree_params or {"num_problems": 1, "depth": 7, "difficulty": 0.5}
            diff_val = tree_params.get("difficulty", 0.5)
            env_title = f"Single Procedural Tree (Difficulty theta={diff_val}, Depth={tree_params.get('depth', 7)})"
            difficulties = diff_val
            num_probs = 1
        else:
            tree_params = tree_params or {"num_problems": 10, "depth": 7}
            num_probs = tree_params.get("num_problems", 10)
            env_title = f"Difficulty Spectrum Procedural Tree (P={num_probs}, Depth={tree_params.get('depth', 7)})"
            difficulties = "spectrum"

        env = ProceduralTreeEnv(
            num_problems=num_probs,
            depth=tree_params.get("depth", 7),
            difficulties=difficulties,
            p_correct=tree_params.get("p_correct", None),
            percent_correct=tree_params.get("percent_correct", None),
            seed=master_seed,
        )
        samples_per_p = max(8, 256 // num_probs)
        eval_set = build_tree_eval_set(env, samples_per_problem=samples_per_p, seed=env.eval_seed)
        num_steps = env.max_depth

    # Measure empirical active trunk length per trajectory to calibrate buffer replay ratio (~epochs_per_update)
    probe_rng = jax.random.PRNGKey(master_seed + 999)
    probe_obs, probe_st = jax.vmap(env.reset)(jax.random.split(probe_rng, trajectories_per_update))

    def _probe_step(carry, _):
        st, o, k = carry
        k, ak, sk = jax.random.split(k, 3)
        a = jax.vmap(env.sample_base_action)(jax.random.split(ak, trajectories_per_update), st)
        no, nst, _, _, _ = jax.vmap(env.step)(jax.random.split(sk, trajectories_per_update), st, a)
        return (nst, no, k), ~st.done

    _, probe_active = jax.lax.scan(_probe_step, (probe_st, probe_obs, probe_rng), None, length=num_steps)
    avg_active_per_traj = float(jnp.mean(jnp.sum(probe_active, axis=0)))
    est_active_buffer = trajectories_per_update * avg_active_per_traj

    if grad_steps_per_update is not None:
        actual_grad_steps = int(grad_steps_per_update)
    else:
        raw_grad_steps = max(1, round(est_active_buffer * epochs_per_update / minibatch_size))
        # Align to nearest positive multiple of num_steps so Staged FBI executes identical grad steps
        actual_grad_steps = max(1, round(raw_grad_steps / num_steps)) * num_steps

    # Fixed outer updates across all main algorithms (default: 25 for single_dag, 100 for full suites)
    target_num_updates = resolved_updates
    base_total_grad_steps = target_num_updates * actual_grad_steps
    eff_epochs = (actual_grad_steps * minibatch_size) / max(1.0, est_active_buffer)

    print(f"  [Protocol] Fixed Trunk Batch B={trajectories_per_update} traj/update | Avg active steps/traj={avg_active_per_traj:.1f} (~{est_active_buffer:.0f} active states/update)")
    print(f"  [Protocol] Grad steps/update={actual_grad_steps} (minibatch={minibatch_size}, ~{eff_epochs:.2f} buffer epochs) | Updates={target_num_updates} | Total grad steps={base_total_grad_steps:,}")

    flop_assumptions = (
        f"LLM FLOPs Model (units of N params): 2N/inference step (KV-cached), "
        f"2N/target-eval step (FBI), 6N/trained state (fwd+bwd) | "
        f"B={trajectories_per_update} trunk traj/update, {actual_grad_steps} grad steps/update (~{eff_epochs:.1f} epochs)"
    )

    results: Dict[str, Any] = {
        "env_type": env_type,
        "env_title": env_title,
        "eval_true_values": np.asarray(eval_set["exact_v"]),
        "thetas": np.asarray(env.thetas) if hasattr(env, "thetas") else np.array([0.5]),
        "base_total_grad_steps": base_total_grad_steps,
        "flop_assumptions": flop_assumptions,
        "methods": {},
    }

    def _format_algo_name(a: str) -> str:
        if a == "mc":
            return "Standard MC (K=1)"
        if a.startswith("vine_k"):
            return f"Vine-Target (K={a.split('_k')[1]})"
        if a.startswith("fbi_staged_lam"):
            return f"Staged FBI (lam={a.split('_lam')[1]})"
        if a.startswith("fbi_batched_lam") or a.startswith("fbi_lam"):
            return f"Batched FBI (lam={a.split('_lam')[1]})"
        return a

    def _execute_algo_run(algo_key: str, display_name: str, n_updates_run: int, is_compute_matched: bool = False):
        print(f"\n--- Running: {display_name} ---", flush=True)
        print(
            f"  Trunk envs/update: {trajectories_per_update} | Updates: {n_updates_run} | "
            f"Grad steps/update: {actual_grad_steps} | Total grad steps: {n_updates_run * actual_grad_steps:,}",
            flush=True,
        )

        cfg = {
            "GAMMA": 1.0,
            "NUM_ENVS": trajectories_per_update,
            "NUM_STEPS": num_steps,
            "NUM_UPDATES": n_updates_run,
            "GRAD_STEPS_PER_UPDATE": actual_grad_steps,
            "MINIBATCH_SIZE": minibatch_size,
            "LR": lr,
            "WEIGHT_DECAY": weight_decay,
            "WARMUP_RATIO": warmup_ratio,
            "MAX_GRAD_NORM": max_grad_norm,
            "D_MODEL": 64,
            "N_HEADS": 4,
            "N_LAYERS": 2,
            "FINAL_HIDDEN_DIM": 32,
        }

        train_fn = make_train_loop(algo_key, "dag" if is_dag_family else "tree", env, eval_set, cfg)
        vmapped_train = jax.jit(jax.vmap(train_fn))

        rng = jax.random.PRNGKey(master_seed)
        seed_keys = jax.random.split(rng, n_seeds)

        print("  Compiling & Executing on Device...", flush=True)
        t0 = time.perf_counter()
        out = vmapped_train(seed_keys)
        _ = jax.block_until_ready(out["metrics"]["msve"])
        elapsed = time.perf_counter() - t0
        print(f"  Finished {n_seeds} seeds in {elapsed:.2f} seconds ({elapsed / n_updates_run:.4f} s/update).", flush=True)

        msve_arr = np.asarray(out["metrics"]["msve"])  # (n_seeds, n_updates_run)
        adv_cos_arr = np.asarray(out["metrics"]["advantage_cosine_sim"])
        snap_1 = np.asarray(out["snap_1_5"])            # (n_seeds, num_eval_states)
        snap_4 = np.asarray(out["snap_4_5"])

        infer_arr = np.asarray(out["metrics"]["active_infer_steps"])  # (n_seeds, n_updates_run)
        flops_arr = np.asarray(out["metrics"]["llm_flops_per_n"])     # (n_seeds, n_updates_run)

        msve_p_arr = np.asarray(out["metrics"]["msve_per_problem"])
        adv_p_arr = np.asarray(out["metrics"]["adv_cos_per_problem"])
        final_msve_p = msve_p_arr[:, -1, :]
        final_adv_p = adv_p_arr[:, -1, :]

        x_updates = np.arange(1, n_updates_run + 1) * actual_grad_steps
        x_samples = np.mean(np.cumsum(infer_arr, axis=1), axis=0)
        x_llm_flops = np.mean(np.cumsum(flops_arr, axis=1), axis=0)
        x_wallclock = np.linspace(elapsed / n_updates_run, elapsed, n_updates_run)

        print(
            f"  Active Rollout Steps: {x_samples[-1]:,.0f} | Est. LLM Compute: {x_llm_flops[-1]:,.0f} N-FLOPs",
            flush=True,
        )

        results["methods"][display_name] = {
            "x_updates": x_updates,
            "x_samples": x_samples,
            "x_llm_flops": x_llm_flops,
            "x_wallclock": x_wallclock,
            "msve": msve_arr,
            "advantage_cosine_sim": adv_cos_arr,
            "msve_per_problem": final_msve_p,
            "adv_cos_per_problem": final_adv_p,
            "is_compute_matched_mc": is_compute_matched,
            "scatter_snapshots": {
                "1_5": snap_1,
                "4_5": snap_4,
            },
        }

    # 2. Iterate Over Main Algorithms (identical trunk B, identical grad steps, identical replay ratio)
    for algo in algos:
        display_name = _format_algo_name(algo)
        _execute_algo_run(algo, display_name, target_num_updates, is_compute_matched=False)

    # 3. Optional Bonus: Compute-Matched MC Baseline (same B & replay ratio, scaled updates to match max LLM FLOPs)
    if compute_matched_mc and ("mc" in algos) and len(results["methods"]) > 1:
        mc_flops = float(results["methods"][_format_algo_name("mc")]["x_llm_flops"][-1])
        max_flops = max(float(m["x_llm_flops"][-1]) for m in results["methods"].values())
        flops_per_mc_update = mc_flops / float(target_num_updates)
        matched_updates = int(round(max_flops / max(1.0, flops_per_mc_update)))
        if matched_updates > int(round(1.25 * target_num_updates)):
            _execute_algo_run(
                "mc",
                "Standard MC (Compute-Matched)",
                matched_updates,
                is_compute_matched=True,
            )

    # 3. Save Benchmark Data
    pkl_path = os.path.join(save_dir, "benchmark_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[Benchmark] Saved raw benchmark results to: {pkl_path}")

    # 4. Generate Publication Figures & AUC Summary
    plot_results.plot_comparison_curves(results, save_dir, prefix="benchmark")
    plot_results.plot_scatter_true_vs_estimated(results, save_dir, prefix="benchmark")
    plot_results.compute_and_save_auc_summary(results, save_dir, prefix="benchmark")
    plot_results.plot_difficulty_breakdown(results, save_dir, prefix="benchmark")

    return results


def main():
    parser = argparse.ArgumentParser(description="Value Benchmark Suite (GPU Cluster & Local)")
    parser.add_argument("--env_type", type=str, default="single_dag", choices=["single_dag", "dag", "single_tree", "tree"], help="Environment type: single_dag, dag, single_tree, tree")
    parser.add_argument("--total_transitions", type=int, default=150000, help="(Deprecated/unused: updates are now fixed via --num_updates)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--master_seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--algos", type=str, default="mc,vine_k2,vine_k4,fbi_lam0.0,fbi_lam0.8,fbi_staged_lam0.0,fbi_staged_lam0.8", help="Comma-separated algos")
    parser.add_argument("--out_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Explicit save directory (overrides --out_dir/--sweep_id)")
    parser.add_argument("--sweep_id", type=str, default=None, help="Optional sweep / run identifier")
    
    # Shared difficulty / problem params
    parser.add_argument("--num_problems", type=int, default=None, help="Number of problems (defaults: 1 for single_*, 10-20 for spectrum)")
    parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty theta in [0, 1] for single_* (default: 0.5)")

    # DAG specific params
    parser.add_argument("--num_milestones", type=int, default=None, help="DAG: Number of milestones (defaults: 8 for single_dag, 6 for dag)")
    parser.add_argument("--max_steps", type=int, default=None, help="DAG: Max steps (defaults: 32 for single_dag, 24 for dag)")
    parser.add_argument("--num_actions", type=int, default=8, help="DAG: Number of discrete actions")

    # Tree specific params
    parser.add_argument("--depth", type=int, default=7, help="Tree: Depth of binary tree (default: 7)")
    parser.add_argument("--p_correct", type=float, default=None, help="Tree: Base policy transition probability (default: derived from difficulty theta)")
    parser.add_argument("--percent_correct", type=float, default=None, help="Tree: Percentage of correct leaves (default: derived from difficulty theta)")

    # RLVR Training & Fair Comparison params (fixed trunk batch & replay ratio across all algos)
    parser.add_argument("--trajectories_per_update", type=int, default=64, help="Fixed trunk trajectories B collected per update across all algorithms (default: 64)")
    parser.add_argument("--epochs_per_update", type=int, default=2, help="Target critic epochs over active trunk buffer per update (default: 2)")
    parser.add_argument("--num_updates", type=int, default=None, help="Number of updates for main equal-gradient-steps comparison (default: 25 for single_dag, 100 otherwise)")
    parser.add_argument("--grad_steps_per_update", type=int, default=None, help="Override gradient steps per update (default: calibrated to ~epochs_per_update)")
    parser.add_argument("--no_compute_matched_mc", action="store_true", help="Disable bonus compute-matched Standard MC baseline")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate for warmup-cosine schedule (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--warmup_ratio", type=float, default=0.10, help="Warmup fraction of total gradient steps (default: 0.10)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm clipping (default: 1.0)")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Minibatch size for gradient steps (default: 64)")

    args = parser.parse_args()

    if args.save_dir:
        save_dir = args.save_dir
    elif args.sweep_id:
        save_dir = os.path.join(args.out_dir, f"{args.env_type}_{args.sweep_id}")
    else:
        save_dir = os.path.join(args.out_dir, f"{args.env_type}_full")

    algo_list = [a.strip() for a in args.algos.split(",") if a.strip()]

    # Set smart defaults based on env_type
    is_single = args.env_type in ("single_dag", "single_tree")
    num_prob = args.num_problems if args.num_problems is not None else (1 if is_single else 10)
    num_mile = args.num_milestones if args.num_milestones is not None else (8 if is_single else 6)
    max_st = args.max_steps if args.max_steps is not None else (32 if is_single else 24)

    dag_params = {
        "num_problems": num_prob,
        "num_milestones": num_mile,
        "max_steps": max_st,
        "num_actions": args.num_actions,
        "difficulty": args.difficulty,
    }
    tree_params = {
        "num_problems": num_prob,
        "depth": args.depth,
        "difficulty": args.difficulty,
        "p_correct": args.p_correct,
        "percent_correct": args.percent_correct,
    }

    run_benchmark(
        env_type=args.env_type,
        total_transitions_budget=args.total_transitions,
        n_seeds=args.n_seeds,
        master_seed=args.master_seed,
        algos=algo_list,
        save_dir=save_dir,
        dag_params=dag_params,
        tree_params=tree_params,
        trajectories_per_update=args.trajectories_per_update,
        epochs_per_update=args.epochs_per_update,
        num_updates=args.num_updates,
        grad_steps_per_update=args.grad_steps_per_update,
        compute_matched_mc=not args.no_compute_matched_mc,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        minibatch_size=args.minibatch_size,
    )


if __name__ == "__main__":
    main()
