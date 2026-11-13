"""Split actor/critic PPO on controlled DAGs. Actor optimization is trunk-only."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from transformer import TransformerCritic, TransformerPolicy
from envs.dag_mdp import make_dag_mdp
from vine_mc import prefix_values
from dirichlet_value import bilinear_double_sampled_loss


class TrunkBatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    active: jax.Array
    log_prob: jax.Array
    value: jax.Array
    next_value: jax.Array
    next_obs: jax.Array


def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(mask.sum(), 1)


def action_log_prob(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]


def lambda_returns(reward, done, value, next_value, active, gamma, lmbda):
    """Complete episodes: task-horizon termination is not a time-limit truncation."""
    delta = reward + gamma * (1-done) * next_value - value
    def backward(next_adv, data):
        residual, terminal, valid = data
        adv = jnp.where(valid, residual + gamma*lmbda*(1-terminal)*next_adv, 0.)
        return adv, adv
    _, adv = jax.lax.scan(backward, jnp.zeros(reward.shape[1]), (delta,done,active), reverse=True)
    return jax.lax.stop_gradient(adv), jax.lax.stop_gradient(value+adv)


def actor_loss(params, model, obs, action, old_log_prob, advantage, active,
               clip_eps=0.2, entropy_coef=0.):
    logits = model.apply(params, obs)
    log_prob = action_log_prob(logits, action)
    ratio = jnp.exp(log_prob-old_log_prob)
    objective = jnp.minimum(ratio*advantage, jnp.clip(ratio,1-clip_eps,1+clip_eps)*advantage)
    log_pi = jax.nn.log_softmax(logits)
    entropy = -(jnp.exp(log_pi)*log_pi).sum(-1)
    loss = -masked_mean(objective,active)-entropy_coef*masked_mean(entropy,active)
    aux = dict(actor_loss=loss, entropy=masked_mean(entropy,active),
               approx_kl=masked_mean((ratio-1)-(log_prob-old_log_prob),active),
               clip_fraction=masked_mean((jnp.abs(ratio-1)>clip_eps).astype(float),active))
    return loss, aux


def make_train(env, config):
    """Jittable, vmappable full-episode PPO, following separate critic/actor phases."""
    cfg = dict(num_updates=100, num_envs=64, actor_epochs=4, critic_epochs=4,
               minibatch_size=64, actor_lr=3e-5, critic_lr=5e-4, gamma=1.,
               actor_lambda=1., critic_lambda=0.8, clip_eps=0.2, value_clip=0.2,
               entropy_coef=0., max_grad_norm=1., critic_method="td", k=4,
               include_trunk=True, bc_updates=0, bc_lr=3e-4, vine_batch_size=256, eval_episodes=128,
               d_model=64, n_heads=4, n_layers=2, final_hidden_dim=32)
    cfg.update(config)
    for name in ("num_updates","num_envs","actor_epochs","critic_epochs","minibatch_size","k","eval_episodes","vine_batch_size"):
        if cfg[name] < 1: raise ValueError(f"{name} must be positive")
    for name in ("gamma","actor_lambda","critic_lambda"):
        if not 0 <= cfg[name] <= 1: raise ValueError(f"{name} must be in [0,1]")
    if cfg["bc_updates"] < 0: raise ValueError("bc_updates must be nonnegative")
    if cfg["critic_method"] not in ("mc","td","fbi","vine","vine_mc","e_min"):
        raise ValueError("critic_method must be mc, td, fbi, vine, vine_mc, or e_min")
    for name in ("actor_lr", "bc_lr", "critic_lr", "max_grad_norm", "clip_eps"):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0: raise ValueError(f"{name} must be finite and positive")
    if cfg["value_clip"] < 0 or cfg["entropy_coef"] < 0:
        raise ValueError("value_clip and entropy_coef must be nonnegative")
    if cfg["critic_method"] == "vine_mc":
        cfg["actor_lambda"] = 0.
    if cfg["critic_method"] == "e_min":
        cfg["k"] = 2  # One trunk transition and one independent probe.
    T, B, K = env.max_steps, cfg["num_envs"], cfg["k"]
    net = dict(vocab_size=env.vocab_size,max_len=env.max_seq_len,
               d_model=cfg["d_model"],n_heads=cfg["n_heads"],n_layers=cfg["n_layers"],
               final_hidden_dim=cfg["final_hidden_dim"])
    actor = TransformerPolicy(num_actions=env.num_actions,**net)
    critic = TransformerCritic(add_bos=False,**net)
    adam = lambda lr: optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]),optax.adam(lr,eps=1e-5))
    mb = cfg["minibatch_size"]
    n_mb = (T*B+mb-1)//mb
    padded_size = n_mb*mb

    def collect(actor_params, critic_params, key, reference=False, count=B):
        key, rk = jax.random.split(key)
        obs, states = jax.vmap(env.reset)(jax.random.split(rk,count))
        def step(carry,_):
            obs,states,key = carry
            key,ak,sk = jax.random.split(key,3)
            logits = actor.apply(actor_params,obs)
            if reference:
                action = jax.vmap(env.sample_reference_action)(jax.random.split(ak,count),states)
            else:
                action = jax.random.categorical(ak,logits).astype(jnp.int32)
            active = ~states.done
            value = jnp.zeros(count) if cfg["critic_method"] == "vine_mc" else critic.apply(critic_params,obs)
            no,ns,r,d,_ = jax.vmap(env.step)(jax.random.split(sk,count),states,action)
            nv = jnp.zeros(count) if cfg["critic_method"] == "vine_mc" else critic.apply(critic_params,no)
            batch = TrunkBatch(obs,action,r,d,active,action_log_prob(logits,action),value,nv,no)
            return (no,ns,key),(batch,states)
        _,(batch,states) = jax.lax.scan(step,(obs,states,key),None,length=T)
        return jax.tree_util.tree_map(jax.lax.stop_gradient,(batch,states))

    def evaluate(ap,cp,key):
        batch,_ = collect(ap,cp,key,count=cfg["eval_episodes"])
        return batch.reward.sum()/cfg["eval_episodes"]

    def critic_targets(batch, states, ap, cp, key):
        method = cfg["critic_method"]
        if method == "e_min":
            # Actual full suffix returns, without value cancellation or a
            # bootstrapped horizon. The terminal continuation is exactly zero.
            def backward(next_return,data):
                reward,done,active = data
                ret = jnp.where(active,reward+cfg["gamma"]*(1-done)*next_return,0.)
                return ret,ret
            _,target = jax.lax.scan(backward,jnp.zeros(B),
                                   (batch.reward,batch.done,batch.active),reverse=True)
            return jax.lax.stop_gradient(target),jnp.array(0.)
        if method in ("mc","td") or (method=="fbi" and cfg["include_trunk"] and K==1):
            lam = 1. if method == "mc" else cfg["critic_lambda"]
            return lambda_returns(batch.reward,batch.done,batch.value,batch.next_value,batch.active,cfg["gamma"],lam)[1],jnp.array(0.)
        fresh = K-int(cfg["include_trunk"]) if method=="fbi" else K
        flat_states = jax.tree_util.tree_map(lambda x:x.reshape((-1,)+x.shape[2:]),states)
        fork_states = jax.tree_util.tree_map(lambda x:jnp.repeat(x,fresh,axis=0),flat_states)
        fork_obs = fork_states.history_tokens
        n_forks = T*B*fresh
        def fork_step(carry,_):
            st,obs,ret,discount,steps,key = carry
            key,ak,sk = jax.random.split(key,3)
            action = jax.random.categorical(ak,actor.apply(ap,obs)).astype(jnp.int32)
            active = ~st.done
            no,ns,r,d,_ = jax.vmap(env.step)(jax.random.split(sk,n_forks),st,action)
            ret += discount*r
            discount *= cfg["gamma"]*(1-d)
            return (ns,no,ret,discount,steps+active.sum(),key),None
        length = 1 if method=="fbi" else T
        (ns,no,ret,discount,steps,_),_ = jax.lax.scan(
            fork_step,(fork_states,fork_obs,jnp.zeros(n_forks),jnp.ones(n_forks),jnp.array(0),key),None,length=length)
        if method=="fbi": ret += discount*critic.apply(cp,no)
        branch = ret.reshape(T,B,fresh)
        if method=="vine":
            target = branch.mean(-1)
        else:
            if cfg["include_trunk"]:
                trunk_backup = batch.reward+cfg["gamma"]*(1-batch.done)*batch.next_value
                branch = jnp.concatenate([trunk_backup[...,None],branch],axis=-1)
            mean = branch.mean(-1)
            # Branched residual traces are critic targets, never action advantages.
            residual = mean-batch.value
            def backward(next_adv,data):
                delta,done,active = data
                adv = jnp.where(active,delta+cfg["gamma"]*cfg["critic_lambda"]*(1-done)*next_adv,0.)
                return adv,adv
            _,adv = jax.lax.scan(backward,jnp.zeros(B),(residual,batch.done,batch.active),reverse=True)
            target = batch.value+adv
        return jax.lax.stop_gradient(target),steps.astype(float)

    def sample_probe(ap, states, key):
        # Sampling is under the frozen actor, independently of the trunk suffix.
        flat = jax.tree_util.tree_map(lambda x:x.reshape((-1,)+x.shape[2:]),states)
        ak,sk = jax.random.split(key)
        action = jax.random.categorical(ak,actor.apply(ap,flat.history_tokens)).astype(jnp.int32)
        no,_,reward,done,_ = jax.vmap(env.step)(jax.random.split(sk,T*B),flat,action)
        return jax.tree_util.tree_map(jax.lax.stop_gradient,(no,reward,done))

    def flatten(x): return x.reshape((-1,)+x.shape[2:])

    def optimize(ts, data, epochs, key, loss_fn):
        # Shuffle each trunk entry once per epoch; extra padding has mask zero.
        data = jax.tree_util.tree_map(lambda x:jnp.pad(x,((0,padded_size-T*B),)+((0,0),)*(x.ndim-1)),data)
        def epoch(carry,_):
            ts,key = carry
            key,pk = jax.random.split(key)
            indices = jax.random.permutation(pk,padded_size).reshape(n_mb,mb)
            def minibatch(ts,idx):
                sample = jax.tree_util.tree_map(lambda x:x[idx],data)
                (loss,aux),grad = jax.value_and_grad(loss_fn,has_aux=True)(ts.params,sample)
                # An entirely padded minibatch must not advance Adam momentum.
                ts = jax.lax.cond(sample[-1].any(),lambda s:s.apply_gradients(grads=grad),lambda s:s,ts)
                return ts,aux
            ts,aux = jax.lax.scan(minibatch,ts,indices)
            return (ts,key),aux
        (ts,_),aux = jax.lax.scan(epoch,(ts,key),None,length=epochs)
        return ts,jax.tree_util.tree_map(jnp.mean,aux)

    def train(key):
        key,ak,ck,rk = jax.random.split(key,4)
        obs,_ = env.reset(rk)
        actor_state = TrainState.create(apply_fn=actor.apply,params=actor.init(ak,obs[None]),tx=adam(cfg["bc_lr"]))
        critic_state = TrainState.create(apply_fn=critic.apply,params=critic.init(ck,obs[None]),tx=adam(cfg["critic_lr"]))
        # Optional supervised initialization from fixed-base trunk trajectories.
        def bc_step(carry,_):
            state,key = carry
            key,rk,ok = jax.random.split(key,3)
            batch,_ = collect(state.params,critic_state.params,rk,reference=True)
            data = (flatten(batch.obs),flatten(batch.action),flatten(batch.active))
            def loss_fn(params,sample):
                obs,action,mask = sample
                loss = -masked_mean(action_log_prob(actor.apply(params,obs),action),mask)
                return loss,dict(bc_loss=loss)
            state,_ = optimize(state,data,1,ok,loss_fn)
            return (state,key),None
        (actor_state,key),_ = jax.lax.scan(bc_step,(actor_state,key),None,length=cfg["bc_updates"])
        actor_state = TrainState.create(apply_fn=actor.apply,params=actor_state.params,tx=adam(cfg["actor_lr"]))
        initial_actor_params = actor_state.params
        key,ek = jax.random.split(key)
        initial_success = evaluate(actor_state.params,critic_state.params,ek)

        def update(carry,idx):
            ac,cr,key = carry
            key,rk,bk,ck,ak = jax.random.split(key,5)
            batch,states = collect(ac.params,cr.params,rk)
            if cfg["critic_method"] == "vine_mc":
                target,aux_steps = prefix_values(env,actor,ac.params,states,bk,K,cfg["vine_batch_size"],cfg["gamma"])
                next_values = jnp.concatenate([target[1:],jnp.zeros_like(target[:1])],axis=0)
                advantage = jax.lax.stop_gradient(jnp.where(batch.active,
                    batch.reward+cfg["gamma"]*(1-batch.done)*next_values-target,0.))
            else:
                advantage,_ = lambda_returns(batch.reward,batch.done,batch.value,batch.next_value,batch.active,cfg["gamma"],cfg["actor_lambda"])
                target,aux_steps = critic_targets(batch,states,ac.params,cr.params,bk)
            # Freeze the entire actor batch before any critic optimization.
            mean = masked_mean(advantage,batch.active)
            std = jnp.sqrt(masked_mean((advantage-mean)**2,batch.active)+1e-8)
            normalized_adv = jax.lax.stop_gradient(jnp.where(batch.active,(advantage-mean)/std,0.))
            critic_data = tuple(map(flatten,(batch.obs,batch.value,target,batch.active)))
            def closs(params,sample):
                obs,old,target,active = sample
                pred = critic.apply(params,obs)
                error = (pred-target)**2
                if cfg["value_clip"] > 0:
                    clipped = old+jnp.clip(pred-old,-cfg["value_clip"],cfg["value_clip"])
                    error = jnp.maximum(error,(clipped-target)**2)
                loss = 0.5*masked_mean(error,active)
                return loss,dict(critic_loss=loss)
            if cfg["critic_method"] == "vine_mc":
                cmetrics = dict(critic_loss=jnp.array(0.))
            elif cfg["critic_method"] == "e_min":
                probe_obs,probe_reward,probe_done = sample_probe(ac.params,states,bk)
                aux_steps = batch.active.sum().astype(float)
                energy_data = (flatten(batch.obs),flatten(batch.next_obs),probe_obs,
                               flatten(target),flatten(batch.reward),flatten(batch.done),
                               probe_reward,probe_done,flatten(batch.active))
                def eloss(params,sample):
                    obs,trunk_obs,probe_obs,G,reward,done,pr,pd,active = sample
                    # All predictions use current critic parameters. Coefficients
                    # are detached inside the exact product-rule surrogate.
                    values = critic.apply(params,jnp.concatenate([obs,trunk_obs,probe_obs],axis=0))
                    v,v1,v2 = jnp.split(values,3)
                    loss = bilinear_double_sampled_loss(v,v1,v2,G,reward,done,pr,pd,active,cfg["gamma"])
                    # optimize pads TB to a full minibatch multiple; undo that
                    # fixed dilution, never divide by random active occupancy.
                    loss *= padded_size/(T*B)
                    return loss,dict(critic_loss=loss)
                cr,cmetrics = optimize(cr,energy_data,cfg["critic_epochs"],ck,eloss)
            else:
                cr,cmetrics = optimize(cr,critic_data,cfg["critic_epochs"],ck,closs)
            actor_data = tuple(map(flatten,(batch.obs,batch.action,batch.log_prob,normalized_adv,batch.active)))
            def aloss(params,sample):
                obs,action,old_lp,adv,active = sample
                return actor_loss(params,actor,obs,action,old_lp,adv,active,cfg["clip_eps"],cfg["entropy_coef"])
            ac,ametrics = optimize(ac,actor_data,cfg["actor_epochs"],ak,aloss)
            metrics = dict(**cmetrics,**ametrics,success_rate=batch.reward.sum()/B,
                           trunk_transitions=batch.active.sum(),aux_transitions=aux_steps,
                           actor_data_transitions=batch.active.sum(),
                           critic_target_mean=masked_mean(target,batch.active),
                           raw_advantage_mean=mean)
            return (ac,cr,key),metrics
        (actor_state,critic_state,key),metrics = jax.lax.scan(update,(actor_state,critic_state,key),jnp.arange(cfg["num_updates"]))
        final_success = evaluate(actor_state.params,critic_state.params,jax.random.fold_in(key,777))
        return dict(metrics=metrics,initial_success=initial_success,final_success=final_success,
                    initial_actor_params=initial_actor_params,actor_params=actor_state.params,
                    critic_params=critic_state.params)
    return train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env_type",choices=("dag","dag_quality","dag_approach"),default="dag_approach")
    p.add_argument("--num_problems",type=int,default=20)
    p.add_argument("--num_milestones",type=int,default=12)
    p.add_argument("--max_steps",type=int,default=40)
    p.add_argument("--num_actions",type=int,default=8)
    p.add_argument("--n_seeds",type=int,default=5)
    p.add_argument("--master_seed",type=int,default=42)
    p.add_argument("--critic_method",choices=("mc","td","fbi","vine","vine_mc","e_min"),default="td")
    p.add_argument("--fresh_branches",action="store_true",help="FBI uses K fresh transitions instead of trunk + K-1")
    for name,default in (("num_updates",100),("num_envs",64),("actor_epochs",4),("critic_epochs",4),
                         ("minibatch_size",64),("vine_batch_size",256),("k",4),("bc_updates",0),("eval_episodes",128),
                         ("d_model",64),("n_heads",4),("n_layers",2),("final_hidden_dim",32)):
        p.add_argument("--"+name,type=int,default=default)
    for name,default in (("actor_lr",3e-5),("bc_lr",3e-4),("critic_lr",5e-4),("gamma",1.),("actor_lambda",1.),
                         ("critic_lambda",0.8),("clip_eps",0.2),("value_clip",0.2),("entropy_coef",0.),("max_grad_norm",1.)):
        p.add_argument("--"+name,type=float,default=default)
    p.add_argument("--num_approaches",type=int,default=3)
    p.add_argument("--error_rate",type=float,default=1.)
    p.add_argument("--recovery_rate",type=float,default=1.)
    p.add_argument("--shortcut_weight",type=float,default=.04)
    p.add_argument("--wait_weight",type=float,default=.10)
    p.add_argument("--quality_levels",type=int,default=7)
    p.add_argument("--quality_change_rate",type=float,default=.35)
    p.add_argument("--quality_up_bias",type=float,default=.6)
    p.add_argument("--quality_jump",type=int,default=1)
    p.add_argument("--absorbing_dead",action="store_true")
    p.add_argument("--save_dir",default="results/dag_ppo")
    args = p.parse_args()
    if args.n_seeds < 1: p.error("n_seeds must be positive")
    env_keys = {"dag": (), "dag_quality": ("quality_levels", "quality_change_rate", "quality_up_bias", "quality_jump", "absorbing_dead"),
                "dag_approach": ("num_approaches", "error_rate", "recovery_rate", "shortcut_weight", "wait_weight")}[args.env_type]
    env_kwargs = {k:getattr(args,k) for k in env_keys}
    env = make_dag_mdp(args.env_type,num_problems=args.num_problems,num_milestones=args.num_milestones,
                       max_steps=args.max_steps,num_actions=args.num_actions,seed=args.master_seed,
                       variable_depth=args.num_problems>1,**env_kwargs)
    if args.critic_method == "vine_mc": args.actor_lambda = 0.
    if args.critic_method == "e_min": args.k = 2
    config = vars(args).copy();config["include_trunk"] = not args.fresh_branches
    if args.critic_method == "e_min":
        config["critic_estimator"] = "bilinear_double_sampled_trunk_probe_v1"
        config["critic_objective"] = "0.5 * <V-v, (I-gamma*P)(V-v)>_D"
        config["critic_loss_is_surrogate"] = True
        config["critic_value_clipping_applied"] = False
    train = jax.jit(make_train(env,config))
    outdir = Path(args.save_dir);outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/"config.json").open("w") as f:json.dump(config,f,indent=2)
    with (outdir/"environment.json").open("w") as f:
        json.dump(dict(version="dag_mdp_v1",family=args.env_type,
                       base_environment=getattr(env.base_env,"config",{}),
                       exact_base_values_used=False,actor_observation="history_tokens_only"),f,indent=2)
    summaries, all_metrics = [],[]
    for seed in range(args.n_seeds):
        print(f"PPO seed {seed+1}/{args.n_seeds}: {args.env_type}, critic={args.critic_method}",flush=True)
        result = jax.device_get(train(jax.random.fold_in(jax.random.PRNGKey(args.master_seed),seed)))
        metrics = result["metrics"]
        all_metrics.append(metrics)
        with (outdir/f"metrics_seed{seed}.csv").open("w") as f:
            writer = csv.writer(f);writer.writerow(["update",*metrics])
            for i in range(args.num_updates):writer.writerow([i+1,*[float(v[i]) for v in metrics.values()]])
        for name in (("actor_params",) if args.critic_method == "vine_mc" else ("actor_params","critic_params")):
            (outdir/f"{name}_seed{seed}.msgpack").write_bytes(serialization.to_bytes(result[name]))
        summaries.append(dict(seed=seed,initial_success=float(result["initial_success"]),final_success=float(result["final_success"])))
        print(summaries[-1],flush=True)
    with (outdir/"summary.json").open("w") as f:json.dump(summaries,f,indent=2)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,("success_rate","actor_loss","critic_loss")):
        values=np.stack([m[name] for m in all_metrics])
        mean=values.mean(0);se=values.std(0)/np.sqrt(len(values));x=np.arange(1,args.num_updates+1)
        ax.plot(x,mean);ax.fill_between(x,mean-se,mean+se,alpha=.2)
        ax.set_xlabel("PPO update");ax.set_ylabel(name.replace("_"," "));ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(outdir/"learning_curves.png",dpi=160);plt.close(fig)


if __name__=="__main__":main()
