"""Frozen-policy Monte Carlo references for program tasks; never training targets."""
import jax
import jax.numpy as jnp
import numpy as np


def reference_values(env,states,key,k=128,batch_size=128):
    n=states.done.shape[0];total=n*k
    def chunk(chunk_index):
        indices=chunk_index*batch_size+jnp.arange(batch_size)
        st=jax.tree_util.tree_map(lambda x:x[jnp.minimum(indices//k,n-1)],states)
        st=st._replace(done=st.done|(indices>=total))
        keys=jax.vmap(lambda i:jax.random.fold_in(key,i))(indices)
        def condition(c):return (c[0]<env.max_steps)&(~c[1].done).any()
        def step(c):
            t,st,ret,discount,keys=c
            keys,ak,sk=jnp.moveaxis(jax.vmap(lambda q:jax.random.split(q,3))(keys),1,0)
            action=jax.vmap(env.sample_base_action)(ak,st)
            _,ns,r,d,_=jax.vmap(env.step)(sk,st,action)
            ret+=discount*r;discount*=env.gamma*(1-d)
            return t+1,ns,ret,discount,keys
        _,_,ret,_,_=jax.lax.while_loop(condition,step,(jnp.array(0),st,jnp.zeros(batch_size),jnp.ones(batch_size),keys))
        return ret
    returns=jax.lax.map(chunk,jnp.arange((total+batch_size-1)//batch_size)).reshape(-1)[:total].reshape(n,k)
    return returns.mean(-1),returns.std(-1,ddof=1)/jnp.sqrt(k)


def build_program_eval_set(env,samples_per_problem=2,reference_k=128,action_samples=4,seed=12345,max_eval_problems=32):
    if samples_per_problem<1 or reference_k<2 or action_samples<1:raise ValueError('positive sample counts and reference_k >=2 required')
    count=max(4,samples_per_problem)
    # Number tasks can expose almost a million pairs. Evaluation uses a fixed,
    # representative-size subset, rather than constructing one rollout per pair.
    rng=np.random.default_rng(seed)
    selected_problems=(env.evaluation_problem_ids(max_eval_problems,seed) if hasattr(env,'evaluation_problem_ids')
                       else np.sort(rng.choice(env.num_problems,size=min(max_eval_problems,env.num_problems),replace=False)))
    problem_ids=jnp.repeat(jnp.asarray(selected_problems,jnp.int32),count)
    _,states=jax.vmap(env.reset_problem)(problem_ids)
    def step(carry,_):
        st,key=carry;key,ak,sk=jax.random.split(key,3)
        actions=jax.vmap(env.sample_base_action)(jax.random.split(ak,len(problem_ids)),st)
        _,ns,_,_,_=jax.vmap(env.step)(jax.random.split(sk,len(problem_ids)),st,actions)
        return (ns,key),st
    _,pool=jax.jit(lambda key:jax.lax.scan(step,(states,key),None,length=env.max_steps))(jax.random.PRNGKey(seed))
    pool=jax.tree_util.tree_map(lambda x:x.reshape((-1,)+x.shape[2:]),pool)
    rng=np.random.default_rng(seed);selected=[]
    for p in selected_problems:
        choices=np.flatnonzero((np.asarray(pool.prompt_idx)==p)&~np.asarray(pool.done))
        selected.extend(rng.choice(choices,size=min(samples_per_problem,len(choices)),replace=False).tolist())
    states=jax.tree_util.tree_map(lambda x:x[jnp.asarray(selected)],pool)
    n=len(selected);key=jax.random.PRNGKey(seed+1)
    action_keys=jax.random.split(key,action_samples*n).reshape(action_samples,n,2)
    actions=jax.vmap(lambda keys:jax.vmap(env.sample_base_action)(keys,states))(action_keys)
    def next_states(a,keys):
        return jax.vmap(env.step)(keys,states,a)
    keys=jax.random.split(jax.random.PRNGKey(seed+2),action_samples*n).reshape(action_samples,n,2)
    next_obs,next_state,reward,done,_=jax.vmap(next_states)(actions,keys)
    combined=jax.tree_util.tree_map(lambda x,y:jnp.concatenate([x,y.reshape((-1,)+y.shape[2:])]),states,next_state)
    values,error=jax.jit(lambda st:reference_values(env,st,jax.random.PRNGKey(seed+3),reference_k))(combined)
    v=values[:n];nv=values[n:].reshape(action_samples,n)
    advantage=reward+env.gamma*(1-done)*nv-v[None,:]
    return dict(eval_obs=states.history_tokens,eval_state=states,exact_v=v,true_adv=advantage,
                all_eval_tokens=combined.history_tokens,r_all=reward,d_all=done,
                next_obs_all=next_obs,next_states_all=next_state,num_actions=action_samples,
                prompts=jnp.searchsorted(jnp.asarray(selected_problems),states.prompt_idx),num_problems=len(selected_problems),
                evaluation_problem_ids=jnp.asarray(selected_problems),
                reference_is_exact=False,reference_standard_errors=error,
                reference_k=reference_k,evaluation_actions=actions,
                reference_description='Independent MC continuation estimates under the frozen reference policy; sampled policy actions')


def plot_program_critic_results(results,save_dir,prefix='benchmark'):
    """Explicitly label approximate references instead of claiming exact MSVE."""
    import csv
    from pathlib import Path
    import matplotlib.pyplot as plt
    out=Path(save_dir);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    rows=[]
    for name,m in results['methods'].items():
        for col,(axis,label) in enumerate((('x_updates','Gradient steps'),('x_samples','Generated transitions'),('x_llm_flops','Estimated compute (N-FLOPs)'))):
            for row,key in enumerate(('msve','advantage_cosine_sim')):
                y=np.asarray(m[key]);mean=y.mean(0)
                error=y.std(0,ddof=1)/np.sqrt(len(y)) if len(y)>1 else np.zeros_like(mean)
                axes[row,col].plot(m[axis],mean,label=name)
                axes[row,col].fill_between(m[axis],mean-error,mean+error,alpha=.12)
                axes[row,col].set_xlabel(label);axes[row,col].grid(alpha=.2)
        rows.append(dict(method=name,final_mc_reference_mse=float(np.asarray(m['msve'])[:,-1].mean()),
                         final_sampled_advantage_cosine=float(np.asarray(m['advantage_cosine_sim'])[:,-1].mean()),
                         generated_transitions=float(m['x_samples'][-1]),estimated_compute=float(m['x_llm_flops'][-1])))
    for ax in axes[0]:ax.set_ylabel('MSE to independent MC reference ↓')
    for ax in axes[1]:ax.set_ylabel('MC-reference advantage cosine ↑')
    fig.suptitle('Program critic comparison — approximate fixed-policy references, not exact values')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.12,1,.94))
    for ext in ('png','pdf'):fig.savefig(out/f'{prefix}_comparison_curves.{ext}',dpi=180)
    plt.close(fig)
    with (out/f'{prefix}_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import json
    (out/'evaluation_reference.json').write_text(json.dumps(dict(
        exact=False,description=results.get('evaluation_reference_description'),
        continuations_per_state=results.get('evaluation_reference_k'),
        mean_value_reference_variance=float(np.mean(np.asarray(results['evaluation_reference_standard_errors'])**2)),
        caveat='Reference MSE contains MC reference noise; action advantages use sampled reference-policy actions.'),indent=2)+'\n')
