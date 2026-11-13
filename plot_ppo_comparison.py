"""Plot PPO method comparisons and available FBI K/lambda ablations.

Reads config.json, metrics_seed*.csv and summary.json from run subdirectories.
Never trains or submits jobs. Use a single environment's result directory.
Example:
  python3 plot_ppo_comparison.py --results_dir results/ppo_comparison_tuned
Outputs default to RESULTS_DIR/comparison_plots (override with --save_dir).
Main curves use updates, trunk samples and total generated samples. FBI panels
hold lambda fixed to compare K, then hold K fixed to compare lambda. Only
completed runs are included; the script never invents unrun combinations.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sem(a):
    return a.std(axis=0, ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.zeros(a.shape[1:])


def describe(cfg):
    method=cfg['critic_method']
    k=cfg.get('k',1); lam=cfg.get('critic_lambda',.8)
    if method=='fbi':
        branches='trunk included' if cfg.get('include_trunk',True) else 'fresh branches'
        return f'FBI K={k}, λ={lam:g} ({branches})'
    if method=='td': return f'TD λ={lam:g}'
    if method=='vine': return f'Vine-target K={k}'
    if method=='vine_mc': return f'MC VinePPO K={k}'
    return {'mc':'MC','e_min':'E_min (bilinear double-sampled)'}.get(method,method)


def load(root):
    runs=[]; skipped=[]
    for config_path in sorted(root.rglob('config.json')):
        folder=config_path.parent
        try:
            cfg=json.loads(config_path.read_text())
            if 'critic_method' not in cfg: continue
            paths=sorted(folder.glob('metrics_seed*.csv'))
            if not paths: raise ValueError('no metric files')
            summaries=json.loads((folder/'summary.json').read_text())
            evaluations={int(s['seed']):s for s in summaries}
            expected=set(range(cfg['n_seeds']))
            if set(evaluations)!=expected or len(summaries)!=len(expected): raise ValueError('incomplete evaluation seeds')
            seeds={int(p.stem.removeprefix('metrics_seed')):p for p in paths}
            if set(seeds)!=expected: raise ValueError('incomplete metric seeds')
            data=[];aucs=[];final=[];initial=[]
            for seed in sorted(seeds):
                a=np.atleast_1d(np.genfromtxt(seeds[seed],delimiter=',',names=True))
                if not np.array_equal(a['update'],np.arange(1,cfg['num_updates']+1)):
                    raise ValueError('incomplete update history')
                y=a['success_rate'];x=a['update']
                if len(x)<2 or not np.isfinite(y).all(): raise ValueError('invalid learning curve')
                integrate=getattr(np,'trapezoid',None) or np.trapz
                aucs.append(float(integrate(y,x)/(x[-1]-x[0])))
                final.append(float(evaluations[seed]['final_success']))
                initial.append(float(evaluations[seed]['initial_success']))
                if not np.isfinite([final[-1],initial[-1]]).all(): raise ValueError('nonfinite evaluation')
                data.append(dict(update=x,success_rate=y,
                                 trunk_samples=np.cumsum(a['trunk_transitions']),
                                 total_samples=np.cumsum(a['trunk_transitions']+a['aux_transitions'])))
            runs.append(dict(path=str(folder.relative_to(root)),label=describe(cfg),cfg=cfg,data=data,
                             auc=np.asarray(aucs),final=np.asarray(final),initial=np.asarray(initial),
                             heldout=np.asarray([evaluations[s]['final_heldout_success'] for s in sorted(seeds)])
                             if all('final_heldout_success' in evaluations[s] for s in seeds) else None))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            skipped.append(dict(path=str(folder),reason=str(exc)))
    return runs,skipped


def band(ax,run,axis,label=None):
    data=run['data']
    lo=max(d[axis][0] for d in data);hi=min(d[axis][-1] for d in data)
    if lo>=hi: return
    grid=np.linspace(lo,hi,200)
    # Align seeds only over observed support; no extrapolation or smoothing.
    y=np.asarray([np.interp(grid,d[axis],d['success_rate']) for d in data])
    line,=ax.plot(grid,y.mean(0),label=label or run['label'])
    error=sem(y)
    ax.fill_between(grid,y.mean(0)-error,y.mean(0)+error,color=line.get_color(),alpha=.13)


def save(fig,dest,stem):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(dest/f'{stem}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_dir',required=True)
    p.add_argument('--save_dir')
    args=p.parse_args()
    root=Path(args.results_dir).resolve();dest=Path(args.save_dir).resolve() if args.save_dir else root/'comparison_plots'
    runs,skipped=load(root)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'collection_status.json').write_text(json.dumps(dict(complete=len(runs),excluded=skipped),indent=2)+'\n')
    if not runs: raise SystemExit(f'No complete PPO runs. See {dest}/collection_status.json')
    if len({r['cfg'].get('env_type') for r in runs})!=1:
        raise SystemExit('Choose a single environment directory; refusing to mix different DAG families.')
    # Record mismatches rather than silently asserting that runs were controlled.
    shared=('num_problems','num_milestones','max_steps','num_actions','master_seed','n_seeds',
            'num_updates','num_envs','bc_updates','bc_lr','actor_lr','actor_epochs',
            'clip_eps','entropy_coef','minibatch_size','d_model','n_heads','n_layers','final_hidden_dim',
            'gamma','actor_lambda','critic_lr','critic_epochs','value_clip','max_grad_norm','eval_episodes',
            'program_min_ops','program_max_ops','program_degree','program_max_inputs','program_registers',
            'program_coefficient_limit','program_teacher_probability','program_problem_file',
            'task_bank_hash','heldout_task_bank_hash')
    mismatches={k:[dict(path=r['path'],value=r['cfg'].get(k)) for r in runs]
                for k in shared if len({json.dumps(r['cfg'].get(k)) for r in runs})>1}
    (dest/'config_differences.json').write_text(json.dumps(mismatches,indent=2)+'\n')
    for axis,xlabel in (('update','PPO update'),('trunk_samples','Trunk transitions'),
                        ('total_samples','Trunk + auxiliary transitions')):
        fig,ax=plt.subplots(figsize=(11,6))
        for run in runs:band(ax,run,axis)
        ax.set(xlabel=xlabel,ylabel='Mean trunk return',ylim=(0,1),title='PPO comparison — mean ± seed SEM')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,dest,'main_'+axis)
    fig,ax=plt.subplots(figsize=(11,6))
    for i,run in enumerate(runs):
        ax.errorbar(i,run['final'].mean(),yerr=sem(run['final']),fmt='o',capsize=4)
    ax.set_xticks(range(len(runs)),[r['label'] for r in runs],rotation=35,ha='right')
    ax.set(ylabel='Fresh final evaluation return',ylim=(0,1));ax.grid(alpha=.2)
    save(fig,dest,'final_evaluation')
    if any(r['heldout'] is not None for r in runs):
        fig,ax=plt.subplots(figsize=(11,6))
        for i,run in enumerate(runs):
            if run['heldout'] is not None:
                ax.errorbar(i,run['heldout'].mean(),yerr=sem(run['heldout']),fmt='o',capsize=4)
        ax.set_xticks(range(len(runs)),[r['label'] for r in runs],rotation=35,ha='right')
        ax.set(ylabel='Held-out problem success',ylim=(0,1));ax.grid(alpha=.2)
        save(fig,dest,'heldout_evaluation')
    fbi=[r for r in runs if r['cfg']['critic_method']=='fbi']
    for grouping,stem in (('critic_lambda','fbi_by_k'),('k','fbi_by_lambda')):
        groups=sorted({r['cfg'][grouping] for r in fbi})
        if not groups:continue
        fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),5),squeeze=False)
        for ax,value in zip(axes[0],groups):
            for run in fbi:
                if run['cfg'][grouping]==value:band(ax,run,'update')
            # K=1 trunk-included FBI should match ordinary TD at that lambda.
            for run in runs:
                if run['cfg']['critic_method']=='td' and (grouping=='k' or run['cfg']['critic_lambda']==value):
                    band(ax,run,'update',label=run['label']+' reference')
            name='λ' if grouping=='critic_lambda' else 'K'
            ax.set(title=f'FBI: {name}={value:g}',xlabel='PPO update',ylabel='Mean trunk return',ylim=(0,1))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,dest,stem)
    rows=[]
    for run in sorted(runs,key=lambda r:-r['auc'].mean()):
        c=run['cfg']
        rows.append(dict(run=run['path'],method=c['critic_method'],k=c.get('k'),
                         critic_lambda=c.get('critic_lambda'),n_seeds=len(run['data']),
                         auc_mean=run['auc'].mean(),auc_sem=sem(run['auc']),
                         final_return_mean=run['final'].mean(),final_return_sem=sem(run['final']),
                         initial_return_mean=run['initial'].mean(),
                         heldout_final_return_mean=float(run['heldout'].mean()) if run['heldout'] is not None else '',
                         heldout_final_return_sem=float(sem(run['heldout'])) if run['heldout'] is not None else ''))
    for row in rows:
        row['auc_rank']=1+sum(q['auc_mean']>row['auc_mean'] for q in rows)
        row['final_return_rank']=1+sum(q['final_return_mean']>row['final_return_mean'] for q in rows)
    with (dest/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Plots and summary: {dest}')
    print('AUC uses unsmoothed training return, normalized over recorded updates. Final return uses fresh final evaluation.')
    if mismatches:print('Configuration differences detected; inspect config_differences.json before interpreting comparisons.')
    ks=sorted({r['cfg']['k'] for r in fbi});lambdas=sorted({r['cfg']['critic_lambda'] for r in fbi})
    print(f'Available FBI settings: K={ks}, lambda={lambdas}. Missing combinations are not fabricated.')
    if skipped: print(f'Excluded {len(skipped)} incomplete/invalid runs; see collection_status.json.')


if __name__=='__main__':main()
