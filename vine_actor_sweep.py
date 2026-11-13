"""Plan, execute one cluster array task, or summarize a pure MC VinePPO sweep.

Planning uses only the Python standard library. Execution is never implicit.
"""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GRID = dict(actor_lr=[1e-4, 3e-4], clip_eps=[0.1, 0.2],
            entropy_coef=[0., 0.001], actor_epochs=[1, 4])
DEFAULTS = dict(num_problems=20, num_milestones=12, max_steps=40,
                num_actions=8, num_updates=100, num_envs=64, n_seeds=3,
                master_seed=42, bc_updates=100, bc_lr=3e-4,
                eval_episodes=1024, minibatch_size=64, vine_batch_size=256,
                d_model=64, n_heads=4, n_layers=2, final_hidden_dim=32)


def command(manifest, setting):
    flags = dict(manifest['common'], **setting['parameters'],
                 save_dir=str(Path(manifest['output']) / setting['id']))
    return [manifest['python'], '-u', str(ROOT / 'train_dag_ppo.py'),
            *[word for k, v in flags.items() for word in ('--'+k, str(v))]]


def plan(args):
    out = Path(args.output).resolve()
    if (out / 'manifest.json').exists():
        raise SystemExit('Output already has a manifest. Use a new directory to avoid mixing sweeps.')
    common = {k: type(v)(os.environ.get(k.upper(), v)) for k, v in DEFAULTS.items()}
    for k in ('num_problems','num_milestones','max_steps','num_actions','num_updates',
              'num_envs','n_seeds','eval_episodes','minibatch_size','vine_batch_size'):
        if common[k] < 1: raise SystemExit(f'{k} must be positive')
    if common['num_updates'] < 2: raise SystemExit('AUC requires at least two PPO updates')
    grid = {k: [type(v[0])(x) for x in os.environ.get(k.upper()+'_GRID', ','.join(map(str,v))).split(',')]
            for k,v in GRID.items()}
    if any(x <= 0 for k in ('actor_lr','clip_eps','actor_epochs') for x in grid[k]):
        raise SystemExit('Learning rates, clipping and epochs must be positive')
    if any(x < 0 for x in grid['entropy_coef']): raise SystemExit('Entropy must be nonnegative')
    common.update(env_type=args.env, critic_method='vine_mc', k=128, actor_lambda=0., gamma=1.)
    settings = [dict(id=f'setting_{i:03d}', parameters=dict(zip(grid, values)))
                for i,values in enumerate(itertools.product(*grid.values()))]
    manifest = dict(version=1, output=str(out), python=os.environ.get('PYTHON_BIN',sys.executable),
                    common=common, grid=grid, settings=settings)
    out.mkdir(parents=True, exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'commands.txt').write_text('\n'.join(shlex.join(command(manifest,s)) for s in settings)+'\n')
    print(f'Planned {len(settings)} settings × {common["n_seeds"]} matched seeds; K=128. {out}')


def run(args):
    manifest = json.loads((Path(args.output)/'manifest.json').read_text())
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(manifest['settings']): raise SystemExit('Invalid array index')
    setting = manifest['settings'][index]
    destination = Path(manifest['output'])/setting['id']
    if destination.exists(): raise SystemExit(f'Refusing to overwrite {destination}; use a new sweep directory.')
    cmd = command(manifest,setting)
    print(shlex.join(cmd),flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def collect(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(args.output).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    rows, curves, missing, seeds = [], [], [], []
    common = manifest['common']
    n = common['n_seeds']
    def sem(values):
        return float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    for setting in manifest['settings']:
        directory = out/setting['id']
        try:
            config = json.loads((directory/'config.json').read_text())
            expected = dict(common, **setting['parameters'])
            if any(config.get(k) != v for k,v in expected.items()): raise ValueError('configuration mismatch')
            summary = json.loads((directory/'summary.json').read_text())
            if sorted(s['seed'] for s in summary) != list(range(n)): raise ValueError('incomplete evaluation seeds')
            summary = {s['seed']:s for s in summary}
            values, aucs, finals, initials = [], [], [], []
            local_seeds = []
            for seed in range(n):
                data = np.genfromtxt(directory/f'metrics_seed{seed}.csv',delimiter=',',names=True)
                data = np.atleast_1d(data)
                x,y = data['update'],data['success_rate']
                if not np.array_equal(x,np.arange(1,common['num_updates']+1)): raise ValueError('incomplete updates')
                if not np.isfinite(y).all(): raise ValueError('nonfinite returns')
                integrate = getattr(np,'trapezoid',None) or np.trapz
                auc = float(integrate(y,x)/(x[-1]-x[0]))
                final = float(summary[seed]['final_success'])
                initial = float(summary[seed]['initial_success'])
                if not np.isfinite([final,initial]).all(): raise ValueError('nonfinite evaluation')
                values.append(y);aucs.append(auc);finals.append(final);initials.append(initial)
                local_seeds.append(dict(setting=setting['id'],seed=seed,auc=auc,final_return=final,
                                        initial_return=initial))
            seeds.extend(local_seeds)
            rows.append(dict(setting=setting['id'],**setting['parameters'],n_seeds=n,
                             auc_mean=float(np.mean(aucs)),auc_sem=sem(aucs),
                             final_return_mean=float(np.mean(finals)),final_return_sem=sem(finals),
                             initial_return_mean=float(np.mean(initials))))
            curves.append((setting,np.stack(values),x))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            missing.append(dict(setting=setting['id'],reason=str(exc)))
    (out/'collection_status.json').write_text(json.dumps(dict(complete=len(rows),expected=len(manifest['settings']),
                                                            excluded=missing),indent=2)+'\n')
    if not rows:
        raise SystemExit('No complete settings. See collection_status.json; no rankings generated.')
    rows.sort(key=lambda r:(-r['auc_mean'],-r['final_return_mean'],r['setting']))
    final_order = sorted(rows,key=lambda r:(-r['final_return_mean'],-r['auc_mean'],r['setting']))
    # Competition ranks preserve ties instead of assigning an arbitrary winner.
    for r in rows:
        r['auc_rank'] = 1+sum(q['auc_mean']>r['auc_mean'] for q in rows)
        r['final_return_rank'] = 1+sum(q['final_return_mean']>r['final_return_mean'] for q in rows)
    def write_csv(name, records):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_csv('ranking_auc.csv',rows)
    write_csv('ranking_final_return.csv',final_order)
    write_csv('per_seed.csv',seeds)
    fig,ax=plt.subplots(figsize=(14,7))
    colors=plt.get_cmap('tab20')
    for i,(setting,ys,x) in enumerate(curves):
        p=setting['parameters']
        label=f'{setting["id"]}: lr={p["actor_lr"]:g}, clip={p["clip_eps"]:g}, ent={p["entropy_coef"]:g}, epochs={p["actor_epochs"]}'
        mean=ys.mean(0)
        error=ys.std(0,ddof=1)/np.sqrt(n) if n>1 else np.zeros_like(mean)
        ax.plot(x,mean,color=colors(i%20),linestyle=['-','--',':','-.'][(i//20)%4],label=label)
        ax.fill_between(x,mean-error,mean+error,color=colors(i%20),alpha=.08)
    ax.set(xlabel='PPO update',ylabel='Mean trunk return (success rate)',ylim=(0,1),
           title=f'Pure MC VinePPO, K=128 — {common["env_type"]} ({n} seeds per setting)')
    ax.grid(alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'learning_curves.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Pure MC VinePPO actor sweep','',
           f'{len(rows)}/{len(manifest["settings"])} complete settings; {n} seeds each. K=128.',
           '', 'AUC is the trapezoidal mean training return over updates 1 through N (normalized by N−1).',
           'Final return is a fresh stochastic evaluation of the last policy, not the last training batch.',
           'Uncertainty is the standard error across seeds; one seed has no uncertainty estimate.',
           'All curves share the same warmup learning rate, environment and seed schedule.',
           'Select actor settings here, then hold them fixed across critic comparisons and validate with fresh seeds.', '',
           '| AUC rank | Final rank | Setting | LR | Clip | Entropy | Epochs | AUC ± SEM | Final return ± SEM |',
           '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["auc_rank"]} | {r["final_return_rank"]} | {r["setting"]} | {r["actor_lr"]:g} | {r["clip_eps"]:g} | {r["entropy_coef"]:g} | {r["actor_epochs"]} | {r["auc_mean"]:.4f} ± {r["auc_sem"]:.4f} | {r["final_return_mean"]:.4f} ± {r["final_return_sem"]:.4f} |')
    if missing:
        lines += ['', 'Incomplete settings excluded:', *[f'- {s["setting"]}: {s["reason"]}' for s in missing]]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('plan');p.add_argument('--env',choices=['dag','dag_quality','dag_approach'],default='dag_approach')
    p.add_argument('--output',required=True);p.set_defaults(func=plan)
    p=sub.add_parser('run');p.add_argument('--output',required=True);p.add_argument('--index',type=int);p.set_defaults(func=run)
    p=sub.add_parser('collect');p.add_argument('--output',required=True);p.set_defaults(func=collect)
    args=parser.parse_args();args.func(args)


if __name__=='__main__':main()
