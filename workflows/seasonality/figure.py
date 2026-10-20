#!/usr/bin/env python3
"""Fresh journal Figure 1: pilot, resumable compute, and separate plot stages."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from boss.bridge.r_session import RSeasonalityEvaluator
from boss.config import BOConfig
from boss.bo.surrogate import build_surrogate, model_hyperparameters, physical_to_normalized
from boss.bo.acquisition import select_candidate, beta_schedule

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'output/pdf/journal_figure1_reproduction'
GRID = np.linspace(.5, 4.5, 801)
BUDGETS = tuple(range(10, 81, 5))

def config(seed):
    return BOConfig.journal(((.5,), (4.5,)), budget=80, seed=seed,
                            num_initial=5, lengthscale_fit_restarts=20,
                            num_restarts=20, acquisition_maxiter=100)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def normalize(logp, x=None):
    x = GRID if x is None else x
    logp = np.asarray(logp, dtype=float)
    if not np.isfinite(logp).all():
        raise ValueError('nonfinite log density')
    z = float(logp.max()) + np.log(np.trapz(np.exp(logp-logp.max()), x))
    return np.exp(logp-z), logp-z

def metrics(logp, logq, x=None):
    x = GRID if x is None else x
    p, q = np.exp(logp), np.exp(logq)
    # Authors drop zero-p points before trapezoidal KL; retain log densities
    # to avoid spurious infinities when q underflows in negligible tails.
    keep = p > 0
    kl = np.trapz(p[keep]*(logp[keep]-logq[keep]), x[keep])
    dx = np.r_[np.diff(x), 0.]
    ks = np.max(np.abs(np.cumsum(p*dx)-np.cumsum(q*dx)))
    ks_trap = np.max(np.abs(cumulative_trapezoid(p-q, x, initial=0)))
    return {'KL': float(kl), 'KS': float(ks), 'KS_trapezoidal': float(ks_trap)}

def evaluation_row(e, **extra):
    return dict(alpha=float(e.alpha[0]), objective=e.objective,
                log_marginal=e.log_marginal, log_prior=e.log_prior,
                elapsed_seconds=e.elapsed_seconds, **extra)

def new_evaluator(seed, directory):
    e = RSeasonalityEvaluator(None, ROOT/'R/seasonality_evaluator.R', seed=seed)
    e.initialize()
    data_path = directory/'dataset.csv'
    temporary = directory/'dataset.generated.csv'
    e.ro.r['write.csv'](e.ro.r('.boss_seasonality_state$data'), str(temporary), **{"row.names":False})
    if data_path.exists() and sha(data_path) != sha(temporary):
        temporary.unlink()
        raise ValueError('Regenerated R data differs from saved dataset')
    temporary.replace(data_path)
    return e

def audit(e, directory):
    raw = e.ro.r('jsonlite::toJSON(.boss_seasonality_state$prior_audit, auto_unbox=TRUE, digits=16)')
    data = json.loads(str(raw[0]))
    if not data:
        # Resume can load all evaluations without making another LGM fit.
        return
    data['R_version'] = str(e.ro.r('R.version.string')[0])
    data['package_versions'] = json.loads(str(e.ro.r('jsonlite::toJSON(setNames(lapply(c("BayesGP","TMB","aghq","Matrix"), function(p) as.character(packageVersion(p))), c("BayesGP","TMB","aghq","Matrix")),auto_unbox=TRUE)')[0]))
    data['TMB_SD_prior'] = 'phi=-log(alpha)/u; log(0.5*phi)-phi*exp(-0.5*theta)-0.5*theta'
    data['verified'] = True
    write_json(directory/'prior_audit.json', data)

def pilot(output, seed):
    directory = output/'pilot'; directory.mkdir(parents=True, exist_ok=True)
    e = new_evaluator(seed, directory)
    rows = [evaluation_row(e.evaluate(np.array([a]))) for a in (.5, 1.5, 2.5, 3.5, 4.5, 1.5)]
    assert abs(rows[1]['objective']-rows[-1]['objective']) < 1e-10
    audit(e, directory)
    pd.DataFrame(rows).to_csv(directory/'evaluations.csv', index=False)
    seconds = [r['elapsed_seconds'] for r in rows]
    result = {'fits':len(rows), 'mean_fit_seconds':float(np.mean(seconds)),
              'median_fit_seconds':float(np.median(seconds)),
              'full_LGM_fit_count':9691,
              'serial_LGM_hours_estimate':float(np.mean(seconds)*9691/3600),
              'note':'Pilot excludes GP fitting/acquisition overhead and startup; runtime varies by data/alpha.'}
    write_json(directory/'timing.json', result)
    print(json.dumps(result), flush=True)

def runtime_fingerprint(e):
    return {'R_version':str(e.ro.r('R.version.string')[0]),
               'packages':json.loads(str(e.ro.r('jsonlite::toJSON(setNames(lapply(c("BayesGP","TMB","aghq","Matrix"), function(p) as.character(packageVersion(p))), c("BayesGP","TMB","aghq","Matrix")),auto_unbox=TRUE)')[0])),
               'python':{p:importlib.metadata.version(p) for p in ['torch','botorch','gpytorch','numpy','scipy','pandas']}}

def compute_seed(output, seed):
    directory = output/f'seed_{seed}'; directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'checkpoint.json'
    cfg = config(seed)
    signature = {'grid':GRID.tolist(), 'budgets':list(BUDGETS), 'runner_sha256':sha(Path(__file__)), 'config':json.loads(json.dumps(asdict(cfg))),
                 'evaluator_sha256':sha(ROOT/'R/seasonality_evaluator.R'),
                 'gp_sha256':sha(ROOT/'src/boss/bo/surrogate.py'),
                 'acquisition_sha256':sha(ROOT/'src/boss/bo/acquisition.py')}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        'signature':signature, 'oracle':[], 'bo':[], 'model':None,
        'sessions':[], 'prefix_fits':[]}
    if state['signature'] != signature:
        raise ValueError(f'Checkpoint configuration/source mismatch: {checkpoint}')
    e = new_evaluator(seed, directory)
    runtime = runtime_fingerprint(e)
    if state.get('runtime',runtime) != runtime:
        raise ValueError('Installed runtime differs from checkpoint')
    previous_audit = directory/'prior_audit.json'
    if previous_audit.exists():
        previous = json.loads(previous_audit.read_text())
        if previous['R_version'] != runtime['R_version'] or previous['package_versions'] != runtime['packages']:
            raise ValueError('Installed R packages differ from saved prior audit')
    state['runtime'] = runtime
    dataset_hash = sha(directory/'dataset.csv')
    if state.get('dataset_sha256', dataset_hash) != dataset_hash:
        raise ValueError('Regenerated R data differs from checkpoint')
    state['dataset_sha256'] = dataset_hash
    started = time.perf_counter()
    session = {'started_utc':pd.Timestamp.now(tz='UTC').isoformat(), 'elapsed_seconds':0.}
    state['sessions'].append(session)
    def save():
        session['elapsed_seconds'] = time.perf_counter()-started
        write_json(checkpoint, state)
    for index in range(len(state['oracle']), len(GRID)):
        state['oracle'].append(evaluation_row(e.evaluate(np.array([GRID[index]])), index=index))
        save()
        if index % 100 == 0:
            print(f'seed {seed}: oracle {index+1}/{len(GRID)}', flush=True)
    audit(e, directory)
    initial = np.linspace(.5,4.5,5)
    for index in range(len(state['bo']), 5):
        state['bo'].append(evaluation_row(e.evaluate(np.array([initial[index]])),
                                           iteration=index+1, candidate_source='initial_design'))
        save()
    def training():
        x = np.array([r['alpha'] for r in state['bo']])[:,None]
        y = np.array([r['objective'] for r in state['bo']])
        return physical_to_normalized(x, np.array(cfg.bounds)), y-y.mean(), y
    model = None
    if state['model'] is not None:
        x,y,_ = training()
        model = build_surrogate(x,y,cfg, journal_outputscale=state['model']['outputscale'])
        model.covar_module.base_kernel.lengthscale = torch.tensor(state['model']['lengthscale'], dtype=torch.float64)
    while len(state['bo']) < cfg.budget:
        observed = len(state['bo'])
        x,y,absolute = training()
        fitted = observed == 5 or cfg.should_fit_hyperparameters(observed)
        scale = float(np.var(absolute,ddof=1)) if fitted else state['model']['outputscale']
        step_start = time.perf_counter()
        model = build_surrogate(x,y,cfg, previous_model=model,
                                journal_outputscale=scale, fit_hyperparameters=fitted,
                                fit_observation_count=observed)
        lengthscale, outputscale = model_hyperparameters(model)
        candidate, acq, beta, source = select_candidate(model,x,observed+1,cfg)
        physical = .5+4*candidate
        row = evaluation_row(e.evaluate(physical), iteration=observed+1,
                             candidate_source=source, beta=beta, acquisition_value=acq,
                             lengthscale=float(lengthscale[0]), outputscale=outputscale,
                             hyperparameters_fitted=fitted, fit_observation_count=observed if fitted else None,
                             selection_and_fit_seconds=time.perf_counter()-step_start,
                             center_offset=float(absolute.mean()))
        state['bo'].append(row)
        state['model'] = {'lengthscale':float(lengthscale[0]), 'outputscale':outputscale}
        save()
        if (observed+1) % 5 == 0:
            print(f'seed {seed}: BO {observed+1}/80 ({source})', flush=True)
    audit(e, directory)
    pd.DataFrame(state['oracle']).to_csv(directory/'oracle_evaluations.csv',index=False)
    pd.DataFrame(state['bo']).to_csv(directory/'bo_evaluations.csv',index=False)
    logoracle = np.array([r['objective'] for r in state['oracle']])
    oracle, logp = normalize(logoracle)
    pd.DataFrame({'alpha':GRID,'objective':logoracle,'density':oracle,'log_density':logp}).to_csv(directory/'oracle_density.csv',index=False)
    rows = []
    for budget in BUDGETS:
        path = directory/f'density_b{budget}.csv'
        fitpath = directory/f'prefix_b{budget}.json'
        if not path.exists() or not fitpath.exists():
            step_start = time.perf_counter()
            prefix = state['bo'][:budget]
            x = (np.array([r['alpha'] for r in prefix])[:,None]-.5)/4
            y = np.array([r['objective'] for r in prefix])
            gp = build_surrogate(x,y-y.mean(),cfg,fit_hyperparameters=True,
                                 journal_outputscale=float(np.var(y,ddof=1)),fit_observation_count=budget)
            with torch.no_grad():
                mean = gp.posterior(torch.tensor(((GRID-.5)/4)[:,None],dtype=torch.float64)).mean.numpy().ravel()+y.mean()
            density, logq = normalize(mean)
            ls, scale = model_hyperparameters(gp)
            pd.DataFrame({'alpha':GRID,'gp_mean':mean,'density':density,'log_density':logq}).to_csv(path,index=False)
            write_json(fitpath, {'budget':budget, 'training_count':len(prefix),
                                 'training_sha256':hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest(),
                                 'lengthscale':float(ls[0]),'outputscale':scale,'center':float(y.mean()),
                                 'elapsed_seconds':time.perf_counter()-step_start})
        candidate = pd.read_csv(path)
        assert np.isclose(np.trapz(candidate.density,GRID),1.,atol=1e-12)
        values = metrics(logp,candidate.log_density.to_numpy())
        assert np.isfinite(list(values.values())).all() and values['KL'] >= -1e-10
        rows.append(dict(seed=seed,budget=budget,**values))
        state['prefix_fits'] = sorted(set(state['prefix_fits'])|{budget})
        save()
    pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False)
    state['complete'] = True
    save()
    print(f'seed {seed}: complete; {len(state["oracle"])+len(state["bo"])} LGM fits',flush=True)

def aggregate(output, seeds, illustration):
    tables = []
    counts = []
    for seed in [illustration,*seeds]:
        d = output/f'seed_{seed}'
        state = json.loads((d/'checkpoint.json').read_text())
        if not state.get('complete') or len(state['oracle']) != 801 or len(state['bo']) != 80:
            raise ValueError(f'incomplete seed {seed}')
        if seed in seeds:
            tables.append(pd.read_csv(d/'metrics.csv'))
        counts.append({'seed':seed,'oracle_fits':801,'bo_fits':80,
                       'LGM_seconds':sum(r['elapsed_seconds'] for r in state['oracle']+state['bo']),
                       'measured_session_seconds':sum(s['elapsed_seconds'] for s in state['sessions']),
                       'BO_selection_and_GP_seconds':sum(r.get('selection_and_fit_seconds',r['elapsed_seconds'])-r['elapsed_seconds'] for r in state['bo']),
                       'prefix_GP_seconds':sum(json.loads(p.read_text())['elapsed_seconds'] for p in d.glob('prefix_b*.json'))})
    all_metrics = pd.concat(tables,ignore_index=True)
    assert len(set(seeds)) == 10 and len(all_metrics) == 150
    all_metrics.to_csv(output/'replicate_metrics.csv',index=False)
    summaries = []
    for budget, table in all_metrics.groupby('budget'):
        assert len(table)==10 and set(table.seed)==set(seeds)
        for metric in ['KL','KS','KS_trapezoidal']:
            a = table[metric].to_numpy()
            summaries.append({'budget':int(budget),'metric':metric,'replicates':len(a),
                              'mean':float(a.mean()),'median':float(np.median(a)),
                              'q10':float(np.quantile(a,.1)), 'q90':float(np.quantile(a,.9)),
                              'sd':float(a.std(ddof=1))})
    pd.DataFrame(summaries).to_csv(output/'aggregate_statistics.csv',index=False)
    pd.DataFrame(counts).to_csv(output/'fit_counts_runtimes.csv',index=False)
    return pd.DataFrame(summaries)

def plot(output, seeds, illustration, paper, dpi, *, figures_only=False):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib'))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = (pd.read_csv(output/'aggregate_statistics.csv') if figures_only
               else aggregate(output,seeds,illustration))
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(2,2,figsize=(9,7),layout='constrained')
    oracle = pd.read_csv(output/f'seed_{illustration}/oracle_density.csv')
    expanded = []
    for ax,budget,label in zip(axes[0],(30,60),('a','b')):
        density = pd.read_csv(output/f'seed_{illustration}/density_b{budget}.csv')
        ax.plot(GRID,oracle.density,color='black',ls='-',lw=1.4,label='Grid oracle')
        ax.plot(GRID,density.density,color='red',ls='--',lw=1.4,label='BoTorch BOSS')
        raw_max=float(max(oracle.density.max(),density.density.max()))
        top = 30. if raw_max<=30. else raw_max*1.08
        if top > 30: expanded.append(f'density B={budget}: 0-{top:.2f}')
        ax.set(xlim=(.5,4.5),ylim=(0,top),xlabel=r'$\alpha$',ylabel='Density',title=f'({label}) Posterior: B = {budget}')
        ax.grid(color='.9',lw=.6); ax.legend(fontsize=9)
    for ax,metric,label,limit in zip(axes[1],('KL','KS'),('c','d'),(2.1,.55)):
        table=summary[summary.metric==metric]
        ax.fill_between(table.budget,table.q10,table.q90,color='.75',label='10th-90th percentiles')
        ax.plot(table.budget,table['mean'],color='black',marker='o',ms=3,lw=1,label='Mean (10 datasets)')
        raw_max=float(max(table.q90.max(),table['mean'].max()))
        top=limit if raw_max<=limit else raw_max*1.08
        if top > limit: expanded.append(f'{metric}: 0-{top:.2f}')
        ax.set(xlim=(8,82),ylim=(0,top),xlabel='Evaluation budget B',ylabel=metric,title=f'({label}) Oracle-to-BOSS {metric}')
        ax.grid(color='.9',lw=.6); ax.legend(fontsize=8)
    fig.savefig(output/'figure1_reproduction.pdf'); fig.savefig(output/'figure1_reproduction.png',dpi=dpi)
    plt.close(fig)
    return
