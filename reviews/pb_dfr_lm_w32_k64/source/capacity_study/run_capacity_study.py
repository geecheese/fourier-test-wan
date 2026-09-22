"""Three architecture cases, reuse completed A; train B/C once, no tuning."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'fourier_resolution_study'))
import run_study as study
import numpy as np
import torch
from torch import nn
from train_pb_dfr_smoke import GlobalTrial
from run_full_solver import diagnostic, backward, finite_parameters, CONFIG
from pb_fourier_residual import moments, metrics, quadrature
ARCH={'A':[2,32,32,32,1],'B':[2,64,64,64,1],'C':[2,64,64,64,64,1]}


class Trial(GlobalTrial):
    # Inherit the unchanged global hard-boundary forward; vary only layer sizes.
    def __init__(self, dims):
        nn.Module.__init__(self)
        layers=[]
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(dims)-2: layers.append(nn.Tanh())
        self.net=nn.Sequential(*layers)
        self.double()


def model_for(label):
    torch.manual_seed(2026); np.random.seed(2026)
    return Trial(ARCH[label])


def save(model,opt,label,stage,step):
    torch.save(dict(model=model.state_dict(),optimizer=None if opt is None else opt.state_dict(),
                    config={**CONFIG,'K':24,'network':ARCH[label]},stage=stage,step=step,
                    rng_state=torch.get_rng_state()),ROOT/'checkpoints'/f'{label}_{stage}.pt')


def reuse_A():
    source=BASE/'fourier_resolution_study'
    meta=json.loads((source/'comparisons/k24_metadata.json').read_text())
    expected={**CONFIG,'K':24}
    assert meta['config']==expected
    initial=torch.load(source/'checkpoints/k24_initial.pt',weights_only=False)['model']
    model=model_for('A')
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    for stage in ('initial','adam','final'):
        target=ROOT/'checkpoints'/f'A_{stage}.pt'
        assert not target.exists(), 'A already imported; do not overwrite'
        shutil.copy2(source/'checkpoints'/f'k24_{stage}.pt',target)
    shutil.copy2(source/'comparisons/k24_trajectory.json',ROOT/'reports/A_trajectory.json')
    meta.update(reused=True,source=str(source/'checkpoints/k24_final.pt'),
                source_sha256=hashlib.sha256((source/'checkpoints/k24_final.pt').read_bytes()).hexdigest(),
                parameter_count=sum(p.numel() for p in model.parameters()))
    (ROOT/'reports/A_metadata.json').write_text(json.dumps(meta,indent=2))
    print('A: verified identical protocol and seeded initialization; reused existing K=24 final state, no retraining.',flush=True)


def train(label):
    assert not (ROOT/'checkpoints'/f'{label}_initial.pt').exists(), 'Case already started; do not retrain'
    model=model_for(label); initial_hash=study.state_hash(model)
    low=study.ResidualK(32,24)
    checks=study.preflight(model,low,24)
    print(label,'preflight',checks,flush=True)
    save(model,None,label,'initial',0)
    rows=[]
    def log(stage,step):
        print(label,stage,flush=True)
        rows.append(diagnostic(model,low,stage,step))
        (ROOT/'reports'/f'{label}_trajectory.json').write_text(json.dumps(rows,indent=2))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    start=time.monotonic(); log('adam',0)
    for step in range(1,3001):
        opt.zero_grad(set_to_none=True); backward(model,low); opt.step(); finite_parameters(model)
        if step in (100,250,500,1000,1500,2000,2500,3000): log('adam',step)
    save(model,opt,label,'adam',3000)
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=5,history_size=50,
                         line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
    calls=0;log('lbfgs',0)
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad(set_to_none=True)
        return backward(model,low)
    for step in range(1,201):
        opt.step(closure);finite_parameters(model)
        if step in (5,10,25,50,75,100,150,200):log('lbfgs',step)
    save(model,opt,label,'final',200)
    meta=dict(config={**CONFIG,'K':24,'network':ARCH[label]},initial_sha256=initial_hash,checks=checks,
              runtime_seconds=time.monotonic()-start,adam_updates=3000,lbfgs_outer_steps=200,
              closure_calls=calls,internal_lbfgs_iterations=opt.state[next(iter(model.parameters()))]['n_iter'],
              parameter_count=sum(p.numel() for p in model.parameters()),reused=False,all_finite=True,
              torch_version=torch.__version__,numpy_version=np.__version__)
    (ROOT/'reports'/f'{label}_metadata.json').write_text(json.dumps(meta,indent=2))


def audit(label):
    model=model_for(label)
    model.load_state_dict(torch.load(ROOT/'checkpoints'/f'{label}_final.pt',weights_only=False)['model'])
    meta=json.loads((ROOT/'reports'/f'{label}_metadata.json').read_text())
    final=diagnostic(model,study.ResidualK(64,24),'final_n64',200)
    trajectory=json.loads((ROOT/'reports'/f'{label}_trajectory.json').read_text())
    val,grad=study.functions(model)
    # Batch evaluation to bound memory only; quadrature and moment sums unchanged.
    def value(x):return np.concatenate([val(b) for b in np.array_split(x,max(1,(len(x)+8191)//8192))])
    def gradient(x):return np.concatenate([grad(b) for b in np.array_split(x,max(1,(len(x)+8191)//8192))])
    r=moments(value,gradient,40,quadrature(64))
    r96=moments(value,gradient,40,quadrature(96))
    m,m96=metrics(r),metrics(r96)
    final.update(architecture=label,dimensions=ARCH[label],training_loss=trajectory[-1]['loss'],
                 L_DFR_40=m['L_DFR'],runtime_seconds=meta['runtime_seconds'],parameter_count=meta['parameter_count'],
                 quadrature_relative=abs(trajectory[-1]['loss']-final['loss'])/final['loss'],
                 audit40_quadrature_relative=abs(m['L_DFR']-m96['L_DFR'])/m96['L_DFR'])
    assert final['quadrature_relative']<1e-8 and final['audit40_quadrature_relative']<1e-8
    return dict(final=final,metadata=meta,audit40=m,audit40_n96=m96)


def report():
    results={label:audit(label) for label in ARCH}
    (ROOT/'reports/final_results.json').write_text(json.dumps(results,indent=2))
    with (ROOT/'reports/final_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results['A']['final']));writer.writeheader()
        writer.writerows(r['final'] for r in results.values())
    a=results['A']['final']
    # Reporting criterion fixed before new training: improvement must halve both errors.
    substantial=any(all(results[k]['final'][e]<.5*a[e] for e in ('rel','h1')) and
                    results[k]['final']['L_DFR_40']<a['L_DFR_40'] for k in ('B','C'))
    decision=('DECISION: NETWORK CAPACITY LIMITS CURRENT ACCURACY' if substantial else
              'DECISION: INCREASING NETWORK CAPACITY GIVES ONLY MARGINAL IMPROVEMENT')
    lines=['# Controlled network-capacity study','',
        '**provisional implementation benchmark**; three architecture cases at fixed K_train=24.', '',
        '## Experimental control and provenance','',
        'Question: is the current approximately 1% regular RelL2 mainly limited by network capacity? A is the completed K=24 case from fourier_resolution_study, reused to obey the instruction not to retrain previous experiments. Its configuration and seeded initialization were verified against this study, and its checkpoints copied byte-for-byte. Exactly two new training runs, B and C, complete the three-case comparison. No previous experiment was retrained or overwritten.', '',
        'Architecture A: 2-32-32-32-1; B: 2-64-64-64-1; C: 2-64-64-64-64-1. Hidden activations tanh, float64, seed 2026 reset before each construction. All use the same default PyTorch Linear initialization rule (including its fan-in scaling); different shapes cannot have identical parameter tensors. Architecture A reproduces the previous initial tensors exactly.', '',
        'The validated benchmark is unchanged: Omega=[-1,1]², circle R=0.5, epsilon-=2, epsilon+=80, kappa-=kappa+=1, sigma=0.08, Q=1. The coefficient-scaled softened u_s, manufactured regular solution P(r²), singular source plus manufactured remainder, and exterior data are imported unchanged. P(t)=0.2(t-0.25)²(1-t)².', '',
        'One globally smooth trial u_r,theta=ell+(1-x²)(1-y²)N_theta, with the same boundary-data Coons lift ell=P(1+x²)+P(1+y²)-P(2). The inherited forward method is unchanged. The only objective is L_DFR=sum(R_kl²/(1+lambda_kl)) at K_train=24, using the validated nonlinear weak residual including the singular-source and interface contributions. No penalties, new method, K increase during training, or extra regularization.', '',
        'Fixed schedule for every architecture: Adam lr=1e-3 for 3000 updates; L-BFGS for 200 outer steps, max_iter=5, lr=1, history_size=50, strong_wolfe, tolerance_grad=1e-10, tolerance_change=1e-14, default max_eval=6. No per-architecture tuning, early-state selection or additional iterations. CPU, two threads, deterministic algorithms.', '',
        'Training retains the eight-sector, circle-split radial Gauss quadrature at n=32. Common final errors and K=24 residuals use n=64; K_eval=40 uses the original validated NumPy weak-moment evaluator at n=64, cross-checked at n=96. Batching final model evaluation changes memory usage only, not the quadrature or residual. Initial exact-solution, NumPy/Torch coupling, directional derivative and quadrature checks passed.', '',
        '## Final metrics','',
        'Final L_DFR below is the training-rule value at K=24. Regular/total RelL2 and absolute H1 error use common n=64 quadrature. H1 error=sqrt(integral(error²+|grad(error)|²)). Total error uses u=u_r+u_s with the known singular field unchanged. Runtime includes training, scheduled diagnostics and checkpoint writes; it excludes preflight/final audit. A runtime is taken from the original run and is not used to assess quality.', '',
        '| Case | Architecture | Parameters | Final L_DFR | Regular RelL2 | Total RelL2 | H1 error | L_DFR(K_eval=40) | Runtime (s) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in results.items():
        x=r['final'];lines.append('| '+label+' | '+'-'.join(map(str,ARCH[label]))+f" | {x['parameter_count']} | "+' | '.join(f'{x[k]:.6e}' for k in ('training_loss','rel','total_rel','h1','L_DFR_40','runtime_seconds'))+' |')
    lines+=['','## Physical flux and boundary diagnostics','',
        'Physical flux residual=[epsilon partial_n u_r]+[epsilon partial_n u_s], sampled at 4096 interface points with the disk-outward normal. The singular weighted flux jump is zero for this benchmark. Both regular traces belong to the same smooth trial. Boundary max uses 1028 edge points. These diagnostics are never training penalties.', '',
        '| Case | Flux RMS | Flux max | Boundary max | Regular solution jump RMS |','|---|---:|---:|---:|---:|']
    for label,r in results.items():
        x=r['final'];lines.append('| '+label+' | '+' | '.join(f'{x[k]:.6e}' for k in ('flux_rms','flux_max','boundary','jump_rms'))+' |')
    lines+=['','## Validation and optimization checks','',
        '| Case | Relative K=24 quadrature discrepancy (n=32/64) | Relative K=40 discrepancy (n=64/96) | L-BFGS closure calls | Internal iterations |',
        '|---|---:|---:|---:|---:|']
    for label,r in results.items():
        x,m=r['final'],r['metadata'];lines.append(f"| {label} | {x['quadrature_relative']:.6e} | {x['audit40_quadrature_relative']:.6e} | {m['closure_calls']} | {m['internal_lbfgs_iterations']} |")
    lines+=['','All three schedules completed. Every Adam update and L-BFGS closure passed finite-loss, residual, solution, spatial-gradient and parameter-gradient checks. Updated parameters and logged errors were finite; no NaN/Inf occurred. Internal iteration counts may differ under identical line-search settings. Metadata and trajectories preserve those differences.', '',
        '## Interpretation','']
    for label in ('B','C'):
        x=results[label]['final'];lines.append(label+'/A final ratios: '+', '.join(f'{k}={x[k]/a[k]:.6f}' for k in ('rel','h1','flux_rms','L_DFR_40'))+'.')
    lines+=['','This is one seed and one fixed optimizer budget per architecture. Larger networks alter both representational capacity and optimization difficulty. A capacity limitation is supported here only if a larger model at least halves both regular RelL2 and H1 error and lowers common K_eval=40 loss; otherwise improvement is classified conservatively as marginal (including no improvement). This reporting rule was set before B/C training and did not affect optimization. The experiment cannot prove that capacity is the sole or main mathematical error source, or that either network has reached its approximation limit.', '',
        '## Artifacts and reproducibility','',
        'checkpoints/ contains initial, final Adam and final L-BFGS states for A/B/C; A is a verified copy of the previous K=24 run. reports/ contains per-case trajectory and metadata JSON, final_metrics.csv and final_results.json. No figures or additional experiments are created.', '',
        '```bash','OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/capacity_study/run_capacity_study.py',
        '# Re-evaluate checkpoints and regenerate the report without training:',
        'OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/capacity_study/run_capacity_study.py --report-only','```','',
        '## Final decision','',decision,'']
    (ROOT/'reports/capacity_report.md').write_text('\n'.join(lines))
    print(json.dumps({k:r['final'] for k,r in results.items()},indent=2),flush=True)
    print(decision,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report-only',action='store_true');args=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if not args.report_only:
        reuse_A()
        for label in ('B','C'):train(label)
    report()


if __name__=='__main__':main()
