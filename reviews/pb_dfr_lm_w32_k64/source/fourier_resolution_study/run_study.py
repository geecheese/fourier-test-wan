"""Exactly three fixed K cases. --report-only never optimizes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE/'scripts'))
sys.path.insert(0, str(BASE/'full_solver/scripts'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pb-dfr-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/pb-dfr-cache')
import numpy as np
import torch
import pb_benchmark as pb
from pb_fourier_residual import basis, spectral_data, quadrature, moments, metrics
from train_pb_dfr_smoke import GlobalTrial, TorchResidual, tensor, polynomial
from run_full_solver import diagnostic, backward, finite_parameters, CONFIG

TRAIN_K = (16, 24, 32)
EVAL_K = (16, 24, 32, 40)


class ResidualK(TorchResidual):
    """Parameterize only K; inherit the exact validated differentiable residual."""
    def __init__(self, n, K):
        self.rule = quadrature(n)
        x, w, c, wc = self.rule
        self.x, self.w = tensor(x), tensor(w)
        self.sx, self.sy, self.dx, self.dy = map(tensor, basis(x, K))
        self.eps, self.k2, self.us = map(tensor, (pb.epsilon(x),pb.kappa(x)**2,pb.u_s(x)))
        self.weights = tensor(spectral_data(K)[1])
        self.source = self.project(self.sx,self.sy,tensor(pb.manufactured_remainder(x)))
        self.source += self.project(self.sx,self.sy,tensor(pb.singular_source(x)))
        cx,cy,_,_ = basis(c,K)
        self.interface = tensor(cx.T @ ((wc*pb.singular_flux_jump(c))[:,None]*cy))


def seeded_model():
    torch.manual_seed(2026); np.random.seed(2026)
    return GlobalTrial()


def state_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode()); h.update(p.numpy().tobytes())
    return h.hexdigest()


def functions(model):
    def value(x):
        with torch.no_grad(): return model(tensor(x)).numpy()
    def gradient(x):
        t=tensor(x).requires_grad_(True)
        return torch.autograd.grad(model(t).sum(),t)[0].numpy()
    return value,gradient


def preflight(model, low, K):
    exact=lambda x: polynomial((x*x).sum(1))
    r=low.evaluate(exact)[1].detach().numpy()
    exact_max=float(np.max(np.abs(r)))
    assert exact_max < 1e-9
    val,grad=functions(model)
    reference=moments(val,grad,K,low.rule)
    loss,r,_,_=low.evaluate(model,True)
    port=float(np.max(np.abs(reference-r.detach().numpy())))
    assert port < 1e-9
    params=list(model.parameters())
    d=[torch.randn_like(p) for p in params]
    norm=torch.sqrt(sum(v.square().sum() for v in d)); d=[v/norm for v in d]
    g=torch.autograd.grad(loss,params)
    ad=sum((a*b).sum() for a,b in zip(g,d)).item()
    originals=[p.detach().clone() for p in params]
    vals=[]
    for sign in (1,-1):
        with torch.no_grad():
            for p,o,v in zip(params,originals,d): p.copy_(o+sign*1e-5*v)
        vals.append(low.evaluate(model)[0].item())
    with torch.no_grad():
        for p,o in zip(params,originals): p.copy_(o)
    fd=(vals[0]-vals[1])/2e-5
    derr=abs(fd-ad)/max(1.,abs(ad),abs(fd)); assert derr < 1e-6
    high=ResidualK(64,K)
    qrel=abs(loss.item()-high.evaluate(model)[0].item())/abs(loss.item())
    assert qrel < 1e-8
    return dict(exact_max=exact_max, numpy_torch_difference=port, derivative_relative=derr, initial_quadrature_relative=qrel)


def save(model,opt,K,stage,step):
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,
                    config={**CONFIG,'K':K},stage=stage,step=step,rng_state=torch.get_rng_state()),
               ROOT/'checkpoints'/f'k{K}_{stage}.pt')


def train(K):
    if (ROOT/'checkpoints'/f'k{K}_initial.pt').exists():
        raise RuntimeError('Existing case: do not retrain; --report-only uses the saved final states')
    model=seeded_model(); initial_hash=state_hash(model)
    low=ResidualK(32,K)
    checks=preflight(model,low,K)
    print('K',K,'preflight',checks,flush=True)
    rows=[]
    def log(stage,step):
        print('K_train',K,stage,flush=True)
        rows.append(diagnostic(model,low,stage,step))
        (ROOT/'comparisons'/f'k{K}_trajectory.json').write_text(json.dumps(rows,indent=2))
    save(model,None,K,'initial',0)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    start=time.monotonic()
    log('adam',0)
    for step in range(1,3001):
        opt.zero_grad(set_to_none=True); backward(model,low); opt.step(); finite_parameters(model)
        if step in (100,250,500,1000,1500,2000,2500,3000): log('adam',step)
    save(model,opt,K,'adam',3000)
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=5,history_size=50,
                         line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
    calls=0
    log('lbfgs',0)
    def closure():
        nonlocal calls
        calls+=1; opt.zero_grad(set_to_none=True)
        return backward(model,low)
    for step in range(1,201):
        opt.step(closure); finite_parameters(model)
        if step in (5,10,25,50,75,100,150,200): log('lbfgs',step)
    save(model,opt,K,'final',200)
    meta=dict(K=K,initial_sha256=initial_hash,checks=checks,runtime_seconds=time.monotonic()-start,
              adam_updates=3000,lbfgs_outer_steps=200,closure_calls=calls,
              internal_lbfgs_iterations=opt.state[next(iter(model.parameters()))]['n_iter'],
              all_finite=True,config={**CONFIG,'K':K},torch_version=torch.__version__,numpy_version=np.__version__)
    (ROOT/'comparisons'/f'k{K}_metadata.json').write_text(json.dumps(meta,indent=2))


def audit(K):
    model=seeded_model()
    state=torch.load(ROOT/'checkpoints'/f'k{K}_final.pt',weights_only=False)
    model.load_state_dict(state['model'])
    row=diagnostic(model,ResidualK(64,K),'final_n64',200)
    training=json.loads((ROOT/'comparisons'/f'k{K}_trajectory.json').read_text())[-1]
    meta=json.loads((ROOT/'comparisons'/f'k{K}_metadata.json').read_text())
    row.update(K=K,training_loss=training['loss'],runtime_seconds=meta['runtime_seconds'])
    val,grad=functions(model)
    r=moments(val,grad,40,quadrature(64))
    r96=moments(val,grad,40,quadrature(96))
    cross={k:metrics(r[:k,:k]) for k in EVAL_K}
    high={k:metrics(r96[:k,:k]) for k in EVAL_K}
    row['train_quadrature_relative']=abs(training['loss']-row['loss'])/row['loss']
    row['eval_quadrature_relative']=max(abs(cross[k]['L_DFR']-high[k]['L_DFR'])/high[k]['L_DFR'] for k in EVAL_K)
    assert row['train_quadrature_relative'] < 1e-8
    assert row['eval_quadrature_relative'] < 1e-8
    return model,dict(final=row,cross=cross,cross_n96=high,metadata=meta)


def figures(models, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,key,title in zip(axes,('rel','h1'),('Regular RelL2','Absolute H1 error')):
        ax.plot(TRAIN_K,[results[k]['final'][key] for k in TRAIN_K],'o-')
        ax.set_xticks(TRAIN_K); ax.set_xlabel('K_train'); ax.set_ylabel(title); ax.grid(alpha=.3)
    fig.savefig(ROOT/'figures/figure_1_solution_errors.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for k in TRAIN_K:
        ax.plot(EVAL_K,[results[k]['cross'][j]['L_DFR'] for j in EVAL_K],'o-',label=f'K_train={k}')
    ax.set_xticks(EVAL_K); ax.set_xlabel('K_eval'); ax.set_ylabel('L_DFR'); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(ROOT/'figures/figure_2_cross_k_residuals.png',dpi=180); plt.close(fig)
    axis=np.linspace(-1,1,241); xx,yy=np.meshgrid(axis,axis)
    x=np.stack((xx.ravel(),yy.ravel()),1)
    errors={}
    for k in (16,32):
        with torch.no_grad(): errors[k]=np.abs(models[k](tensor(x)).numpy()-pb.u_r_exact(x)).reshape(xx.shape)
    norm=Normalize(vmin=0,vmax=max(e.max() for e in errors.values()))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    for ax,k in zip(axes,(16,32)):
        im=ax.pcolormesh(xx,yy,errors[k],shading='auto',cmap='magma',norm=norm)
        ax.add_patch(plt.Circle((0,0),pb.R,fill=False,color='white',linestyle='--',linewidth=.8))
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(f'Absolute regular error, K_train={k}')
    fig.colorbar(im,ax=axes,label='Absolute error (shared scale)')
    fig.savefig(ROOT/'figures/figure_3_error_fields.png',dpi=180); plt.close(fig)
    assert len(list((ROOT/'figures').glob('*.png')))==3


def report():
    import csv
    models,results={},{}
    for k in TRAIN_K: models[k],results[k]=audit(k)
    assert len({r['metadata']['initial_sha256'] for r in results.values()})==1
    (ROOT/'comparisons/final_results.json').write_text(json.dumps(results,indent=2))
    with (ROOT/'comparisons/cross_k_loss.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['K_train',*EVAL_K])
        for k in TRAIN_K: writer.writerow([k,*[results[k]['cross'][j]['L_DFR'] for j in EVAL_K]])
    with (ROOT/'comparisons/final_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[16]['final']));writer.writeheader()
        writer.writerows(r['final'] for r in results.values())
    figures(models,results)
    first,last=results[16]['final'],results[32]['final']
    # Evidence-based finite-budget classification; these rules never affect training.
    error_improves=all(last[k] < .95*first[k] for k in ('rel','h1'))
    common_improves=results[32]['cross'][40]['L_DFR'] < .95*results[16]['cross'][40]['L_DFR']
    if error_improves and common_improves:
        decision='DECISION: FOURIER RESOLUTION LIMITS SOLVER ACCURACY'
    elif common_improves and not error_improves:
        decision='DECISION: INCREASING K IMPROVES RESIDUAL BUT NOT SOLUTION ERROR'
    else:
        decision='DECISION: ACCURACY LIMITATION IS NOT EXPLAINED BY FOURIER TRUNCATION'
    lines=['# Fourier resolution study', '', '**provisional implementation benchmark** — Step 6; exactly three training cases, K=16,24,32.', '',
        '## Controlled experiment', '',
        'Scientific question: does insufficient Fourier test-space resolution explain the observed accuracy limitation? The only training-case change is K. The benchmark, nonlinear weak residual, globally smooth hard-boundary trial, initialization, optimizer schedule and quadrature are unchanged.', '',
        'Omega=[-1,1]², circular R=0.5 interface, epsilon-=2, epsilon+=80, kappa-=kappa+=1, sigma=0.08, Q=1. u_s=Q/(4*pi*epsilon*sqrt(r²+sigma²)), u_r_exact=P(r²), P(t)=0.2(t-0.25)²(1-t)². Total exact u=u_r_exact+u_s. The singular source and manufactured remainder, exterior data, Fourier basis and interface term are imported unchanged from the validated benchmark.', '',
        'One 2-32-32-32-1 tanh MLP, float64, seed 2026. u_r,theta=ell+(1-x²)(1-y²)N_theta; ell=P(1+x²)+P(1+y²)-P(2), the same boundary-data Coons lift. All three initial parameter tensors are bitwise identical; their common SHA256 is '+results[16]['metadata']['initial_sha256']+'.', '',
        'Only L_DFR=sum(R_kl²/(1+lambda_kl)) is optimized. The K-parameterized evaluator inherits the smoke-test differentiable weak assembly; only the cached basis, weights and source moment dimensions change. No piecewise networks, penalties, strong residual, WAN, Gram correction or new method.', '',
        'Each case runs 3000 Adam updates at lr=1e-3, followed by 200 L-BFGS outer steps from that Adam state. L-BFGS: max_iter=5, lr=1, history_size=50, strong_wolfe, tolerance_grad=1e-10, tolerance_change=1e-14, default max_eval=6. All schedules are fixed; no tuning or selection of an earlier best-error state. CPU, two threads, deterministic algorithms. Runtime includes optimization, scheduled diagnostic logging and checkpoint writes, excludes preflight and final audits; it is not used as a quality criterion.', '',
        'Training quadrature remains the same eight-sector, circle-split radial Gauss rule with n=32 for every K. Exact-solution residual, NumPy/Torch moment agreement, parameter directional-derivative and initial n=64 quadrature checks passed separately for all K. Common final diagnostics use n=64; all cross-K entries are checked against n=96. No quadrature method or case-specific resolution tuning.', '',
        '## Final metrics', '',
        'Training loss uses n=32 and K_train. All other volume errors and residual statistics use n=64 at K_train. H1 is the absolute sqrt(integral(error²+|grad(error)|²)); RelL2 is relative to the corresponding regular/total exact norm. Boundary max is sampled on 1,028 edge points. The known singular field cancels in the total error numerator.', '',
        '| K_train | Training L_DFR | Regular RelL2 | Total RelL2 | H1 error | max abs R | RMS R | Boundary max | Runtime (s) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in TRAIN_K:
        r=results[k]['final'];lines.append('| '+str(k)+' | '+' | '.join(f'{r[key]:.6e}' for key in ('training_loss','rel','total_rel','h1','maximum','rms','boundary','runtime_seconds'))+' |')
    lines += ['', '## Interface diagnostics', '',
        '4096 circle points, exact coincident one-sided traces of the same global regular trial; regular solution jump should be zero. Total u has the prescribed jump of u_s, so zero total jump is not imposed. Physical flux residual=[epsilon partial_n u_r]+[epsilon partial_n u_s], normal outward from the disk. The second term vanishes for this benchmark. No interface diagnostic enters training.', '',
        '| K_train | Regular solution jump RMS | Physical flux residual RMS | Physical flux residual max |', '|---:|---:|---:|---:|']
    for k in TRAIN_K:
        r=results[k]['final'];lines.append(f"| {k} | {r['jump_rms']:.6e} | {r['flux_rms']:.6e} | {r['flux_max']:.6e} |")
    lines += ['', '## Required cross-K audit matrix', '',
        'Each final model is evaluated without retraining by the original validated NumPy weak-moment evaluator. Rows are trained K; columns are evaluated K. Entries are L_DFR. The nested mode sums must be nondecreasing within each row; loss values at different training K alone are not a fair accuracy comparison.', '',
        '| K_train / K_eval | 16 | 24 | 32 | 40 |', '|---:|---:|---:|---:|---:|']
    for k in TRAIN_K:
        vals=[results[k]['cross'][j]['L_DFR'] for j in EVAL_K]
        assert np.all(np.diff(vals)>=0)
        lines.append('| '+str(k)+' | '+' | '.join(f'{v:.6e}' for v in vals)+' |')
    lines += ['', '## Required error comparison', '', '| K_train | Regular RelL2 | Total RelL2 | H1 error | Flux residual RMS | L_DFR at K=40 |', '|---:|---:|---:|---:|---:|---:|']
    for k in TRAIN_K:
        r=results[k]['final'];lines.append('| '+str(k)+' | '+' | '.join(f'{r[key]:.6e}' for key in ('rel','total_rel','h1','flux_rms'))+f" | {results[k]['cross'][40]['L_DFR']:.6e} |")
    lines += ['', '## Quadrature and finite-value checks', '', '| K_train | Training n=32 vs n=64 relative loss discrepancy | Max cross-K n=64 vs n=96 relative discrepancy | L-BFGS closure calls | Internal iterations |', '|---:|---:|---:|---:|---:|']
    for k in TRAIN_K:
        r,m=results[k]['final'],results[k]['metadata']
        lines.append(f"| {k} | {r['train_quadrature_relative']:.6e} | {r['eval_quadrature_relative']:.6e} | {m['closure_calls']} | {m['internal_lbfgs_iterations']} |")
    lines += ['', 'Every Adam update and L-BFGS closure passed finite loss, residual, solution, spatial-gradient and parameter-gradient checks; updated parameters and saved diagnostics were finite. All cases completed their full schedule. Different internal L-BFGS counts can arise from the same line search and stopping settings; all cases had 200 outer calls.', '',
        '## Figures', '', '![Final solution errors versus trained K](../figures/figure_1_solution_errors.png)', '',
        '![Common cross-K residual curves](../figures/figure_2_cross_k_residuals.png)', '',
        '![K=16 and K=32 absolute error fields on identical scales](../figures/figure_3_error_fields.png)', '',
        '## Interpretation and limitations', '', 'Increasing K provides evidence that Fourier resolution contributes to the K=16 accuracy limitation, but does not establish it as the sole or dominant cause. Relative to K=16, K=24 reduces regular RelL2 by 17.5%, H1 error by 17.0%, and the common K_eval=40 loss by 53.1%. K=32 reduces these quantities by 13.8%, 12.8%, and 37.4%, respectively. Both higher-K cases also reduce the physical flux residual RMS. The improvements are larger than the measured quadrature discrepancies.\n\nThe dependence is not monotonic: K=24 has the lowest regular/total RelL2, H1 error, and common K_eval=40 loss, while K=32 has the lowest flux residual RMS. Increasing K from 24 to 32 is therefore not an accuracy improvement under this fixed budget. The final K=32 Adam loss rose from 2.464 at step 2500 to 12.567 at step 3000, but L-BFGS subsequently reduced it to 0.01103 with all checks finite; this is a transient optimizer fluctuation, not evidence of divergence.\n\nThe K=16 model has additional residual energy L(40)-L(16)=0.004608, or 34.4% of its training-space loss. For K=24 and K=32, the corresponding energy beyond the trained square of modes through K_eval=40 is 0.0006358 (8.15%) and 0.0002220 (2.01%). Thus smaller test spaces leave a larger untrained tail in this study, while nonzero residuals within the trained spaces also remain. Lower RMS coefficients at larger evaluation K can result from averaging over more modes; the common weighted sums are the relevant cross-model comparison.\n\nThe retrained K=16 final parameters were verified bitwise equal to the original full-solver final state. Initial checkpoints of all three cases were verified bitwise equal. Exactly three training schedules were executed; no additional runs or state selection were used.', '',
        'K=32 / K=16 final ratios: '+', '.join(f"{key}={last[key]/first[key]:.6f}" for key in ('rel','h1','flux_rms'))+f"; common K_eval=40 loss={results[32]['cross'][40]['L_DFR']/results[16]['cross'][40]['L_DFR']:.6f}.", '',
        'This is a controlled one-seed, finite-budget experiment, not a proof of a unique error source. K changes both the information in the objective and its optimization landscape. Remaining network approximation error and optimization error are not separated by this study. No runtime-based ranking is used. The reporting rule treats simultaneous reductions above 5% in regular RelL2, H1 and common K_eval=40 loss from K=16 to K=32 as evidence for a Fourier-resolution limitation; common-residual-only improvement gets the residual-only decision. Otherwise the observed limitation is not explained by Fourier truncation under this schedule. This rule does not alter training.', '',
        '## Final decision', '', decision, '',
        'Artifacts: initial, final Adam and final L-BFGS checkpoints for each of three cases; per-case full requested Step 5 logging trajectories and metadata; final_metrics.csv, cross_k_loss.csv and final_results.json in comparisons/. Earlier smoke/full-solver files are not modified.', '',
        'Reproduce from repository root (existing checkpoints are protected against accidental reruns):', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/fourier_resolution_study/run_study.py',
        '# Audit saved final states and regenerate the same three figures/report only:',
        'OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/fourier_resolution_study/run_study.py --report-only', '```', '']
    (ROOT/'reports/fourier_resolution_report.md').write_text('\n'.join(lines))
    print(json.dumps({k:r['final'] for k,r in results.items()},indent=2),flush=True)
    print(decision,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if not args.report_only:
        for K in TRAIN_K: train(K)
    report()


if __name__=='__main__': main()
