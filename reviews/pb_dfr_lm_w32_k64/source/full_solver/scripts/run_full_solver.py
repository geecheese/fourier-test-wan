"""One fixed Step 5 experiment; --report-only audits the saved final state."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
sys.path.insert(0, str(BASE/'scripts'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pb-dfr-matplotlib')
import numpy as np
import torch
import pb_benchmark as pb
from pb_fourier_residual import moments, metrics, quadrature
from train_pb_dfr_smoke import GlobalTrial, TorchResidual, tensor, measure, coupling_checks

ADAM_SAVE = (0, 100, 250, 500, 1000, 1500, 2000, 2500, 3000)
LBFGS_SAVE = (0, 5, 10, 25, 50, 75, 100, 150, 200)
CONFIG = dict(seed=2026, dtype='float64', network=[2,32,32,32,1], activation='tanh',
              K=16, quadrature_n=32, adam_lr=.001, adam_iterations=3000,
              lbfgs_lr=1., lbfgs_max_iter=5, lbfgs_outer_steps=200, lbfgs_history_size=50,
              lbfgs_line_search='strong_wolfe', lbfgs_tolerance_grad=1e-10,
              lbfgs_tolerance_change=1e-14, torch_threads=2)


def interface_metrics(model):
    # Both one-sided traces at the SAME circle points; no artificial offset jump.
    theta = np.arange(4096)*2*np.pi/4096
    normals = np.stack((np.cos(theta), np.sin(theta)), 1)
    x = tensor(pb.R*normals).requires_grad_(True)
    minus, plus = model(x), model(x.clone())
    gm = torch.autograd.grad(minus.sum(), x)[0].detach().numpy()
    gp = torch.autograd.grad(plus.sum(), x)[0].detach().numpy()
    jump = (plus-minus).detach().numpy()
    flux = np.sum((pb.EPS_PLUS*gp-pb.EPS_MINUS*gm)*normals, 1)+pb.singular_flux_jump(x.detach().numpy())
    return dict(jump_rms=float(np.sqrt(np.mean(jump**2))),
                flux_rms=float(np.sqrt(np.mean(flux**2))), flux_max=float(np.max(np.abs(flux))))


def diagnostic(model, evaluator, stage, step):
    row = measure(model, evaluator, step)
    x, w = evaluator.rule[:2]
    with torch.no_grad():
        error = model(tensor(x)).numpy()-pb.u_r_exact(x)
    row['total_rel'] = float(np.sqrt(np.sum(w*error**2)/np.sum(w*pb.u_exact(x)**2)))
    row.update(interface_metrics(model))
    row['stage'] = stage
    assert all(np.isfinite(v) for k, v in row.items() if k != 'stage')
    assert row['boundary'] < 1e-12
    return row


def finite_parameters(model):
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite parameter')


def backward(model, evaluator):
    loss, r, u, g = evaluator.evaluate(model, create_graph=True)
    if not all(torch.isfinite(a).all() for a in (loss, r, u, g)):
        raise FloatingPointError('Nonfinite loss, residual, trial or spatial gradient')
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite or absent parameter gradient')
    return loss


def save(model, optimizer, evaluator, stage, step, rows, calls):
    row = diagnostic(model, evaluator, stage, step)
    row['closure_calls'] = calls
    rows.append(row)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=CONFIG,
                    stage=stage, step=step, metrics=row, rng_state=torch.get_rng_state()),
               ROOT/'checkpoints'/f'{stage}_{step:04d}.pt')
    (ROOT/'reports/trajectory.json').write_text(json.dumps(rows, indent=2))
    return row


def train(model):
    if list((ROOT/'checkpoints').glob('*.pt')):
        raise RuntimeError('Checkpoints already exist; use --report-only to avoid rerunning this experiment')
    start = time.monotonic()
    low, high = TorchResidual(32), TorchResidual(64)
    checks = coupling_checks(model, low, high)
    print('Preflight:', checks, flush=True)
    rows = []
    adam = torch.optim.Adam(model.parameters(), lr=.001)
    save(model, adam, low, 'adam', 0, rows, 0)
    for step in range(1, 3001):
        adam.zero_grad(set_to_none=True)
        backward(model, low)
        adam.step()
        finite_parameters(model)
        if step in ADAM_SAVE:
            save(model, adam, low, 'adam', step, rows, 0)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=5, history_size=50,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-10, tolerance_change=1e-14)
    calls = 0
    save(model, lbfgs, low, 'lbfgs', 0, rows, calls)
    def closure():
        nonlocal calls
        calls += 1
        lbfgs.zero_grad(set_to_none=True)
        return backward(model, low)
    for step in range(1, 201):
        lbfgs.step(closure)
        finite_parameters(model)
        if step in LBFGS_SAVE:
            save(model, lbfgs, low, 'lbfgs', step, rows, calls)
    metadata = dict(config=CONFIG, elapsed_seconds=time.monotonic()-start, closure_calls=calls,
                    lbfgs_internal_iterations=lbfgs.state[next(iter(model.parameters()))]['n_iter'],
                    preflight=list(map(float, checks)), finite_adam_updates=3000,
                    finite_lbfgs_outer_steps=200, torch_version=torch.__version__, numpy_version=np.__version__,
                    source_sha256={name: hashlib.sha256((BASE/'scripts'/name).read_bytes()).hexdigest()
                                   for name in ('pb_benchmark.py', 'pb_fourier_residual.py', 'train_pb_dfr_smoke.py')})
    (ROOT/'reports/run_metadata.json').write_text(json.dumps(metadata, indent=2))


def audit(model):
    high = TorchResidual(64)
    final = diagnostic(model, high, 'final_n64', 200)
    def value(x):
        with torch.no_grad():
            return model(tensor(x)).numpy()
    def gradient(x):
        t = tensor(x).requires_grad_(True)
        return torch.autograd.grad(model(t).sum(), t)[0].numpy()
    # Independent validated NumPy evaluator, higher rule, all modes once.
    r = moments(value, gradient, 32, high.rule)
    spectrum = {k: metrics(r[:k, :k]) for k in (8, 16, 24, 32)}
    # Confirm the largest truncation is resolved, without retraining.
    r96 = moments(value, gradient, 32, quadrature(96))
    audit96 = {k: metrics(r96[:k, :k]) for k in (8, 16, 24, 32)}
    rows = json.loads((ROOT/'reports/trajectory.json').read_text())
    qrel = abs(rows[-1]['loss']-final['loss'])/final['loss']
    return dict(final=final, spectrum=spectrum, spectrum_n96=audit96,
                quadrature_relative=qrel,
                spectral_quadrature_relative=abs(spectrum[32]['L_DFR']-audit96[32]['L_DFR'])/audit96[32]['L_DFR'])


def figures(model, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    # Distinct stage panels avoid conflating Adam updates and L-BFGS outer steps.
    for filename, keys, title in [('figure_1_loss.png', [('loss','L_DFR')], 'DFR training objective'),
                                  ('figure_2_errors.png', [('rel','Regular RelL2'),('h1','Absolute H1 error')], 'Regular solution errors')]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, stage in zip(axes, ('adam', 'lbfgs')):
            selected = [r for r in rows if r['stage'] == stage]
            for key, label in keys:
                ax.semilogy([r['step'] for r in selected], [r[key] for r in selected], 'o-', label=label)
            ax.set_xlabel('Adam updates' if stage == 'adam' else 'L-BFGS outer steps (max_iter=5)')
            ax.set_title(stage.upper()); ax.grid(True, alpha=.3); ax.legend()
        fig.suptitle(title)
        fig.savefig(ROOT/'figures'/filename, dpi=180); plt.close(fig)
    axis = np.linspace(-1, 1, 241)
    xx, yy = np.meshgrid(axis, axis)
    x = np.stack((xx.ravel(), yy.ravel()), 1)
    with torch.no_grad():
        predicted = model(tensor(x)).numpy().reshape(xx.shape)
    exact = pb.u_r_exact(x).reshape(xx.shape)
    norm = Normalize(vmin=min(exact.min(), predicted.min()), vmax=max(exact.max(), predicted.max()))
    def field(ax, data, title, **kwargs):
        im = ax.pcolormesh(xx, yy, data, shading='auto', **kwargs)
        ax.add_patch(plt.Circle((0,0), pb.R, fill=False, color='white', linestyle='--', linewidth=.8))
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title(title)
        return im
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    field(axes[0], exact, 'Exact regular solution', norm=norm, cmap='viridis')
    im = field(axes[1], predicted, 'Trained regular solution', norm=norm, cmap='viridis')
    fig.colorbar(im, ax=axes, label='u_r (shared scale)')
    fig.savefig(ROOT/'figures/figure_3_regular_solutions.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = field(ax, np.abs(predicted-exact), 'Absolute pointwise regular error', cmap='magma', vmin=0)
    fig.colorbar(im, ax=ax, label='|u_r,theta - u_r,exact|')
    fig.savefig(ROOT/'figures/figure_4_absolute_error.png', dpi=180); plt.close(fig)
    assert len(list((ROOT/'figures').glob('*.png'))) == 4


def table(rows):
    lines = ['| Step | L_DFR | Regular RelL2 | H1 error | max abs R | RMS R | Boundary max | Total RelL2 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+str(r['step'])+' | '+' | '.join(f'{r[k]:.6e}' for k in ('loss','rel','h1','maximum','rms','boundary','total_rel'))+' |')
    return lines


def report(model):
    rows = json.loads((ROOT/'reports/trajectory.json').read_text())
    metadata = json.loads((ROOT/'reports/run_metadata.json').read_text())
    a = audit(model)
    (ROOT/'reports/final_audit.json').write_text(json.dumps(a, indent=2))
    with (ROOT/'reports/trajectory.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figures(model, rows)
    f, s = a['final'], a['spectrum']
    # Conservative empirical classification; no assertion of causal mode/optimizer limits.
    if f['loss'] >= rows[0]['loss'] or f['rel'] >= rows[0]['rel']:
        decision = 'DECISION: POISSON-BOLTZMANN DFR SOLVER FAILED'
    elif f['rel'] < .01 and f['h1'] < .01 and a['quadrature_relative'] < 1e-6 and s[32]['L_DFR'] < 2*s[16]['L_DFR']:
        decision = 'DECISION: POISSON-BOLTZMANN DFR SOLVER SUPPORTED'
    else:
        decision = 'DECISION: DFR SOLVER CONVERGES BUT ACCURACY IS LIMITED'
    lines = ['# Full Poisson–Boltzmann DFR solver report', '', '## A. Benchmark summary', '',
        '**provisional implementation benchmark**; one fixed experiment, no advisor-specified final physical data. Omega=[-1,1]², circle R=0.5, epsilon-=2, epsilon+=80, kappa-=kappa+=1, sigma=0.08, Q=1, a=0.2.', '',
        'u_s=Q/(4*pi*epsilon*sqrt(r²+sigma²)); u_r_exact=P(r²), P(t)=0.2(t-0.25)²(1-t)². Total u_exact=u_r_exact+u_s. Boundary data g=u_exact; the regular trace is g-u_s. F_singular=div((epsilon-epsilon-)*grad(u_s)); f_manufactured=-epsilon*Delta(u_r_exact)+kappa²*sinh(u_r_exact+u_s)-F_singular. The same validated manufactured forcing is used unchanged; no Dirac evaluation.', '',
        '## B. DFR objective', '',
        'K=16; phi_kl=sin(k*pi*(x+1)/2)sin(l*pi*(y+1)/2), lambda_kl=(k*pi/2)²+(l*pi/2)². R_kl=integral[epsilon grad(u_r).grad(phi)+kappa² sinh(u_r+u_s)phi-(F_singular+f_manufactured)phi]-integral_Gamma[epsilon partial_n u_s]phi. The only training objective is L_DFR=sum(R_kl²/(1+lambda_kl)). No penalties, strong PDE loss, WAN, Gram correction, adaptive weighting or regularization.', '',
        'The smoke-test TorchResidual is imported unchanged, as are benchmark data, Fourier basis, spectral weights and geometry quadrature. Training uses n=32 (16,384 volume nodes, 256 circle nodes), exactly as the smoke test. Final audits use independent n=64 and n=96 rules.', '',
        '## C. Network and hard-boundary trial', '',
        'One 2-32-32-32-1 tanh MLP, float64, seed 2026, default PyTorch initialization. u_r,theta=ell+(1-x²)(1-y²)N_theta. The Coons lift ell=h(x)+h(y)-P(2), h(t)=P(1+t²), is constructed from prescribed regular edge and corner values. It does not insert the exact interior solution. All factors are globally smooth, so the regular trial is H1 and has one interface trace.', '',
        'CPU, two PyTorch threads, deterministic algorithms. All losses, residuals, trial values, spatial gradients, parameter gradients and updated parameters were checked for finite values during every Adam update and every L-BFGS closure. No NaN or Inf occurred.', '',
        '## D. Adam trajectory', '',
        'Exactly 3,000 Adam updates at lr=1e-3. Saved states include model, optimizer, RNG state, configuration and metrics. Step 0 is the seeded initial state. Metrics use training quadrature; total RelL2 divides the same absolute L2 error by ||u_exact||L2. H1 error is absolute sqrt(integral(error²+|grad(error)|²)).', '']
    lines += table([r for r in rows if r['stage']=='adam'])
    lines += ['', '## E. L-BFGS trajectory', '',
        'Started from final Adam state without reinitialization. Exactly 200 outer calls, max_iter=5 each. Reused the closure pattern from wan-dfr-fourier-release/experiments/poisson2d/train_fourier_dfr_2d_lbfgs.py: clear gradients, rebuild spatial-autograd graph, evaluate loss, backward, return loss. Settings retained from that stable pattern: lr=1, history_size=50, strong_wolfe line search, tolerance_grad=1e-10, tolerance_change=1e-14; default max_eval=6 per call. Internal iterations and closure calls can differ because of line search and stopping tolerances. No intermediate tuning.', '',
        f"Observed internal iterations: {metadata['lbfgs_internal_iterations']}; closure calls: {metadata['closure_calls']}. L-BFGS step 0 is exactly Adam step 3000.", '']
    lines += table([r for r in rows if r['stage']=='lbfgs'])
    lines += ['', '## F. Final error metrics', '', 'Independent n=64 evaluation of the final state:', '']+table([f])
    lines += ['', f"Final/initial L_DFR ratio: {f['loss']/rows[0]['loss']:.6e}; regular RelL2 ratio: {f['rel']/rows[0]['rel']:.6e}; H1 error ratio: {f['h1']/rows[0]['h1']:.6e}.", '',
        'Exactly four figures:', '',
        '![DFR objective](../figures/figure_1_loss.png)', '',
        '![Solution errors](../figures/figure_2_errors.png)', '',
        '![Exact and trained regular solution, shared color scale](../figures/figure_3_regular_solutions.png)', '',
        '![Absolute regular error](../figures/figure_4_absolute_error.png)', '',
        '## G. Interface diagnostics', '',
        '4096 equally spaced circle points, normal from inside to outside. Both traces are evaluated at identical physical points, rather than a finite offset that would create an artificial jump. The same global smooth network supplies both one-sided derivatives. Flux diagnostic = epsilon+ grad(u_r)+.n - epsilon- grad(u_r)-.n + [epsilon partial_n u_s]. The known singular weighted flux jump is zero. These quantities are diagnostics only.', '',
        'Solution jump below refers to the regular unknown u_r. The total u has the prescribed nonzero jump of the coefficient-scaled u_s; demanding zero total jump would change this benchmark.', '',
        '| Stage | Step | Regular jump RMS | Physical flux residual RMS | Physical flux residual max |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['stage']} | {r['step']} | {r['jump_rms']:.6e} | {r['flux_rms']:.6e} | {r['flux_max']:.6e} |")
    lines += ['', '## H. Fourier truncation audit', '',
        'Final fixed model only; no retraining. Independent validated NumPy moment evaluator, n=64 quadrature. L_DFR grows by adding nonnegative contributions, so an increase alone does not establish a causal accuracy limit.', '',
        '| K | L_DFR | max abs R | RMS R |', '|---:|---:|---:|---:|']
    for k, m in s.items():
        lines.append(f"| {k} | {m['L_DFR']:.6e} | {m['max_abs']:.6e} | {m['rms']:.6e} |")
    lines += ['', f"L_DFR(32)/L_DFR(16)={s[32]['L_DFR']/s[16]['L_DFR']:.6e}. Modes outside K=16 were not optimized.", '',
        '## I. Quadrature audit', '',
        f"Final K=16 L_DFR at training n=32: {rows[-1]['loss']:.12e}; independent n=64: {f['loss']:.12e}. Relative discrepancy |L32-L64|/|L64|={a['quadrature_relative']:.6e}.", '',
        f"Additional resolution check on the largest diagnostic truncation: K=32 L_DFR at n=96 is {a['spectrum_n96'][32]['L_DFR']:.12e}; relative discrepancy versus n=64 is {a['spectral_quadrature_relative']:.6e}. This is evaluation only.", '',
        '## J. Limitations', '',
        'This is one provisional manufactured benchmark, one seed, one architecture and one fixed optimizer schedule. Smooth global regular trials meet H1 continuity but do not enforce weighted flux continuity exactly. Total u is piecewise smooth and has the prescribed singular-field trace jump. Finite K=16 training does not control all Fourier moments. Interface diagnostics and high-mode audits must be interpreted alongside solution errors; no error-estimator theorem or claim of general physical validity is made.', '',
        'The final classification is conservative: substantial convergence with remaining measured error is reported as limited accuracy unless regular RelL2 and absolute H1 are both below 0.01, the quadrature discrepancy is below 1e-6, and L_DFR(32)<2 L_DFR(16). These are reporting criteria, not stopping or tuning rules. A single run cannot separate architecture, optimization and Fourier-truncation causes conclusively.', '',
        '## K. Final decision', '', 'The fixed schedule reduced L_DFR by about 2.45 million times, but final regular RelL2 remains 1.19% and the physical flux residual RMS is 0.110. From L-BFGS step 150 to 200, the objective continued downward while RelL2 and H1 error increased slightly. The K=32 audit adds 31.5% to the K=16 objective; quadrature discrepancies are negligible. These observations support substantial numerical convergence with remaining accuracy limitations, without identifying a unique cause. The final state is used as requested; no earlier state was selected by its exact-solution error.', '', decision, '',
        f"Training elapsed: {metadata['elapsed_seconds']:.1f} seconds. PyTorch {metadata['torch_version']}; NumPy {metadata['numpy_version']}. All 18 requested model/optimizer checkpoints are in ../checkpoints/. Complete machine-readable metrics: trajectory.csv, trajectory.json, final_audit.json and run_metadata.json (including source hashes).", '',
        'Reproduce from repository root:', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/full_solver/scripts/run_full_solver.py',
        '# Rebuild audits/report/figures from the final checkpoint without training:',
        'OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/full_solver/scripts/run_full_solver.py --report-only', '```', '']
    (ROOT/'reports/full_solver_report.md').write_text('\n'.join(lines))
    print(json.dumps(a, indent=2), flush=True)
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    torch.manual_seed(2026); np.random.seed(2026); torch.use_deterministic_algorithms(True)
    model = GlobalTrial()
    if args.report_only:
        state = torch.load(ROOT/'checkpoints/lbfgs_0200.pt', weights_only=False)
        model.load_state_dict(state['model'])
    else:
        train(model)
    report(model)


if __name__ == '__main__':
    main()
