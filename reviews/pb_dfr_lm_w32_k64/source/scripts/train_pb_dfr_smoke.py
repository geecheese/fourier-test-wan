"""Step 4 only: one global hard-boundary MLP, fixed 500-step Adam smoke test."""
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import pb_benchmark as pb
from pb_fourier_residual import quadrature, basis, spectral_data, moments

K, N, STEPS = 16, 32, 500
DTYPE = torch.float64
REPORT = Path(__file__).resolve().parents[1]/'reports/pb_dfr_smoke.md'


def tensor(a):
    return torch.as_tensor(a, dtype=DTYPE)


def polynomial(t):
    return pb.A*(t-pb.R**2)**2*(1-t)**2


class GlobalTrial(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32),
                                 nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.double()

    def forward(self, x):
        # Coons extension from four prescribed regular boundary traces:
        # opposite edges have h(t)=P(1+t²), all corner values are P(2).
        # Linear edge blending minus bilinear corner blending reduces to this.
        # No interior exact-solution lifting is used.
        lifting = polynomial(1+x[:, 0]**2)+polynomial(1+x[:, 1]**2)-polynomial(2.)
        B = (1-x[:, 0]**2)*(1-x[:, 1]**2)
        return lifting+B*self.net(x).squeeze(-1)


class TorchResidual:
    """Differentiable port of the validated volume-source contractions."""
    def __init__(self, n):
        self.rule = quadrature(n)
        x, w, c, wc = self.rule
        self.x, self.w = tensor(x), tensor(w)
        self.sx, self.sy, self.dx, self.dy = map(tensor, basis(x, K))
        self.eps, self.k2, self.us = map(tensor, (pb.epsilon(x), pb.kappa(x)**2, pb.u_s(x)))
        self.weights = tensor(spectral_data(K)[1])
        self.source = self.project(self.sx, self.sy, tensor(pb.manufactured_remainder(x)))
        self.source += self.project(self.sx, self.sy, tensor(pb.singular_source(x)))
        cx, cy, _, _ = basis(c, K)
        self.interface = tensor(cx.T @ ((wc*pb.singular_flux_jump(c))[:, None]*cy))

    def project(self, a, b, f):
        return a.T @ ((self.w*f)[:, None]*b)

    def evaluate(self, trial, create_graph=False):
        x = self.x.detach().requires_grad_(True)
        u = trial(x)
        g = torch.autograd.grad(u.sum(), x, create_graph=create_graph)[0]
        r = self.project(self.dx, self.sy, self.eps*g[:, 0])
        r = r+self.project(self.sx, self.dy, self.eps*g[:, 1])
        r = r+self.project(self.sx, self.sy, self.k2*torch.sinh(u+self.us))-self.source-self.interface
        return (r.square()*self.weights).sum(), r, u, g


def boundary_points():
    t = np.linspace(-1, 1, 257)
    return np.concatenate([np.stack((t, np.full_like(t, s)), 1) for s in (-1, 1)]+
                          [np.stack((np.full_like(t, s), t), 1) for s in (-1, 1)])


def measure(model, evaluator, step):
    loss, r, u, g = evaluator.evaluate(model)
    x, w = evaluator.rule[:2]
    err = u.detach().numpy()-pb.u_r_exact(x)
    dg = g.detach().numpy()-pb.grad_exact(x)
    e2 = np.sum(w*err**2)
    bp = boundary_points()
    with torch.no_grad():
        boundary = np.max(np.abs(model(tensor(bp)).numpy()-(pb.boundary_data(bp)-pb.u_s(bp))))
    row = dict(step=step, loss=loss.item(), rel=float(np.sqrt(e2/np.sum(w*pb.u_r_exact(x)**2))),
               h1=float(np.sqrt(e2+np.sum(w*np.sum(dg**2, axis=1)))), boundary=float(boundary),
               maximum=r.detach().abs().max().item(), rms=r.detach().square().mean().sqrt().item())
    if not all(np.isfinite(v) for v in row.values()):
        raise FloatingPointError('Nonfinite diagnostic')
    print(row, flush=True)
    return row


def coupling_checks(model, low, high):
    exact = lambda x: polynomial((x*x).sum(1))
    _, r, _, _ = low.evaluate(exact)
    exact_max = r.detach().abs().max().item()
    assert exact_max < 1e-9, f'Exact residual {exact_max}'
    def value(x):
        with torch.no_grad():
            return model(tensor(x)).numpy()
    def gradient(x):
        t = tensor(x).requires_grad_(True)
        return torch.autograd.grad(model(t).sum(), t)[0].numpy()
    loss, r, _, _ = low.evaluate(model, True)
    reference = moments(value, gradient, K, low.rule)
    port_error = np.max(np.abs(reference-r.detach().numpy()))
    assert port_error < 1e-9
    # Directional derivative through both spatial-gradient and sinh terms.
    parameters = list(model.parameters())
    direction = [torch.randn_like(p) for p in parameters]
    norm = torch.sqrt(sum(d.square().sum() for d in direction))
    direction = [d/norm for d in direction]
    grads = torch.autograd.grad(loss, parameters)
    ad = sum((g*d).sum() for g, d in zip(grads, direction)).item()
    original = [p.detach().clone() for p in parameters]
    values = []
    h = 1e-5
    for sign in (1, -1):
        with torch.no_grad():
            for p, base, d in zip(parameters, original, direction):
                p.copy_(base+sign*h*d)
        values.append(low.evaluate(model)[0].item())
    with torch.no_grad():
        for p, base in zip(parameters, original):
            p.copy_(base)
    fd = (values[0]-values[1])/(2*h)
    derivative_error = abs(fd-ad)/max(1., abs(fd), abs(ad))
    assert derivative_error < 1e-6
    low_loss, high_loss = low.evaluate(model)[0].item(), high.evaluate(model)[0].item()
    qrel = abs(low_loss-high_loss)/abs(high_loss)
    assert qrel < 1e-8
    return exact_max, port_error, derivative_error, qrel


def main():
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.use_deterministic_algorithms(True)
    model = GlobalTrial()
    low, high = TorchResidual(N), TorchResidual(64)
    rows, checks, final_qrel = [], None, None
    decision = 'DECISION: FOURIER RESIDUAL COUPLING NEEDS CORRECTION'
    failure = None
    start = time.monotonic()
    finite_steps = 0
    try:
        checks = coupling_checks(model, low, high)
        print('Coupling checks:', checks, flush=True)
        rows.append(measure(model, low, 0))
        decision = 'DECISION: HARD-BOUNDARY TRIAL NEEDS CORRECTION'
        assert rows[0]['boundary'] < 1e-12
        decision = 'DECISION: DFR OPTIMIZATION IS UNSTABLE'
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(1, STEPS+1):
            optimizer.zero_grad(set_to_none=True)
            loss, residual, _, _ = low.evaluate(model, create_graph=True)
            if not torch.isfinite(loss) or not torch.isfinite(residual).all():
                raise FloatingPointError('Nonfinite loss or residual')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite or absent parameter gradient')
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameter')
            finite_steps += 1
            if step % 50 == 0:
                rows.append(measure(model, low, step))
                if rows[-1]['boundary'] >= 1e-12:
                    decision = 'DECISION: HARD-BOUNDARY TRIAL NEEDS CORRECTION'
                    raise RuntimeError('Boundary tolerance exceeded')
        final_high = measure(model, high, STEPS)
        final_qrel = abs(rows[-1]['loss']-final_high['loss'])/final_high['loss']
        if final_qrel >= 1e-8:
            decision = 'DECISION: FOURIER RESIDUAL COUPLING NEEDS CORRECTION'
            raise RuntimeError('Training/evaluation quadrature discrepancy exceeds 1e-8')
        first, last = rows[0], rows[-1]
        assert last['loss'] < .1*first['loss'], 'Loss did not decrease by at least 10x'
        assert all(last[key] < first[key] for key in ('rel', 'h1', 'maximum', 'rms'))
        decision = 'DECISION: DFR NEURAL SOLVER SMOKE TEST PASSED'
    except (AssertionError, FloatingPointError, RuntimeError) as exc:
        failure = str(exc) or type(exc).__name__
    elapsed = time.monotonic()-start
    lines = ['# PB DFR neural smoke test', '', '**provisional implementation benchmark**; Step 4 only.', '',
        'One global float64 MLP: 2 → 32 → 32 → 32 → 1, tanh hidden activations, PyTorch default Linear initialization, seed 2026. CPU, two PyTorch threads, deterministic algorithms enabled.', '',
        'Trial: u_r,theta=ell+(1-x²)(1-y²)N_theta. With P(t)=0.2(t-0.25)²(1-t)², each prescribed regular edge trace is h(t)=P(1+t²). The Coons boundary extension simplifies to ell(x,y)=h(x)+h(y)-P(2). Thus ell is constructed from edge/corner data, not the exact interior solution. The polynomial lift, boundary factor and single tanh network are smooth throughout the square, including Gamma; the trial belongs to H1 and has a single interface trace.', '',
        'Objective: ONLY sum(R_kl²/(1+lambda_kl)), K=16. Fixed Adam learning rate 1e-3, 500 updates; no scheduler, clipping, regularization, penalties, strong PDE objective, WAN or Gram correction. No checkpoints or architecture/learning-rate tuning.', '',
        'Reused pb_benchmark.py data and pb_fourier_residual.py geometry rules, sine basis and spectral weights. Newly implemented PyTorch contractions preserve the validated epsilon gradient term, nonlinear sinh term, both fixed volume sources, and explicit singular flux-jump term. Previous validation files are unchanged.', '',
        'Training quadrature: n=32, 16,384 volume nodes and 256 circle nodes. Independently compare n=64 (65,536 volume nodes) before and after training. This resolution is accepted only after exact residual, NumPy/Torch coupling and quadrature checks pass.', '',
        '## Coupling checks', '']
    if checks is not None:
        lines += [f'- Exact regular trial max residual at training quadrature: {checks[0]:.6e}.',
                  f'- Maximum NumPy/Torch moment difference for initialized neural trial: {checks[1]:.6e}.',
                  f'- Autograd vs central finite-difference parameter directional derivative relative error (h=1e-5): {checks[2]:.6e}.',
                  f'- Initial relative L_DFR discrepancy, n=32 versus n=64: {checks[3]:.6e}.']
    lines += ['', '## Training metrics', '',
        'Iteration 0 precedes optimization; other rows follow that many updates. RelL2=||u_r,theta-u_r_exact||L2/||u_r_exact||L2. H1 error=sqrt(integral(error²+|grad(error)|²)), absolute, on the whole square. Boundary error uses 1,028 edge points including corners. All 50-step logs are retained.', '',
        '| Iteration | L_DFR | RelL2 | H1 error | Boundary max error | max abs R | RMS R |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+str(row['step'])+' | '+' | '.join(f'{row[k]:.6e}' for k in ('loss', 'rel', 'h1', 'boundary', 'maximum', 'rms'))+' |')
    lines += ['', f'Finite loss, residual, parameter-gradient and updated-parameter checks passed for {finite_steps}/500 updates. Logged error metrics were also checked for NaN/Inf.', '']
    if final_qrel is not None:
        lines += [f'Final relative L_DFR quadrature discrepancy: {final_qrel:.6e}.',
                  f"Independent n=64 final metrics: L_DFR={final_high['loss']:.6e}, RelL2={final_high['rel']:.6e}, H1={final_high['h1']:.6e}, max abs R={final_high['maximum']:.6e}, RMS R={final_high['rms']:.6e}.", '']
    if len(rows) > 1:
        lines += ['Final/initial metric ratios: '+', '.join(f"{key}={rows[-1][key]/rows[0][key]:.6e}" for key in ('loss', 'rel', 'h1', 'maximum', 'rms'))+'.', '']
    if failure:
        lines += ['Failure: '+failure+'. No tuning or additional training was attempted.', '']
    lines += ['The predeclared smoke-test criterion is at least a tenfold loss reduction, decreases in both solution errors and both residual statistics, finite optimization, exact boundary trace and consistent independent quadrature. This is a finite-mode smoke test, not a final accuracy result or convergence theorem.', '',
              f'Elapsed run time: {elapsed:.1f} seconds. PyTorch {torch.__version__}; NumPy {np.__version__}.', '',
              'Reproduce from repository root:', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 python -B wan-master/poisson_boltzmann_dfr/scripts/train_pb_dfr_smoke.py', '```', '', decision, '']
    REPORT.write_text('\n'.join(lines))
    print(decision, flush=True)
    if failure:
        raise SystemExit(failure)


if __name__ == '__main__':
    main()
