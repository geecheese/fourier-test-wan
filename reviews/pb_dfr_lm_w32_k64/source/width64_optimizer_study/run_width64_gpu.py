"""Future width-64 PB training on explicit CUDA; never falls back to CPU."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'capacity_study'), str(BASE / 'fourier_resolution_study'),
                str(BASE / 'full_solver/scripts'), str(BASE / 'scripts')]
import run_capacity_study as cap
from run_full_solver import backward, finite_parameters
from train_pb_dfr_smoke import tensor, boundary_points
from pb_fourier_residual import moments, metrics
import pb_benchmark as pb

OUT = Path(__file__).resolve().parent
SOURCE = BASE / 'capacity_study/checkpoints/B_adam.pt'
DIMENSIONS = [2, 64, 64, 64, 1]
DEVICE = None

def sync():
    torch.cuda.synchronize(DEVICE)

def require_cuda(name):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: GPU training is required; no CPU fallback')
    device = torch.device(name)
    if device.type != 'cuda' or device.index is None or device.index >= torch.cuda.device_count():
        raise RuntimeError(f'Invalid explicit CUDA device: {name}')
    torch.empty(1,device=device,dtype=torch.float64)
    return device

def assert_cuda_tensor(value, label):
    assert value.device == DEVICE and value.dtype == torch.float64, (label,value.device,value.dtype)

class CudaResidualK(cap.study.ResidualK):
    def __init__(self, n, K):
        super().__init__(n,K)
        for name in ('x','w','sx','sy','dx','dy','eps','k2','us','weights','source','interface'):
            value=getattr(self,name).to(DEVICE)
            assert_cuda_tensor(value,name)
            setattr(self,name,value)

def check_optimizer_state(optimizer):
    def walk(obj):
        if torch.is_tensor(obj):
            assert obj.device == DEVICE, ('optimizer state',obj.device)
        elif isinstance(obj,dict):
            for value in obj.values(): walk(value)
        elif isinstance(obj,(tuple,list)):
            for value in obj: walk(value)
    walk(optimizer.state)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def csv_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def model_from(state):
    model = cap.model_for('B')
    model.load_state_dict(state['model'])
    model.to(DEVICE)
    for p in model.parameters(): assert_cuda_tensor(p,'model parameter')
    return model

def setup(verify=False, device='cuda:0', result=None):
    global DEVICE
    DEVICE=require_cuda(device)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if not verify:
        assert result is not None and not result.exists(), 'A new output directory is required'
    state = torch.load(SOURCE, map_location='cpu', weights_only=False)
    meta = json.loads((BASE/'capacity_study/reports/B_metadata.json').read_text())
    assert state['stage'] == 'adam' and state['step'] == 3000
    assert state['config'] == meta['config']
    assert state['config']['network'] == DIMENSIONS and state['config']['K'] == 24
    assert state['config']['seed'] == 2026 and state['config']['quadrature_n'] == 32
    assert state['config']['dtype'] == 'float64' and state['config']['adam_lr'] == .001
    assert all(int(v['step']) == 3000 for v in state['optimizer']['state'].values())
    assert len(state['optimizer']['state']) == 8
    initial = cap.model_for('B')
    original = torch.load(BASE/'capacity_study/checkpoints/B_initial.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(v, original['model'][k]) for k,v in initial.state_dict().items())
    assert cap.study.state_hash(initial) == meta['initial_sha256']
    model = model_from(state)
    assert sum(p.numel() for p in model.parameters()) == 8577
    low, high = CudaResidualK(32,24), CudaResidualK(64,24)
    assert low.weights.numel() == 576
    return state, meta, low, high

def setup_verify():
    return setup(verify=True)

def interface_metrics_cuda(model):
    theta=np.arange(4096)*2*np.pi/4096
    normals=np.stack((np.cos(theta),np.sin(theta)),1)
    x=tensor(pb.R*normals).to(DEVICE).requires_grad_(True)
    minus,plus=model(x),model(x.clone())
    gm=torch.autograd.grad(minus.sum(),x)[0].detach().cpu().numpy()
    gp=torch.autograd.grad(plus.sum(),x)[0].detach().cpu().numpy()
    jump=(plus-minus).detach().cpu().numpy()
    flux=np.sum((pb.EPS_PLUS*gp-pb.EPS_MINUS*gm)*normals,1)+pb.singular_flux_jump(x.detach().cpu().numpy())
    return dict(jump_rms=float(np.sqrt(np.mean(jump**2))),flux_rms=float(np.sqrt(np.mean(flux**2))),
                flux_max=float(np.max(np.abs(flux))))

def numpy_functions(model):
    def value(x):
        with torch.no_grad(): return model(tensor(x).to(DEVICE)).detach().cpu().numpy()
    def gradient(x):
        t=tensor(x).to(DEVICE).requires_grad_(True)
        return torch.autograd.grad(model(t).sum(),t)[0].detach().cpu().numpy()
    return value,gradient

def evaluate(model, low, high, step, branch, k40=True):
    sync()
    started = time.monotonic()
    loss, r, _, _ = low.evaluate(model)
    _, _, u, g = high.evaluate(model)
    x, w = high.rule[:2]
    error = u.detach().cpu().numpy() - pb.u_r_exact(x)
    dg = g.detach().cpu().numpy() - pb.grad_exact(x)
    e2 = float(np.sum(w*error**2))
    result = dict(branch=branch, step=step, absolute_l2=float(np.sqrt(e2)),
                  absolute_h1=float(np.sqrt(e2+np.sum(w*np.sum(dg**2,axis=1)))),
                  regular_rell2=float(np.sqrt(e2/np.sum(w*pb.u_r_exact(x)**2))),
                  total_rell2=float(np.sqrt(e2/np.sum(w*pb.u_exact(x)**2))),
                  DFR24=loss.item(), DFR40=None)
    result.update(interface_metrics_cuda(model))
    bp = boundary_points()
    with torch.no_grad():
        result['boundary_max'] = float(np.max(np.abs(model(tensor(bp).to(DEVICE)).detach().cpu().numpy()-(pb.boundary_data(bp)-pb.u_s(bp)))))
    if k40:
        value, gradient = numpy_functions(model)
        def chunks(fn, points):
            return np.concatenate([fn(a) for a in np.array_split(points,max(1,(len(points)+8191)//8192))])
        result['DFR40'] = float(metrics(moments(lambda a:chunks(value,a),lambda a:chunks(gradient,a),40,high.rule))['L_DFR'])
    assert result['boundary_max'] < 1e-12
    assert all(np.isfinite(v) for v in result.values() if isinstance(v,(float,int)))
    sync()
    result['diagnostic_seconds'] = time.monotonic()-started
    return result

def vector(params):
    return torch.cat([p.detach().flatten() for p in params])

def assign(params, theta):
    with torch.no_grad():
        offset = 0
        for p in params:
            p.copy_(theta[offset:offset+p.numel()].view_as(p)); offset += p.numel()

def residual(model, low, params, jacobian=False):
    loss,r,u,g = low.evaluate(model, create_graph=jacobian)
    if not all(torch.isfinite(v).all() for v in (loss,r,u,g,torch.sinh(u+low.us))):
        raise FloatingPointError('Nonfinite residual evaluation')
    rho = (r*low.weights.sqrt()).flatten()
    assert_cuda_tensor(rho,'rho')
    if not jacobian:
        return rho.detach()
    m = rho.numel()
    eye = torch.eye(m, dtype=rho.dtype, device=DEVICE)
    rows = []
    for i in range(0,m,8):
        gs = torch.autograd.grad(rho,params,grad_outputs=eye[i:i+8],is_grads_batched=True,retain_graph=True)
        rows.append(torch.cat([v.reshape(len(eye[i:i+8]),-1) for v in gs],1))
    J = torch.cat(rows).detach()
    assert J.shape == (576,8577) and torch.isfinite(J).all()
    assert_cuda_tensor(J,'J')
    return rho.detach(), J

def preflight(state,low,high):
    model = model_from(state)
    params = list(model.parameters())
    rho,J = residual(model,low,params,True)
    assert rho.shape == (576,) and torch.isfinite(rho).all()
    direction = torch.randn_like(vector(params)); direction /= direction.norm()
    theta = vector(params)
    h = 1e-5
    assign(params,theta+h*direction); plus = residual(model,low,params)
    assign(params,theta-h*direction); minus = residual(model,low,params)
    assign(params,theta)
    discrepancy = ((plus-minus)/(2*h)-J@direction).norm()/((plus-minus)/(2*h)).norm()
    assert discrepancy.item() < 1e-5
    a,b = model_from(state),model_from(state)
    assert torch.equal(vector(list(a.parameters())),vector(list(b.parameters())))
    ea,eb = evaluate(a,low,high,0,'A64'),evaluate(b,low,high,0,'B64')
    assert all(ea[k] == eb[k] for k in ('absolute_l2','absolute_h1','DFR24','DFR40','flux_rms','boundary_max'))
    return dict(parameters=8577, residuals=576, jacobian_shape=list(J.shape),
                directional_relative_error=discrepancy.item(), boundary_max=ea['boundary_max'],
                branch_initial_metrics_equal=True),ea,eb

def run_lbfgs(state,low,high,initial,result):
    model = model_from(state)
    opt = torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=5,history_size=50,
                            line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)
    rows=[dict(initial,optimization_seconds=0.,closure_calls=0)]
    calls=0; optimization_seconds=0.
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad(set_to_none=True)
        return backward(model,low)
    for step in range(1,801):
        sync()
        start=time.monotonic()
        opt.step(closure)
        finite_parameters(model)
        sync()
        optimization_seconds+=time.monotonic()-start
        check_optimizer_state(opt)
        if step%100==0:
            row=evaluate(model,low,high,step,'A64',step==800)
            row.update(optimization_seconds=optimization_seconds,closure_calls=calls)
            rows.append(row);csv_rows(result/'A64_trajectory.csv',rows)
            print(f'A64 outer {step}/800 DFR24={row["DFR24"]:.6e}',flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=800,
                    adam_source_sha256=sha(SOURCE),metrics=rows[-1]),result/'A64_final.pt')
    return rows[-1],optimization_seconds

def run_lm(state,low,high,initial,result):
    model = model_from(state)
    params=list(model.parameters())
    sync()
    start=time.monotonic()
    rho,J=residual(model,low,params,True)
    A=J@J.T
    assert_cuda_tensor(A,'JJ^T')
    sync()
    initial_audit_seconds=time.monotonic()-start
    mu=1e-3*max(A.diag().mean().item(),1.)
    initial_mu=mu
    accepted=total_rejected=streak=0
    optimization_seconds=0.
    rows=[]; proposals=[]
    row=dict(initial,iteration=0,mu=mu,mu_used=mu,total_rejected=0,rejected_since_previous_step=0,
             optimization_seconds=0.,initial_audit_seconds=initial_audit_seconds)
    rows.append(row);csv_rows(result/'B64_trajectory.csv',rows)
    stop='50 accepted iterations completed'
    while accepted<50:
        sync()
        start=time.monotonic()
        theta=vector(params);F=.5*rho.square().sum().item();rejected=0;success=False
        for attempt in range(10):
            used_mu=mu
            try:
                damped=A+mu*torch.eye(576,dtype=rho.dtype,device=DEVICE)
                assert_cuda_tensor(damped,'damped system')
                y=torch.linalg.solve(damped,rho)
                assert_cuda_tensor(y,'dual solution')
                delta=-J.T@y
                assert_cuda_tensor(delta,'delta')
                if not torch.isfinite(delta).all(): raise FloatingPointError('Nonfinite delta')
                assign(params,theta+delta)
                candidate=residual(model,low,params)
                Fnew=.5*candidate.square().sum().item()
                reason='finite candidate'
            except (FloatingPointError,torch.linalg.LinAlgError) as exc:
                Fnew=None;reason=str(exc)
            success=Fnew is not None and np.isfinite(Fnew) and Fnew<F
            proposals.append(dict(next_iteration=accepted+1,attempt=attempt+1,mu=mu,F=F,F_new=Fnew,accepted=success,reason=reason))
            if success:
                accepted+=1
                streak=streak+1 if (F-Fnew)/F<1e-6 else 0
                mu=max(1e-12,mu/10)
                break
            assign(params,theta)
            assert torch.equal(vector(params),theta)
            rejected+=1;total_rejected+=1
            mu=min(1e12,mu*10)
        sync()
        optimization_seconds+=time.monotonic()-start
        dump(result/'B64_proposals.json',proposals)
        if not success:
            stop='No acceptable step after 10 damping attempts; exact accepted parameters restored'
            break
        sync()
        start=time.monotonic()
        rho,J=residual(model,low,params,True)
        A=J@J.T
        assert_cuda_tensor(A,'JJ^T')
        sync()
        optimization_seconds+=time.monotonic()-start
        row=evaluate(model,low,high,accepted,'B64',accepted==50 or streak>=5)
        row.update(iteration=accepted,mu=mu,mu_used=used_mu,total_rejected=total_rejected,
                   rejected_since_previous_step=rejected,optimization_seconds=optimization_seconds,
                   initial_audit_seconds=initial_audit_seconds)
        rows.append(row);csv_rows(result/'B64_trajectory.csv',rows)
        torch.save(dict(model=model.state_dict(),accepted=accepted,mu=mu,total_rejected=total_rejected,
                        rng_state=torch.get_rng_state()),result/'B64_last_valid.pt')
        print(f'B64 accepted {accepted}/50 DFR24={row["DFR24"]:.6e} rejected={total_rejected}',flush=True)
        if streak>=5:
            stop='Relative F decrease < 1e-6 for five consecutive accepted iterations'
            break
    if rows[-1]['DFR40'] is None:
        final=evaluate(model,low,high,accepted,'B64',True)
        rows[-1]['DFR40']=final['DFR40']
        rows[-1]['diagnostic_seconds']+=final['diagnostic_seconds']
        csv_rows(result/'B64_trajectory.csv',rows)
    torch.save(dict(model=model.state_dict(),optimizer_state=dict(mu=mu,accepted=accepted,
                    rejected=total_rejected,stagnation_streak=streak),source_sha256=sha(SOURCE),
                    metrics=rows[-1]),result/'B64_final.pt')
    return rows[-1],dict(stop=stop,initial_mu=initial_mu,final_mu=mu,accepted=accepted,
                         rejected=total_rejected,optimization_seconds=optimization_seconds,
                         initial_audit_seconds=initial_audit_seconds)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    state,meta,low,high=setup(device=args.device,result=args.out)
    checks,initial_a,initial_b=preflight(state,low,high)
    result=args.out;result.mkdir(parents=True,exist_ok=False)
    dump(result/'config.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),
         source_metadata=str(BASE/'capacity_study/reports/B_metadata.json'),
         source_config=state['config'],width=64,parameter_count=8577,K_train=24,
         adam_reused=True,adam_steps=3000,adam_time_seconds='unavailable',
         lbfgs_outer_steps=800,lm_accepted_steps=50,lm_max_attempts=10,
         lm_damping_rule='1e-3*max(mean(diag(JJ^T)),1); /10 accepted, *10 rejected; clamp [1e-12,1e12]',
         lm_acceptance='finite strictly smaller F=0.5*||rho||^2',
         lm_early_stop='relative F decrease <1e-6 for five consecutive accepted steps',
         residual_weighting='rho=R/sqrt(1+lambda); DFR=sum(rho^2)',
         jacobian='actual K=24 training rho; batched reverse autodiff, 8 rows per block',
         lm_system='dual (J J^T + mu I)y=rho; delta=-J^T y',
         evaluation='absolute L2/H1 and relative L2 at n=64; DFR40 at n=64, n=96 final audit',
         device=str(DEVICE),torch_version=torch.__version__,cuda_version=torch.version.cuda,
         gpu_name=torch.cuda.get_device_name(DEVICE),preflight=checks))
    torch.save(state,result/'Adam3000_common.pt')
    dump(result/'checkpoint_hashes.json',dict(source=sha(SOURCE),common_copy=sha(result/'Adam3000_common.pt')))
    dump(result/'Adam3000_metrics.json',initial_a)
    a,atime=run_lbfgs(state,low,high,initial_a,result)
    b,bmeta=run_lm(state,low,high,initial_b,result)
    dump(result/'run_metadata.json',dict(A64_optimization_seconds=atime,B64=bmeta,
         Adam3000_source_sha256=sha(SOURCE),A64_final_sha256=sha(result/'A64_final.pt'),
         B64_final_sha256=sha(result/'B64_final.pt')))
    print('COMPLETE',result,flush=True)

if __name__=='__main__': main()
