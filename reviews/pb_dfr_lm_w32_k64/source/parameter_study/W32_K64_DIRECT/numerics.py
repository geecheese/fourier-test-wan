import sys,json,os
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent; BASE=HERE.parents[1]
sys.path[:0]=[str(BASE/'width64_optimizer_study'),str(BASE/'full_solver/scripts'),str(BASE/'scripts')]
import run_width64_gpu as gpu
from pb_fourier_residual import moments,metrics
CONFIG=json.loads((HERE/'frozen_config.json').read_text())
def setup():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise RuntimeError('CUDA_VISIBLE_DEVICES must be 0')
 gpu.DEVICE=gpu.require_cuda('cuda:0'); torch.set_default_dtype(torch.float64); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
 return gpu.CudaResidualK(96,64),gpu.CudaResidualK(128,64)
def model_initial():
 raw=torch.load(Path(CONFIG['initial_weights_source']),map_location='cpu',weights_only=False); m=gpu.cap.Trial(CONFIG['architecture']);m.load_state_dict(raw['model']);m.to(gpu.DEVICE);return m
def model_from(c):
 m=gpu.cap.Trial(CONFIG['architecture']);m.load_state_dict(c['model']);m.to(gpu.DEVICE);return m
def vector(ps): return torch.cat([p.detach().reshape(-1) for p in ps])
def assign(ps,x):
 with torch.no_grad():
  o=0
  for p in ps:p.copy_(x[o:o+p.numel()].reshape_as(p));o+=p.numel()
def chunks(f,x): return np.concatenate([f(a) for a in np.array_split(x,max(1,(len(x)+8191)//8192))])
def evaluate(model,low,high):
 loss=low.evaluate(model)[0];value,grad=gpu.numpy_functions(model);x,w=high.rule[:2];e=value(x)-gpu.pb.u_r_exact(x);dg=grad(x)-gpu.pb.grad_exact(x);e2=float(np.sum(w*e**2));out={'absolute_l2':float(np.sqrt(e2)),'absolute_h1':float(np.sqrt(e2+np.sum(w*np.sum(dg**2,axis=1)))),'train_DFR64':float(loss)}
 for k in (24,32,40,64):out[f'DFR{k}']=float(metrics(moments(lambda a:chunks(value,a),lambda a:chunks(grad,a),k,high.rule))['L_DFR'])
 assert all(np.isfinite(v) for v in out.values());return out
def dfr64_at(model,rule):
 value,grad=gpu.numpy_functions(model)
 return float(metrics(moments(lambda a:chunks(value,a),lambda a:chunks(grad,a),64,rule))['L_DFR'])
def residual_jacobian(model,low):
 params=list(model.parameters());_,r,u,g=low.evaluate(model,create_graph=True);rho=(r*low.weights.sqrt()).flatten();assert rho.shape==(4096,);eye=torch.eye(4096,dtype=torch.float64,device=gpu.DEVICE);rows=[]
 for i in range(0,4096,8):
  gs=torch.autograd.grad(rho,params,grad_outputs=eye[i:i+8],is_grads_batched=True,retain_graph=True);rows.append(torch.cat([v.reshape(eye[i:i+8].shape[0],-1) for v in gs],1))
 J=torch.cat(rows).detach();assert J.shape==(4096,2241) and torch.isfinite(J).all();return rho.detach(),J
