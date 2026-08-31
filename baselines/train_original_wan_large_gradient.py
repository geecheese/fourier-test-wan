#!/usr/bin/env python3
"""Small classical parameterized WAN baseline for the 1D transition layer.

Objective classification: learned-neural-test weak moment.
"""
from __future__ import annotations
import argparse, math, time, json
import torch
from experiments.poisson2d.train_spectral_wan import TrialNet, exact, grid
PI=math.pi; EPS=1e-30
def grad(m,x):
 x=x.detach().clone().requires_grad_(True); y=m(x); return y,torch.autograd.grad(y.sum(),x,create_graph=True)[0]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--rounds',type=int,default=100); p.add_argument('--inner-u',type=int,default=5); p.add_argument('--inner-v',type=int,default=5); p.add_argument('--grid-size',type=int,default=64); p.add_argument('--audit-size',type=int,default=257); p.add_argument('--device',default='cuda',choices=('cpu','cuda')); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
 if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
 torch.set_default_dtype(torch.float64); torch.manual_seed(a.seed); u=TrialNet(1).to(a.device); torch.manual_seed(a.seed+1); v=TrialNet(1).to(a.device)
 x=grid(a.grid_size,1,a.device); xx=x.detach().clone().requires_grad_(True); f=-torch.autograd.grad(torch.autograd.grad(exact(xx,'large_gradient1d').sum(),xx,create_graph=True)[0].sum(),xx)[0].detach(); w=PI/(a.grid_size-1)
 def score():
  U,DU=grad(u,x); V,DV=grad(v,x); m=w*(DU*DV-f*V).sum(); n=w*(V.square()+DV.square()).sum(); return m.square()/(n+EPS)
 t=time.perf_counter(); calls=0
 for _ in range(a.rounds):
  for net,maximize,inner in ((v,True,a.inner_v),(u,False,a.inner_u)):
   for q in net.parameters(): q.requires_grad_(True)
   other=v if net is u else u
   for q in other.parameters(): q.requires_grad_(False)
   opt=torch.optim.LBFGS(net.parameters(),lr=1.,max_iter=inner,line_search_fn='strong_wolfe')
   def closure():
    nonlocal calls; calls+=1; opt.zero_grad(set_to_none=True); z=score(); loss=-torch.log(z+EPS) if maximize else z; loss.backward(); return loss
   opt.step(closure)
 valx=grid(a.audit_size,1,a.device).requires_grad_(True); U=u(valx); E=exact(valx,'large_gradient1d'); dU=torch.autograd.grad(U.sum(),valx)[0]; dE=torch.autograd.grad(E.sum(),valx)[0]; wt=PI/(a.audit_size-1); l2=wt*(U-E).square().sum(); h1=l2+wt*(dU-dE).square().sum(); result={'relative_l2':float(torch.sqrt(l2/(wt*E.square().sum()))),'h1':float(torch.sqrt(h1)),'training_seconds':time.perf_counter()-t,'calls':calls}; print(json.dumps(result))
if __name__=='__main__': main()
