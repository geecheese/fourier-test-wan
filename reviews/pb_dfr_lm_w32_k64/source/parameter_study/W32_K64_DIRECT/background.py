import os,sys,json,time,copy,random,signal,traceback,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import numerics as n
import torch
LOCK=None;STOP=False

def atomic(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(**kw):
 p=HERE/'status.json'; old=json.loads(p.read_text()) if p.exists() else {};old.update(kw);old['last_update']=time.time();atomic(p,old)
def sig(s,f):
 global STOP;STOP=True

def save(path,c,model,opt=None):
 c['model']={k:v.detach().cpu() for k,v in model.state_dict().items()};c['optimizer_state']=copy.deepcopy(opt.state_dict()) if opt is not None else c.get('optimizer_state',{});c['rng']={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()};tmp=Path(str(path)+'.tmp');torch.save(c,tmp);os.replace(tmp,path);atomic(str(path)+'.sha256.json',{'sha256':sha(path),'phase':c['phase'],'step':c['step']})
def load(path):return torch.load(path,map_location='cuda:0',weights_only=False)
def fresh(stage):return {'schema':2,'phase':stage,'step':0,'model':{},'optimizer_state':{},'metrics':{},'history':{'steps':[],'windows':[],'proposals':[]},'counters':{'best':float('inf'),'window_start':float('inf'),'streak':0,'accepted':0,'rejected':0,'attempts':0},'wall_seconds':0.0,'optimization_seconds':0.0,'stop_reason':None}
def elapsed(t0,base):return base+time.monotonic()-t0
def phase_adam(low,high):
 folder=HERE/'Adam3000';folder.mkdir(exist_ok=True);latest=folder/'latest_checkpoint.pt';final=folder/'final_checkpoint.pt'
 if final.exists():return load(final)
 if latest.exists():c=load(latest);m=n.model_from(c);opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(c['optimizer_state']);step=c['step'];base=c.get('wall_seconds',0.);print('RESUME Adam',step,flush=True)
 else:
  m=n.model_initial();opt=torch.optim.Adam(m.parameters(),lr=.001);c=fresh('Adam3000');step=0;base=0.;c['metrics']=n.evaluate(m,low,high);c['counters']['best']=c['metrics']['train_DFR64'];c['counters']['window_start']=c['metrics']['train_DFR64'];save(folder/'start_checkpoint.pt',c,m,opt);save(latest,c,m,opt)
 t0=time.monotonic();status(stage='Adam3000',state='running',adam_step=step,checkpoint=str(latest),training_complete=False)
 while step<3000:
  if STOP: c['wall_seconds']=elapsed(t0,base);save(latest,c,m,opt);raise InterruptedError('signal')
  opt.zero_grad(set_to_none=True);loss=low.evaluate(m,create_graph=True)[0];loss.backward();opt.step();step+=1;c['step']=step;c['metrics']=n.evaluate(m,low,high) if step%100==0 else c['metrics'];c['history']['steps'].append(float(loss.detach()));c['counters']['best']=min(c['counters']['best'],float(loss.detach()))
  if step%100==0:
   print('Adam3000',step,'DFR64',float(loss),flush=True);c['wall_seconds']=elapsed(t0,base);save(latest,c,m,opt);status(stage='Adam3000',adam_step=step,current_DFR64=float(loss),checkpoint=str(latest),cumulative_elapsed_seconds=c['wall_seconds'])
 c['stop_reason']='adam3000_complete';c['wall_seconds']=elapsed(t0,base);save(final,c,m,opt);save(latest,c,m,opt);status(stage='adam_integration',adam_step=3000,checkpoint=str(final),training_complete=False);return c

def integration_gate(c,low,high):
 m=n.model_from(c);a=n.dfr64_at(m,low.rule);b=n.dfr64_at(m,high.rule)
 d=abs(a-b);r=d/max(abs(b),1e-30);out={'n96_DFR64':a,'n128_DFR64':b,'absolute_difference':d,'relative_difference':r,'threshold':1e-8,'passed':r<1e-8,'reload_finite':True};atomic(HERE/'Adam3000'/'integration_gate.json',out);return out

def phase_lm(c,low,high):
 folder=HERE/'G-LM';folder.mkdir(exist_ok=True);latest=folder/'latest_checkpoint.pt';final=folder/'final_checkpoint.pt'
 if final.exists():return load(final)
 if latest.exists():c=load(latest);m=n.model_from(c);step=c['step'];base=c.get('wall_seconds',0.);ct=c['counters'];print('RESUME LM',step,flush=True)
 else:m=n.model_from(c);c=fresh('G-LM');step=0;base=0.;ct=c['counters'];rho,J=n.residual_jacobian(m,low);A=J@J.T;ct['mu']=1e-3*max(float(torch.diag(A).mean()),1.);c['metrics']=n.evaluate(m,low,high);ct['best']=c['metrics']['train_DFR64'];ct['window_start']=ct['best'];save(folder/'start_checkpoint.pt',c,m,None);save(latest,c,m,None)
 t0=time.monotonic();status(stage='G-LM',accepted_lm_steps=step,checkpoint=str(latest),current_mu=ct['mu'])
 if latest.exists() and step>0:rho,J=n.residual_jacobian(m,low);A=J@J.T
 while step<2000 and not STOP:
  if elapsed(t0,base)>=14400:c['stop_reason']='capped_not_converged';c['cap_type']='time_cap';break
  theta=n.vector(list(m.parameters()));F=.5*float(rho.square().sum());ok=False
  for _ in range(10):
   ct['attempts']+=1;mu=ct['mu'];new=None
   try:
    y=torch.linalg.solve(A+mu*torch.eye(4096,device=n.gpu.DEVICE),rho);delta=-J.T@y;n.assign(list(m.parameters()),theta+delta);new=.5*float(low.evaluate(m)[0]);finite=np.isfinite(new)
   except Exception as e:finite=False;new=None
   ok=bool(finite and new<F)
   if ok:ct['accepted']+=1;ct['mu']=max(1e-12,mu/2);break
   n.assign(list(m.parameters()),theta);ct['rejected']+=1;ct['mu']=min(1e12,mu*2)
  if not ok:c['stop_reason']='capped_not_converged' if elapsed(t0,base)>=14400 else 'no_acceptable_step_after_10_attempts';c['cap_type']='time_cap' if elapsed(t0,base)>=14400 else None;break
  step+=1;c['step']=step;c['history']['steps'].append(new*2);ct['best']=min(ct['best'],new*2)
  if step%100==0:
   imp=(ct['window_start']-ct['best'])/max(ct['window_start'],1e-30);ct['streak']=ct['streak']+1 if imp<1e-4 else 0;c['history']['windows'].append({'end_step':step,'relative_improvement':imp,'streak':ct['streak']});ct['window_start']=ct['best'];c['metrics']=n.evaluate(m,low,high);print('LM',step,'DFR64',new*2,'rejected',ct['rejected'],flush=True)
   if ct['streak']>=3:c['stop_reason']='objective_stagnation';c['cap_type']=None;break
  if step%50==0:c['wall_seconds']=elapsed(t0,base);save(latest,c,m,None);status(stage='G-LM',accepted_lm_steps=step,rejected_count=ct['rejected'],current_mu=ct['mu'],current_DFR64=new*2,checkpoint=str(latest),remaining_accepted_steps=max(0,2000-step),remaining_wall_seconds=max(0,14400-c['wall_seconds']))
  n.gpu.sync();rho,J=n.residual_jacobian(m,low);A=J@J.T
 if step>=2000 and not c['stop_reason']:c['stop_reason']='capped_not_converged';c['cap_type']='step_cap'
 c['wall_seconds']=elapsed(t0,base);c['metrics']=n.evaluate(m,low,high);save(final,c,m,None);save(latest,c,m,None);status(stage='verification',accepted_lm_steps=step,stop_reason=c['stop_reason'],cap_type=c.get('cap_type'),checkpoint=str(final),remaining_accepted_steps=max(0,2000-step),remaining_wall_seconds=max(0,14400-c['wall_seconds']));return c

def main():
 global LOCK
 LOCK=(HERE/'run.lock').open('a');import fcntl;fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
 if (HERE/'status.json').exists() and json.loads((HERE/'status.json').read_text()).get('state')=='complete':print('already complete');return
 if not (HERE/'cuda_diagnostic.json').exists() or not (HERE/'preflight_diagnostic.json').exists():raise RuntimeError('approved diagnostics missing')
 if not json.loads((HERE/'preflight_diagnostic.json').read_text()).get('passed'):raise RuntimeError('preflight not passed')
 status(state='running',stage='runtime_import',pid=os.getpid(),training_complete=False,postprocessing_complete=False,verification_status='not_run')
 low,high=n.setup();assert sum(p.numel() for p in n.model_initial().parameters())==2241
 adam=phase_adam(low,high);gate=integration_gate(adam,low,high);status(stage='adam_integration',integration_gate=gate)
 if not gate['passed']:raise RuntimeError('Adam DFR64 integration gate failed')
 lm=phase_lm(adam,low,high);atomic(HERE/'verification.json',{'LM':lm['metrics'],'stop_reason':lm['stop_reason'],'cap_type':lm.get('cap_type'),'accepted':lm['step'],'rejected':lm['counters']['rejected']});status(state='complete',stage='complete',training_complete=True,postprocessing_complete=False,verification_status='pending_postprocess',exit_code=0)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic(HERE/'error.json',{'failure_stage':json.loads((HERE/'status.json').read_text()).get('stage') if (HERE/'status.json').exists() else 'startup','exception':repr(e),'traceback':traceback.format_exc(),'time':time.time()});status(state='failed',failure_stage='runtime',exception=repr(e),exit_code=1);traceback.print_exc();raise
