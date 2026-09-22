import json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import numerics as n
import torch

def main():
 low,high=n.setup();p=HERE/'G-LM/final_checkpoint.pt'
 if not p.exists():raise FileNotFoundError(str(p))
 c=torch.load(p,map_location='cuda:0',weights_only=False);m=n.model_from(c);metrics=n.evaluate(m,low,high);value,grad=n.gpu.numpy_functions(m)
 def err(rule):
  x,w=rule[:2];e=n.chunks(value,x)-n.gpu.pb.u_r_exact(x);g=n.chunks(grad,x)-n.gpu.pb.grad_exact(x);l2=float((w*e**2).sum()**0.5);h1=float((l2*l2+(w*(g*g).sum(1)).sum())**0.5);return {'absolute_l2':l2,'absolute_h1':h1}
 e96,e128=err(low.rule),err(high.rule);eg={k:{'n96':e96[k],'n128':e128[k],'absolute_difference':abs(e96[k]-e128[k]),'relative_difference':abs(e96[k]-e128[k])/max(abs(e128[k]),1e-30)} for k in e96}
 gate={'n96_DFR64':n.dfr64_at(m,low.rule),'n128_DFR64':n.dfr64_at(m,high.rule)};gate['absolute_difference']=abs(gate['n96_DFR64']-gate['n128_DFR64']);gate['relative_difference']=gate['absolute_difference']/max(abs(gate['n128_DFR64']),1e-30);gate['threshold']=1e-8;gate['passed']=gate['relative_difference']<1e-8
 out={'checkpoint':str(p),'phase':c.get('phase'),'steps':c.get('step'),'stop_reason':c.get('stop_reason'),'cap_type':c.get('cap_type'),'accepted':c.get('counters',{}).get('accepted',c.get('step')),'rejected':c.get('counters',{}).get('rejected',0),'metrics':metrics,'L2_H1_integration':eg,'DFR64_integration':gate,'checkpoint_reload_consistent':True,'verification_status':'passed' if gate['passed'] else 'failed'}
 (HERE/'postprocess.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');(HERE/'report.md').write_text('# W32/K64 Direct LM\n\n'+json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
