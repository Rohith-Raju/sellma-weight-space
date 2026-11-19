import sys; sys.path.insert(0,"/root/sellma/editor_runs/gemma_b10/scripts")
from attnlib import Attn
from bench.editor import Workbench
import torch, numpy as np, pickle
from collections import defaultdict
wb = Workbench.open("gemma_b10")
meta=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass2_meta.pkl","rb"))
m1=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
idx1={k:i for i,k in enumerate(m1)}
r1=wb.load_pass(1); r2=wb.load_pass(2)
LAYER=23
prompts=defaultdict(dict)
for i,(qid,o,name,pos,n) in enumerate(meta): prompts[(qid,o)][name]=(i,pos,n)
kinds=["lab","nl","ins"]
A=Attn(wb,LAYER)
def run(WK=None,norope=False,verbose=True):
    out={k:[] for k in kinds}; ch=[];ky=[]
    for (qid,o),d in prompts.items():
        j=idx1[(qid,o)]; n=d["eot"][2]
        xq=r1.residual[j,LAYER-1][None]; q=A.q(xq,torch.tensor([0 if norope else n-1]))[0]
        for kd in kinds:
            src=torch.stack([r2.residual[d[kd+L][0],LAYER-1] for L in "ABCD"]); pos=torch.tensor([0 if norope else d[kd+L][1] for L in "ABCD"])
            out[kd].append(A.logits(q,A.k(src,pos,WK)))  # [8,4]
        ch.append(S["chosen"][j]); ky.append(S["shown_key"][j])
    ch=torch.tensor(ch);ky=torch.tensor(ky)
    res={}
    for kd in kinds:
        X=torch.stack(out[kd])  # [48,8,4]
        res[kd]=X
        if verbose:
            for h in [3,7,2,0,6]:
                sub=X[:,h]; sel=sub.gather(1,ch[:,None])[:,0]; rest=(sub.sum(1)-sel)/3
                selk=sub.gather(1,ky[:,None])[:,0]; restk=(sub.sum(1)-selk)/3
                print("  %s h%d mean-by-pos %s  chosen-rest %.2f key-rest %.2f  argmax=chosen %.2f argmax=key %.2f argmax dist %s"%(kd,h,np.round(sub.mean(0).numpy(),1),(sel-rest).mean(),(selk-restk).mean(),(sub.argmax(1)==ch).float().mean(),(sub.argmax(1)==ky).float().mean(),np.bincount(sub.argmax(1).numpy(),minlength=4)))
    return res,ch,ky
print("== baseline (rope)"); base,ch,ky=run()
print("== baseline no rope"); run(norope=True)
# combined head3 evidence: logsumexp over kinds per option
def comb(res,h=3,bias=None):
    X=torch.stack([res[k][:,h] for k in kinds],0)  # [3,48,4]
    return torch.logsumexp(X,0)
c=comb(base); print("h3 combined argmax=chosen %.2f =key %.2f dist %s"%((c.argmax(1)==ch).float().mean(),(c.argmax(1)==ky).float().mean(),np.bincount(c.argmax(1).numpy(),minlength=4)))
# idealised: remove per-kind per-position mean
X=torch.stack([base[k][:,3]-base[k][:,3].mean(0,keepdim=True)+base[k][:,3].mean() for k in kinds],0); c2=torch.logsumexp(X,0)
print("h3 combined, prior removed (ideal): argmax=chosen %.2f =key %.2f dist %s"%((c2.argmax(1)==ch).float().mean(),(c2.argmax(1)==ky).float().mean(),np.bincount(c2.argmax(1).numpy(),minlength=4)))
print("chosen dist",np.bincount(ch.numpy(),minlength=4),"key dist",np.bincount(ky.numpy(),minlength=4),"acc",(ch==ky).float().mean().item())
