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
qids=sorted({k[0] for k in prompts})
pass2_q={k[0] for k in prompts}
# final-position inputs (pass 1), generic-position inputs (pass 2)
Hq_all=A.norm_in(r1.residual[:,LAYER-1])            # [1024,2560]
Hgen=A.norm_in(r2.residual[:,LAYER-1])              # [1008,2560] non-final positions
def keys_nrope(x):  # normalised weighted keys without rope [n,4,256]
    k=(A.norm_in(x)@A.WK.T).view(-1,4,256); return k/k.pow(2).mean(-1,keepdim=True).sqrt()*(1+A.kn)
def build(train_q,heads=(3,7),kinds_used=kinds,scale=1.0,verbose=False):
    hq_idx=[i for i,(q,o) in enumerate(m1) if q not in pass2_q or q in train_q]
    hbar=Hq_all[hq_idx].mean(0); c=Hgen.mean(0)
    e=hbar-(hbar@c)/(c@c)*c
    if verbose:
        f=(Hq_all@e)/(hbar@e); g=(Hgen@e)/(hbar@e)
        print("gate factor on final positions: mean %.3f sd %.3f min %.3f | on other positions: mean %.3f sd %.3f max|.| %.3f"%(f.mean(),f.std(),f.min(),g.mean(),g.std(),g.abs().max()))
        print("cos(hbar,c) %.3f"%(torch.cosine_similarity(hbar,c,dim=0)))
    WQ=A.WQ.clone()
    for h in heads:
        kv=h//2
        M=[]
        for kd in kinds_used:
            K=torch.stack([torch.stack([keys_nrope(r2.residual[d[kd+L][0],LAYER-1][None])[0,kv] for (q,o),d in prompts.items() if q in train_q]).mean(0) for L in "ABCD"])
            M.append((K-K.mean(0,keepdim=True))*(1+A.qn))
        M=torch.cat(M)
        U,Sg,Vt=torch.linalg.svd(M,full_matrices=False); r=int((Sg>Sg[0]*1e-4).sum()); B=Vt[:r].T   # [256,r]
        rows=slice(h*256,(h+1)*256)
        qbar=WQ[rows]@hbar
        comp=B@(B.T@qbar)
        if verbose: print("head",h,"|qbar| %.2f |removed| %.2f rank %d"%(qbar.norm(),comp.norm(),r))
        WQ[rows]=WQ[rows]-scale*torch.outer(comp,e)/(hbar@e)
    return WQ
def evaluate(WQ,test_q,label,heads=(3,7)):
    out={k:[] for k in kinds}; ch=[];ky=[]
    for (qid,o),d in prompts.items():
        if qid not in test_q: continue
        j=idx1[(qid,o)]; n=d["eot"][2]
        q=A.q(r1.residual[j,LAYER-1][None],torch.tensor([n-1]),WQ)[0]
        for kd in kinds:
            src=torch.stack([r2.residual[d[kd+L][0],LAYER-1] for L in "ABCD"]); pos=torch.tensor([d[kd+L][1] for L in "ABCD"])
            out[kd].append(A.logits(q,A.k(src,pos)))
        ch.append(S["chosen"][j]); ky.append(S["shown_key"][j])
    ch=torch.tensor(ch);ky=torch.tensor(ky)
    print(label)
    for h in heads:
        for kd in kinds:
            sub=torch.stack(out[kd])[:,h]; sel=sub.gather(1,ch[:,None])[:,0]; rest=(sub.sum(1)-sel)/3
            selk=sub.gather(1,ky[:,None])[:,0]; restk=(sub.sum(1)-selk)/3
            print("   h%d %s mean-by-pos %s chosen-rest %.2f key-rest %.2f argmax=key %.2f dist %s"%(h,kd,np.round(sub.mean(0).numpy(),1),(sel-rest).mean(),(selk-restk).mean(),(sub.argmax(1)==ky).float().mean(),np.bincount(sub.argmax(1).numpy(),minlength=4)))
        X=torch.logsumexp(torch.stack([torch.stack(out[k])[:,h] for k in kinds],0),0)
        print("   h%d combined argmax=key %.2f argmax=chosen %.2f dist %s"%(h,(X.argmax(1)==ky).float().mean(),(X.argmax(1)==ch).float().mean(),np.bincount(X.argmax(1).numpy(),minlength=4)))
if __name__=="__main__":
    trA=set(qids[::2]); trB=set(qids[1::2])
    evaluate(A.WQ,trB,"== baseline half B"); evaluate(build(trA,verbose=True),trB,"== fit A eval B")
    evaluate(A.WQ,trA,"== baseline half A"); evaluate(build(trB),trA,"== fit B eval A")
    WQ=build(set(qids),verbose=True)
    name=wb.layer_param(LAYER,"self_attn.q_proj.weight")
    rows=list(range(3*256,4*256))+list(range(7*256,8*256))
    delta=(WQ-A.WQ)[rows]
    edit={name:{"rows":rows,"values":delta}}
    sz=wb.edit_size(edit); print({k:v for k,v in sz.items() if k!="tensors"},sz["tensors"][name])
    Wr=(wb.weights()[name].float()+(WQ-A.WQ)).to(torch.bfloat16).float()
    evaluate(Wr,set(qids),"== full edit bf16-rounded (in-sample)")
    torch.save({"rows":rows,"delta":delta},"/root/sellma/editor_runs/gemma_b10/notes/edit_q_v1.pt")
