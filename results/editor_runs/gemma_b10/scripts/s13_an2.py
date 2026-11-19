from bench.editor import Workbench
import torch, numpy as np, pickle
from collections import defaultdict
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b10")
P={p.id:p for p in wb.practice_questions()}
meta=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass2_meta.pkl","rb"))
m1=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
idx1={k:i for i,k in enumerate(m1)}
r=wb.load_pass(2)
w=wb.weights()
main_ids=[236776, 236799, 236780, 236796]
E=w.float32(wb.embedding_name)[main_ids]
nw=w.float32("model.language_model.norm.weight")
res=r.residual
by=defaultdict(list)
for i,(qid,o,name,pos,n) in enumerate(meta): by[name].append(i)
def info(i):
    qid,o,name,pos,n=meta[i]; j=idx1[(qid,o)]
    return S["shown_key"][j],S["chosen"][j]
print("positions: lens key-vs-rest / lens-argmax==final chosen, by layer")
Ls=[14,16,18,19,20,21,22,23,24,26]
for name in ["instr_end","eot","nl1","sot","model"]:
    ii=by[name]; k=torch.tensor([info(i)[0] for i in ii]); c=torch.tensor([info(i)[1] for i in ii])
    out=[]
    for L in Ls:
        x=res[ii,L]; lg=((x/x.pow(2).mean(-1,keepdim=True).sqrt())*(1+nw))@E.T; cc=lg-lg.mean(1,keepdim=True)
        out.append("L%d %.2f/%.2f"%(L,cc.gather(1,c[:,None]).mean()*4/3,(lg.argmax(1)==c).float().mean()))
    print(name.ljust(10)," ".join(out))
# option-specific positions: does lens at that position say "its own letter" more when it is the chosen/key option?
print("option positions: centered lens logit of OWN letter when option is chosen vs not chosen")
for kind in ["lab","end","nl","ins"]:
    for L in [8,12,16,18,20,22,24]:
        a=[];b=[]
        for p_,Lt in enumerate("ABCD"):
            ii=by[kind+Lt]
            x=res[ii,L]; lg=((x/x.pow(2).mean(-1,keepdim=True).sqrt())*(1+nw))@E.T; cc=lg-lg.mean(1,keepdim=True)
            for t,i in enumerate(ii):
                (a if info(i)[1]==p_ else b).append(cc[t,p_].item())
        print(kind,"L%d chosen %.2f notchosen %.2f"%(L,np.mean(a),np.mean(b)),end=" | ")
    print()
# linear separability of chosen vs not at option positions: diff-of-means direction, split-half test (by question)
print("diff-of-means separability (AUC, train on half the questions), chosen-option vs others")
qids=sorted({m[0] for m in meta}); tr=set(qids[::2])
def auc(a,b):
    a=np.array(a);b=np.array(b); return (a[:,None]>b[None,:]).mean()
for kind in ["lab","end","nl","ins"]:
    line=[]
    for L in [4,8,12,14,16,18,20,22]:
        X=[];y=[];t=[];pp=[]
        for p_,Lt in enumerate("ABCD"):
            for i in by[kind+Lt]:
                X.append(res[i,L]); y.append(info(i)[1]==p_); t.append(meta[i][0] in tr); pp.append(p_)
        X=torch.stack(X); y=torch.tensor(y); t=torch.tensor(t); pp=torch.tensor(pp)
        # remove per-position mean (position identity) using train set
        for p_ in range(4):
            X[pp==p_]-=X[(pp==p_)&t].mean(0)
        d=X[t&y].mean(0)-X[t&~y].mean(0)
        s=X@d
        line.append("L%d %.2f"%(L,auc(s[~t&y].numpy(),s[~t&~y].numpy())))
    print(kind," ".join(line))
