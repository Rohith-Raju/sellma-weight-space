from bench.editor import Workbench
import numpy as np, torch, json
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
print({k:v for k,v in wb.config["text_config"].items()})
P = {p.id:p for p in wb.practice_questions()}
meta=json.load(open("editor_runs/gemma_b25/notes/read2_meta.json"))
m1=json.load(open("editor_runs/gemma_b25/notes/read1_meta.json"))
r1=wb.load_pass(1); r=wb.load_pass(2)
X=r.residual.view(256,4,34,2560)
ids=[meta[4*i][0] for i in range(256)]
idx1={(a,b):i for i,(a,b) in enumerate(m1)}
last=torch.stack([r1.residual[idx1[(i,"ABCD")]] for i in ids])   # 256,34,2560
key=torch.tensor(["ABCD".index(P[i].question.answer) for i in ids])
iskey=torch.nn.functional.one_hot(key,4).bool()
E=wb.weights().float32(wb.embedding_name); nw=wb.weights().float32("model.language_model.norm.weight")
tid=[wb.token_ids(l)[0] for l in "ABCD"]; U=E[tid]
def lens(x):
    x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw); return x@U.T
# true final log probs
lp=torch.tensor([[np.log(max(P[i].baseline.letter_probs[l],1e-30)) for l in "ABCD"] for i in ids]).float()
g=torch.Generator().manual_seed(0); perm=torch.randperm(256,generator=g); folds=[perm[i::8] for i in range(8)]
def cvscore(L,lamf=1.0):
    score=torch.zeros(256,4)
    for f in folds:
        tr=torch.ones(256,dtype=torch.bool); tr[f]=False
        x=X[:,:,L]; xc=x-x[tr].mean(0,keepdim=True)
        A=xc[tr].reshape(-1,2560); y=(iskey[tr].reshape(-1).float()-0.25)
        K=A@A.T; w=A.T@torch.linalg.solve(K+K.diag().mean()*lamf*torch.eye(len(K)),y)
        score[f]=xc[f]@w
    return score
for L in [16,20]:
    s=cvscore(L); sc=s-s.mean(1,keepdim=True)
    for name,tgt in [("L23lens",lens(last[:,23])),("L24lens",lens(last[:,24])),("final logp",lp)]:
        t=tgt-tgt.mean(1,keepdim=True)
        out=[]
        for k in range(4):
            a=sc[:,k]; b=t[:,k]
            slope=((a-a.mean())*(b-b.mean())).sum()/((a-a.mean())**2).sum()
            out.append("%s: r=%.2f slope=%.1f mean=%.2f sd=%.2f"%("ABCD"[k],np.corrcoef(a,b)[0,1],slope,b.mean(),b.std()))
        print("probe L%d vs %s | "%(L,name)+" | ".join(out))
    print("  probe score sd by pos",np.round(sc.std(0).numpy(),3))
