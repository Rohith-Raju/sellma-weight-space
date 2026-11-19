from bench.editor import Workbench
import numpy as np, torch, json
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
P = {p.id:p for p in wb.practice_questions()}
meta=json.load(open("editor_runs/gemma_b25/notes/read2_meta.json"))
r=wb.load_pass(2)
X=r.residual.view(256,4,34,2560)
ids=[meta[4*i][0] for i in range(256)]
key=torch.tensor(["ABCD".index(P[i].question.answer) for i in ids])
ch=torch.tensor(["ABCD".index(P[i].baseline.chosen_letter) for i in ids])
iskey=torch.nn.functional.one_hot(key,4).bool()
g=torch.Generator().manual_seed(0); perm=torch.randperm(256,generator=g); folds=[perm[i::8] for i in range(8)]
mok=(ch==key)
print("model acc",mok.float().mean().item(),"by key",[round(mok[key==k].float().mean().item(),2) for k in range(4)])
def run(L,qc,lamf):
    pick=torch.zeros(256,dtype=torch.long); score=torch.zeros(256,4)
    for f in folds:
        tr=torch.ones(256,dtype=torch.bool); tr[f]=False
        x=X[:,:,L]
        xc=x-x[tr].mean(0,keepdim=True)
        if qc: xc=xc-xc.mean(1,keepdim=True)
        A=xc[tr].reshape(-1,2560); y=(iskey[tr].reshape(-1).float()-0.25)
        K=A@A.T; lam=K.diag().mean()*lamf
        w=A.T@torch.linalg.solve(K+lam*torch.eye(len(K)),y)
        score[f]=xc[f]@w
    pick=score.argmax(1); ok=pick==key
    return ok,pick,score
for L in [8,10,12,14,16,17,18,20,22,24]:
    for qc in [False,True]:
        for lamf in [0.1,1.0]:
            ok,pick,score=run(L,qc,lamf)
            print("L%02d qc=%d lam=%.1f CV acc %.3f by key %s picks %s | agree model %.3f | probe ok&model wrong %d, model ok&probe wrong %d, either %.3f"%(
              L,qc,lamf,ok.float().mean(),[round(ok[key==k].float().mean().item(),2) for k in range(4)],torch.bincount(pick,minlength=4).tolist(),
              (pick==ch).float().mean(),(ok&~mok).sum(),(mok&~ok).sum(),(ok|mok).float().mean()))
