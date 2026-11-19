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
val=torch.tensor([P[i].paper=="validation" for i in ids]); tr=~val
print("model acc tr %.3f val %.3f"%((key==ch)[tr].float().mean(),(key==ch)[val].float().mean()))
iskey=torch.nn.functional.one_hot(key,4).bool()
for L in range(0,34,1):
    x=X[:,:,L]                      # 256,4,2560
    xc=x-x[tr].mean(0,keepdim=True) # remove per-position mean
    d=xc[tr][iskey[tr]].mean(0)-xc[tr][~iskey[tr]].mean(0)
    proj=xc@d/d.norm()
    am=proj.argmax(1)
    # ridge probe in sample space (dual)
    A=xc[tr].reshape(-1,2560); y=(iskey[tr].reshape(-1).float()-0.25)
    K=A@A.T; lam=K.diag().mean()*0.3
    alpha=torch.linalg.solve(K+lam*torch.eye(len(K)),y); wr=A.T@alpha
    pr=xc@wr; am2=pr.argmax(1)
    print("L%02d meandiff: pick-key tr %.3f val %.3f | ridge val %.3f | val agree-with-model-choice %.3f | meandiff picks dist %s | proj key-mean by pos %s nonkey %s"%(
      L,(am==key)[tr].float().mean(),(am==key)[val].float().mean(),(am2==key)[val].float().mean(),(am2==ch)[val].float().mean(),
      torch.bincount(am,minlength=4).tolist(),
      np.round([proj[:,k][key==k].mean().item() for k in range(4)],0),np.round([proj[:,k][key!=k].mean().item() for k in range(4)],0)))
