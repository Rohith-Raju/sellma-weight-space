from bench.editor import Workbench
import torch, numpy as np
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b10")
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
r=wb.load_pass(1)
w=wb.weights()
main_ids=[236776, 236799, 236780, 236796]
E=w.float32(wb.embedding_name)[main_ids]
nw=w.float32("model.language_model.norm.weight")
res=r.residual
key=torch.tensor(S["shown_key"]);ch=torch.tensor(S["chosen"]);sec=np.array(S["sec"])
for s in ["MEK","LAS"]:
    m=torch.tensor(sec==s); k=key[m]
    print("==",s)
    for L in [19,20,21,22,23,24,25,26,28,30,33]:
        x=res[m,L]; xn=x/x.pow(2).mean(-1,keepdim=True).sqrt()
        lg=(xn*(1+nw))@E.T
        am=lg.argmax(1)
        acc=[round((am[k==p]==p).float().mean().item(),2) for p in range(4)]
        dist=np.bincount(am.numpy(),minlength=4)
        # mean logit at pos p when key / when not key (centered)
        c=lg-lg.mean(1,keepdim=True)
        wk=[round(c[k==p,p].mean().item(),2) for p in range(4)]
        nk=[round(c[k!=p,p].mean().item(),2) for p in range(4)]
        print("L%02d acc-by-key %s argmax-dist %s  logit|key %s  logit|notkey %s"%(L,acc,dist,wk,nk))
