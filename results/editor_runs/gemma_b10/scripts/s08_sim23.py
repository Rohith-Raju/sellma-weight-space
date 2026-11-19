from bench.editor import Workbench
import torch, numpy as np, itertools
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b10")
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
r=wb.load_pass(1)
w=wb.weights()
main_ids=[236776, 236799, 236780, 236796]
E=w.float32(wb.embedding_name)[main_ids]
nw=w.float32("model.language_model.norm.weight")
res=r.residual
key=torch.tensor(S["shown_key"]);ch=torch.tensor(S["chosen"]);sec=np.array(S["sec"]);order=np.array(S["order"])
for L in [23,24,25]:
    x=res[:,L]; lg=((x/x.pow(2).mean(-1,keepdim=True).sqrt())*(1+nw))@E.T
    for s in ["MEK","LAS"]:
        m=torch.tensor(sec==s)
        print("L",L,s)
        for b in [0,0.5,1,1.5,2,3]:
            l2=lg[m].clone(); l2[:,0]+=b; am=l2.argmax(1); k=key[m]
            print("   A+%.1f acc %.3f  A-key acc %.2f  nonA-key acc %.3f  flips: toA right %d wrong %d, lost %d"%(b,(am==k).float().mean(),(am[k==0]==0).float().mean(),(am[k!=0]==k[k!=0]).float().mean(),
               ((am==0)&(lg[m].argmax(1)!=0)&(k==0)).sum(),((am==0)&(lg[m].argmax(1)!=0)&(k!=0)).sum(),((am==0)&(lg[m].argmax(1)==k)&(k!=0)).sum()))
        # margin structure when key=A and wrong: gap between top and A
        l=lg[m]; k=key[m]; am=l.argmax(1); sel=(k==0)&(am!=0)
        gap=(l[sel].max(1).values-l[sel,0])
        print("   keyA wrong: gap quantiles",np.round(np.quantile(gap.numpy(),[.1,.25,.5,.75,.9]),2), "n",sel.sum().item())
        for p in [1,2,3]:
            sel=(k==p)&(am!=p); gap=(l[sel].max(1).values-l[sel,p])
            print("   key%s wrong: gap quantiles"%"ABCD"[p],np.round(np.quantile(gap.numpy(),[.1,.25,.5,.75,.9]),2),"n",sel.sum().item())
