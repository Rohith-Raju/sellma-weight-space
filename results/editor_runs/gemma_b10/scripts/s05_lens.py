from bench.editor import Workbench
import torch, pickle, numpy as np
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
print("resid norms by layer",[round(res[:,L].norm(dim=-1).mean().item(),1) for L in range(0,34,3)])
for s in ["MEK","LAS"]:
    m=torch.tensor(sec==s)
    print("==",s)
    for L in range(34):
        x=res[m,L]; xn=x/x.pow(2).mean(-1,keepdim=True).sqrt()
        lg=(xn*(1+nw))@E.T
        c=lg-lg.mean(1,keepdim=True)
        k=key[m]; 
        ev=(lg.gather(1,k[:,None])[:,0]-(lg.sum(1)-lg.gather(1,k[:,None])[:,0])/3).mean()
        top2=lg.topk(2,dim=1).values; mg=(top2[:,0]-top2[:,1]).mean()
        print("L%02d bias[A,B,C,D]=%s  key-vs-rest %.2f  margin %.2f  lens=final %.3f lens=key %.3f"%(L,np.round(c.mean(0).numpy(),2),ev,mg,(lg.argmax(1)==ch[m]).float().mean(),(lg.argmax(1)==k).float().mean()))
