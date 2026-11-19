from bench.editor import Workbench
import torch, numpy as np, itertools, pickle
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b10")
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
meta=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
P={p.id:p for p in wb.practice_questions()}
r=wb.load_pass(1)
w=wb.weights()
main_ids=[236776, 236799, 236780, 236796]
E=w.float32(wb.embedding_name)[main_ids]
nw=w.float32("model.language_model.norm.weight")
res=r.residual
key=torch.tensor(S["shown_key"]);ch=torch.tensor(S["chosen"]);sec=np.array(S["sec"]);order=np.array(S["order"])
# split by exam date (question id prefix) into two halves
dates=sorted({q.split("/")[0] for q,o in meta}); half=set(dates[::2])
h=torch.tensor([q.split("/")[0] in half for q,o in meta])
print(len(dates),"dates")
grid=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]
for L in [23,24]:
    x=res[:,L]; lg=((x/x.pow(2).mean(-1,keepdim=True).sqrt())*(1+nw))@E.T
    for s in ["MEK","LAS","both"]:
        m=torch.tensor((sec==s) if s!="both" else np.ones(len(sec),bool))
        def acc(b,mm): return ((lg[mm]+torch.tensor(b)).argmax(1)==key[mm]).float().mean().item()
        for tr,te,nm in [(m&h,m&~h,"fit even dates"),(m&~h,m&h,"fit odd dates")]:
            best=max(itertools.product([0],grid,grid,grid),key=lambda b:acc(list(b),tr))
            print("L",L,s,nm,"best",best,"train %.3f->%.3f test %.3f->%.3f"%(acc([0,0,0,0],tr),acc(list(best),tr),acc([0,0,0,0],te),acc(list(best),te)))
# where do key-A failures go, final answers
for s in ["MEK","LAS"]:
    m=(sec==s)
    for k in range(4):
        sel=m&(key.numpy()==k)&(ch.numpy()!=k)
        print(s,"key","ABCD"[k],"wrong->",np.bincount(ch.numpy()[sel],minlength=4))
