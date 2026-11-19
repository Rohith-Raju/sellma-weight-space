from bench.editor import Workbench
import numpy as np, torch, json
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
P = {p.id:p for p in wb.practice_questions()}
meta=json.load(open("editor_runs/gemma_b25/notes/read2_meta.json"))
r=wb.load_pass(2)
M=r.mlp_acts.view(256,4,34,10240)
ids=[meta[4*i][0] for i in range(256)]
key=torch.tensor(["ABCD".index(P[i].question.answer) for i in ids])
iskey=torch.nn.functional.one_hot(key,4).bool()
g=torch.Generator().manual_seed(1); perm=torch.randperm(256,generator=g); h1=perm[:128]; h2=perm[128:]
def tstat(a,ik):
    a=a-a.mean(0,keepdim=True)        # position-centre
    a=a-a.mean(1,keepdim=True)        # question-centre
    x=a[ik]; y=a[~ik]
    return (x.mean(0)-y.mean(0))/torch.sqrt(x.var(0)/len(x)+y.var(0)/len(y)+1e-12)
out={}
for L in range(34):
    a=M[:,:,L]
    t1=tstat(a[h1],iskey[h1]); t2=tstat(a[h2],iskey[h2]); t=tstat(a,iskey)
    c=np.corrcoef(t1.numpy(),t2.numpy())[0,1]
    top=t.abs().topk(5)
    print("L%02d split-half corr of t %.3f | n|t|>4: %d  >5: %d >6: %d | top %s"%(L,c,(t.abs()>4).sum(),(t.abs()>5).sum(),(t.abs()>6).sum(),[(int(i),round(float(t[i]),1)) for i in top.indices]))
    out[L]=t
torch.save(out,"editor_runs/gemma_b25/notes/neuron_t_mek.pt")
