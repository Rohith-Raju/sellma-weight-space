from bench.editor import Workbench
import numpy as np, torch, json
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
rows=json.load(open("editor_runs/gemma_b25/notes/read1_rows.json"))
r=wb.load_pass(1)
E=wb.weights().float32(wb.embedding_name)
nw=wb.weights().float32("model.language_model.norm.weight")
tid=[wb.token_ids(l)[0] for l in "ABCD"]; U=E[tid]
def lens(x):
    x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw)
    return x@U.T
key=torch.tensor(["ABCD".index(x["key"]) for x in rows])
sec=np.array([x["sec"] for x in rows])
for L in [23,24,25,28,33]:
    lg=lens(r.residual[:,L])
    for s in ["MEK","LAS"]:
        m=torch.tensor(sec==s)
        out=[]
        for b in [0,0.5,1,1.5,2,3,4,6]:
            l2=lg[m].clone(); l2[:,0]+=b
            am=l2.argmax(1)
            out.append((b,round((am==key[m]).float().mean().item(),3),(am==0).sum().item()))
        print(L,s,out)
# residual norms & class-mean geometry at L22,23
for L in [20,21,22,23,24]:
    x=r.residual[:,L]; ch=torch.tensor(["ABCD".index(q["chosen"]) for q in rows])
    mu=torch.stack([x[ch==k].mean(0) for k in range(4)]); g=x.mean(0)
    d=mu-mu.mean(0)
    print(L,"resid norm %.1f"%x.norm(dim=1).mean(),"class-mean offsets norms",np.round(d.norm(dim=1).numpy(),2),"within std %.2f"%(x-mu[ch]).norm(dim=1).mean())
    dn=d/d.norm(dim=1,keepdim=True); Un=(U-U.mean(0)); Un=Un/Un.norm(dim=1,keepdim=True)
    print("   cos(class offset, unembed offset):",np.round((dn*Un).sum(1).numpy(),3), " cos between class offsets A-B.. ",np.round((dn@dn.T).numpy(),2).tolist())
