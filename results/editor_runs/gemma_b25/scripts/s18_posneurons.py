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
g=torch.Generator().manual_seed(1); perm=torch.randperm(256,generator=g); h1=perm[:128]; h2=perm[128:]
def eff(a,k,pos):   # a: n,4,N ; effect of being key at position pos, in pooled-sd units -> t
    x=a[k==pos][:,pos]; y=a[k!=pos][:,pos]
    return (x.mean(0)-y.mean(0))/torch.sqrt(x.var(0)/len(x)+y.var(0)/len(y)+1e-12), x.mean(0)-y.mean(0)
out={}
for L in range(10,24):
    a=M[:,:,L]
    res={}
    for nm,idx in [("all",torch.arange(256)),("h1",h1),("h2",h2)]:
        ts=[eff(a[idx],key[idx],p) for p in range(4)]
        res[nm]=(torch.stack([t[0] for t in ts]),torch.stack([t[1] for t in ts]))
    tA,tB,tC,tD=res["all"][0]; d=res["all"][1]
    rest=(d[1]+d[2]+d[3])/3
    c=lambda u,v:np.corrcoef(u.numpy(),v.numpy())[0,1]
    # reliability of the comparative index (rest effect minus A effect) across halves
    comp1=(res["h1"][1][1:].mean(0)-res["h1"][1][0]); comp2=(res["h2"][1][1:].mean(0)-res["h2"][1][0])
    print("L%02d corr(tA,tB) %.2f corr(tB,tC) %.2f corr(tC,tD) %.2f corr(tA,tD) %.2f | split-half: dA %.2f dRest %.2f (dRest-dA) %.2f | sum|dA| %.1f sum|dB| %.1f |dC| %.1f |dD| %.1f | slope dA on dRest %.2f"%(
      L,c(tA,tB),c(tB,tC),c(tC,tD),c(tA,tD),c(res["h1"][1][0],res["h2"][1][0]),c(res["h1"][1][1:].mean(0),res["h2"][1][1:].mean(0)),c(comp1,comp2),
      d[0].abs().sum(),d[1].abs().sum(),d[2].abs().sum(),d[3].abs().sum(), float((d[0]*rest).sum()/(rest*rest).sum())))
    out[L]=dict(t=res["all"][0],d=d)
torch.save(out,"editor_runs/gemma_b25/notes/neuron_pos_t.pt")
