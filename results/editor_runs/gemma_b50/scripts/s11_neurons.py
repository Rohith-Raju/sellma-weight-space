import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
meta=json.load(open(RUN+"notes/prefix_meta.json"))
names=["let0","end0","nl0","let1","end1","nl1","let2","end2","nl2","let3","end3","nl3","t_paren","t_eot","t_nl","t_sot","t_model"]
nm=np.array([n for q,n in meta])
out={}
for l in range(10,23):
    A=torch.cat([wb.load_pass(i).mlp_acts[:,l] for i in (2,3,4)])  # [3060,10240]
    on=A[torch.tensor(nm=="nl0")]; off=A[torch.tensor(nm!="nl0")]
    lo=on.quantile(0.1,dim=0); med=on.median(0).values
    offhi=off.abs().quantile(0.99,dim=0); offmean=off.abs().mean(0)
    sel=lo/(offhi+1e-3)      # positive-firing selective
    sel2=(-on.quantile(0.9,dim=0))/(offhi+1e-3)
    for nm_,s in [("pos",sel),("neg",sel2)]:
        top=s.topk(8)
        print(f"L{l} {nm_}:"," ".join(f"{j}(sel{v:.1f},med{med[j]:.2f},lo{lo[j]:.2f},off99 {offhi[j]:.2f})" for v,j in zip(top.values.tolist(),top.indices.tolist())))
    out[l]={"med":med,"lo":lo,"offhi":offhi,"sel":sel,"sel2":sel2}
    print("   act scale: mean|a| all", A.abs().mean().item(), "max",A.abs().max().item())
torch.save(out,RUN+"notes/nl0_selectivity.pt")
