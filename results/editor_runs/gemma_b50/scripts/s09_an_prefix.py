import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
meta=json.load(open(RUN+"notes/prefix_meta.json"))
R=torch.cat([wb.load_pass(i).residual for i in (2,3,4)])   # [3060,34,2560]
print(R.shape)
names=["let0","end0","nl0","let1","end1","nl1","let2","end2","nl2","let3","end3","nl3","t_paren","t_eot","t_nl","t_sot","t_model"]
qids=[]; 
for q,n in meta:
    if q not in qids: qids.append(q)
ix={(q,n):i for i,(q,n) in enumerate(meta)}
key=torch.tensor(["ABCD".index(byid[q].question.answer) for q in qids])
cho=torch.tensor(["ABCD".index(byid[q].baseline.chosen_letter) for q in qids])
X={n:R[[ix[(q,n)] for q in qids]] for n in names}   # each [180,34,2560]
torch.save({"X":X,"key":key,"cho":cho,"qids":qids},RUN+"notes/prefix_res.pt")
w=wb.weights(); nw=w.float32("model.language_model.norm.weight")
E=wb.unembedding_rows([236776, 236799, 236780, 236796])
def lens(x): return (x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw))@E.T
print("baseline acc of these 180:",(key==cho).float().mean().item())
print("== tail lens: argmax==key / ==final choice per layer")
for n in names[12:]:
    g=lens(X[n]).argmax(-1)
    print(n," ".join(f"{l}:{(g[:,l]==key).float().mean():.2f}/{(g[:,l]==cho).float().mean():.2f}" for l in range(14,34,1)))
# option positions: is-key probe via difference of means, half split CV
N=len(qids); tr=torch.arange(N)%2==0; te=~tr
def auc(pos,neg):
    return (pos[:,None]>neg[None]).float().mean().item()
for kind in ["end","nl"]:
    print("== position kind",kind,": dprime-like stats of key-vs-nonkey mean-diff probe (train on even, test on odd)")
    for l in range(6,34,1):
        pos_tr=[];neg_tr=[]
        for j in range(4):
            x=X[f"{kind}{j}"][:,l]
            pos_tr.append(x[tr&(key==j)]); neg_tr.append(x[tr&(key!=j)])
        # per-position centering to remove position identity
        d=torch.stack([pos_tr[j].mean(0)-neg_tr[j].mean(0) for j in range(4)]).mean(0)
        dn=d/d.norm()
        out=[]
        for j in range(4):
            x=X[f"{kind}{j}"][:,l]@dn
            a=auc(x[te&(key==j)],x[te&(key!=j)])
            # also "chosen" separability
            c=auc(x[te&(cho==j)],x[te&(cho!=j)])
            out.append(f"{L[j]} auc {a:.2f} cho {c:.2f} gap {(x[te&(key==j)].mean()-x[te&(key!=j)].mean()).item():7.1f}")
        print(f"L{l:2d} |d| {d.norm():8.1f} resnorm {X[kind+str(1)][:,l].norm(dim=-1).mean():8.0f} | "+" | ".join(out))
