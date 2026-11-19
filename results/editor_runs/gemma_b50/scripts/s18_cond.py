import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
exec(open(RUN+"scripts/s12_setup.py").read())
S=torch.load(RUN+"notes/attn_scores.pt")[23]
nl=[names.index(f"nl{j}") for j in range(4)]
s=S[:,3][:,nl]
print("head3 score at nl_j (cols) given key k (rows): mean")
M=torch.stack([s[key==k].mean(0) for k in range(4)]); print(M.round(decimals=2))
print("sd"); print(torch.stack([s[key==k].std(0) for k in range(4)]).round(decimals=2))
print("max-nonkey mean by key:",[round(s[key==k][:,[j for j in range(4) if j!=k]].max(1).values.mean().item(),2) for k in range(4)])
ng=torch.tensor([byid[q].question.question.count("____")>=1 and (" – " in byid[q].question.options["A"]) for q in qids])
print("multi-gap share",ng.float().mean().item())
for lab,m in [("single",~ng),("multi",ng)]:
    print(lab,"key-score by key:",[round(s[(key==k)&m,k].mean().item(),2) for k in range(4)],"max-nonkey:",[round(s[(key==k)&m][:,[j for j in range(4) if j!=k]].max(1).values.mean().item(),2) for k in range(4)],"n",[int(((key==k)&m).sum()) for k in range(4)], "acc(model)",[round((cho[(key==k)&m]==k).float().mean().item(),2) for k in range(4)])
