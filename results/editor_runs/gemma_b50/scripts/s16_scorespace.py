import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
import itertools
exec(open(RUN+"scripts/s12_setup.py").read())
S=torch.load(RUN+"notes/attn_scores.pt")[23]
nl=[names.index(f"nl{j}") for j in range(4)]
s=S[:,3][:,nl]   # [N,4]
print("argmax raw ==key",(s.argmax(1)==key).float().mean().item(),"==cho",(s.argmax(1)==cho).float().mean().item())
print("acc by key raw:",[round((s[key==j].argmax(1)==j).float().mean().item(),2) for j in range(4)], "frac argmax",[round((s.argmax(1)==j).float().mean().item(),2) for j in range(4)])
best=(0,None)
grid=[-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
for b0,b1,b2 in itertools.product(grid,grid,grid):
    a=((s+torch.tensor([b0,b1,b2,0.]))).argmax(1)
    acc=(a==key).float().mean().item()
    if acc>best[0]: best=(acc,(b0,b1,b2))
print("best offsets (vs key)",best)
mu=torch.stack([s[key!=j,j].mean() for j in range(4)]); sd=torch.stack([s[key!=j,j].std() for j in range(4)])
z=(s-mu)/sd; print("z-scored argmax==key",(z.argmax(1)==key).float().mean().item(), "by key",[round((z[key==j].argmax(1)==j).float().mean().item(),2) for j in range(4)])
z=(s-mu); print("mean-centered argmax==key",(z.argmax(1)==key).float().mean().item(), "by key",[round((z[key==j].argmax(1)==j).float().mean().item(),2) for j in range(4)])
# what does model do on key==A failures: head3 argmax and model choice
m=(key==0)&(cho!=0)
print("key A & model wrong: n",m.sum().item()," head3 argmax dist",torch.bincount(s[m].argmax(1),minlength=4).tolist()," model choice dist",torch.bincount(cho[m],minlength=4).tolist())
print(" margins s_A - max other in those:",(s[m,0]-s[m,1:].max(1).values).round(decimals=1).tolist())
m=(key==0)&(cho==0); print(" margins when right:",(s[m,0]-s[m,1:].max(1).values).round(decimals=1).tolist())
for j in [1,2,3]:
    m=(key==j)&(cho!=j); oth=[k for k in range(4) if k!=j]
    print(f" key {L[j]} wrong n={m.sum().item()} margins:",(s[m,j]-s[m][:,oth].max(1).values).round(decimals=1).tolist())
