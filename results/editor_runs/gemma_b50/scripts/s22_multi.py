import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
Z=torch.load(RUN+"notes/prefix_res.pt"); X=Z["X"]; key=Z["key"]; cho=Z["cho"]; qids=Z["qids"]
multi=torch.tensor([" – " in byid[q].question.options["A"] for q in qids])
print("n multi",multi.sum().item(),"single",(~multi).sum().item())
cos=torch.nn.functional.cosine_similarity
for kind in ["nl","end"]:
    print("kind",kind,": multi-minus-single mean direction per position; norm, and projection of each position diff on the shared dir of positions 1-3 (unit)")
    for l in [4,6,8,10,12,14,16,18,20,22]:
        u=[X[f"{kind}{j}"][multi,l].mean(0)-X[f"{kind}{j}"][~multi,l].mean(0) for j in range(4)]
        sh=(u[1]+u[2]+u[3])/3; shn=sh/sh.norm()
        # leave-one-out for 1..3
        pr=[]
        for j in range(4):
            if j==0: d=shn
            else:
                o=[k for k in (1,2,3) if k!=j]; d=(u[o[0]]+u[o[1]])/2; d=d/d.norm()
            pr.append((u[j]@d).item())
        rn=X[f"{kind}1"][:,l].norm(dim=-1).mean().item()
        print(f" L{l:2d} resnorm {rn:7.0f} |u_j| {[round(x.norm().item()) for x in u]} proj on shared {[round(x) for x in pr]} cos(u0,shared) {cos(u[0],sh,dim=0):.2f} cos(u1,u2) {cos(u[1],u[2],dim=0):.2f}")
# fit signal: key-vs-nonkey gap along cho-direction split by multi (L19, L21)
for l in [17,19,21]:
    d=torch.stack([X[f"nl{j}"][cho==j,l].mean(0)-X[f"nl{j}"][cho!=j,l].mean(0) for j in range(4)]).mean(0); dn=d/d.norm()
    for lab,m in [("single",~multi),("multi",multi)]:
        print(f"L{l} {lab}: key-nonkey gap on cho-dir by pos",[round(((X[f"nl{j}"][m&(key==j),l]@dn).mean()-(X[f"nl{j}"][m&(key!=j),l]@dn).mean()).item()) for j in range(4)])
