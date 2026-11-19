import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
Z=torch.load(RUN+"notes/prefix_res.pt"); X=Z["X"]; key=Z["key"]; cho=Z["cho"]
N=len(key)
for l in [15,17,19,21,22]:
    # direction from all data: chosen-vs-not, and key-vs-not
    for lab,name in [(key,"key"),(cho,"cho")]:
        d=torch.stack([X[f"nl{j}"][lab==j,l].mean(0)-X[f"nl{j}"][lab!=j,l].mean(0) for j in range(4)]).mean(0); dn=d/d.norm()
        sc=torch.stack([X[f"nl{j}"][:,l]@dn for j in range(4)],1)   # [N,4]
        print(f"L{l} dir={name} |d|={d.norm():.0f}")
        print("   mean score when key   :",[round(sc[key==j,j].mean().item()) for j in range(4)],"  nonkey:",[round(sc[key!=j,j].mean().item()) for j in range(4)], " sd nonkey",[round(sc[key!=j,j].std().item()) for j in range(4)])
        print("   mean score when chosen:",[round(sc[cho==j,j].mean().item()) for j in range(4)],"  nonchosen:",[round(sc[cho!=j,j].mean().item()) for j in range(4)])
        mu=torch.stack([sc[key!=j,j].mean() for j in range(4)])
        z=sc-mu
        print("   argmax(z) predicts chosen:",(z.argmax(1)==cho).float().mean().item()," ==key:",(z.argmax(1)==key).float().mean().item(), "raw argmax==cho",(sc.argmax(1)==cho).float().mean().item())
        for b in [0,50,100,150,200,300]:
            z2=z.clone(); z2[:,0]+=b
            print(f"      A+{b}: pred acc vs key {(z2.argmax(1)==key).float().mean():.3f}  frac A {(z2.argmax(1)==0).float().mean():.3f}",end=";")
        print()
        for gsc in [1.25,1.5,2.0]:
            z2=z.clone(); z2[:,0]*=gsc
            print(f"      A gain x{gsc}: pred acc vs key {(z2.argmax(1)==key).float().mean():.3f} frac A {(z2.argmax(1)==0).float().mean():.3f}",end=";")
        print()
# cosine of direction across layers and between nl positions
l=19
ds=[X[f"nl{j}"][cho==j,l].mean(0)-X[f"nl{j}"][cho!=j,l].mean(0) for j in range(4)]
print("cos between per-position cho-directions L19:",torch.nn.functional.cosine_similarity(torch.stack(ds)[:,None],torch.stack(ds)[None],dim=-1))
