import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
exec(open(RUN+"scripts/s12_setup.py").read())
Lr=23; h=3; kv=1
pre=f"model.language_model.layers.{Lr}."
S,V,q,xd,parts=attn_layer(Lr)
Wk=parts["Wk"]; gk=parts["gk"]; gi=parts["gi"]; Wo=parts["Wo"]; gpa=parts["gpa"]
nl=[names.index(f"nl{j}") for j in range(4)]; N=len(key)
XN={n:rms(X[n][:,Lr-1],gi) for n in names}; XN["self"]=xd
allnames=names+["self"]
# extra negatives: final positions of all pass-1 prompts (MEK/LAS orig + rotated)
xfin_all=rms(r1.residual[:,Lr-1],gi)
Mx=torch.stack([XN[n].mean(0) for n in allnames]+[xfin_all.mean(0)])
Pinv=torch.linalg.pinv(Mx); s=Pinv[:,allnames.index("nl0")]
print("|s|",s.norm().item()); 
for n in allnames: print(f"  s-proj {n}: mean {(XN[n]@s).mean():.3f} sd {(XN[n]@s).std():.3f} min {(XN[n]@s).min():.2f} max {(XN[n]@s).max():.2f}")
pf=xfin_all@s; print("  s-proj all final positions (1024): mean %.3f sd %.3f max %.3f"%(pf.mean(),pf.std(),pf.abs().max()))
# a: mean head-3 query expressed in the frame of the nl0 source position
def unrope(x,pos,base=1e6): return rope(x,-pos,base)
a=unrope(q[:,h],POS["nl0"]).mean(0)       # [hd]
a2=unrope(q[:,2],POS["nl0"]).mean(0)
print("|a|",a.norm().item(),"mean |q|",q[:,h].norm(dim=-1).mean().item(),"cos(a_h3,a_h2)",torch.nn.functional.cosine_similarity(a,a2,dim=0).item())
ah=a/a.norm()
def scores_with(gamma):
    out=[]
    for n in allnames:
        xs=XN[n]; ps=POS[n] if n!="self" else Fpos
        kp=(xs@Wk.T).view(-1,KV,hd)
        kp[:,kv]=kp[:,kv]+gamma*(xs@s)[:,None]*ah
        k=rope(rms(kp,gk),ps[:,None].expand(-1,KV),1e6).repeat_interleave(2,dim=1)
        out.append((q*k).sum(-1)/16)
    return torch.stack(out,-1)
S0=scores_with(0.0); print("recompute check",(S0-S).abs().max().item())
kn=(XN["nl0"]@Wk.T).view(-1,KV,hd)[:,kv].norm(dim=-1).mean(); print("mean |k_pre| at nl0 (kv head 1):",kn.item())
Vs=torch.stack(V,2); att0=torch.softmax(S,-1); an_part=rms(((att0[...,None]*Vs).sum(2)).reshape(N,-1)@Wo.T,gpa)
def lens(x): return (x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*(1+nw))@E.T
for gamma in [0,2,4,6,8,12,16]:
    S2=scores_with(gamma); d=S2-S
    att2=torch.softmax(S2,-1); raw2=((att2[...,None]*Vs).sum(2)).reshape(N,-1)@Wo.T
    g=lens(F[:,Lr]+(rms(raw2,gpa)-an_part)).argmax(1)
    print(f"gamma {gamma}: dscore nl0 h3 {d[:,3,nl[0]].mean():.2f} (sd {d[:,3,nl[0]].std():.2f}) h2 {d[:,2,nl[0]].mean():.2f} | max |mean dscore| elsewhere h3 {d[:,3].mean(0)[[i for i in range(18) if i!=nl[0]]].abs().max():.2f} | sim acc {(g==key).float().mean():.3f} fracA {(g==0).float().mean():.3f} by key {[round((g[key==j]==j).float().mean().item(),2) for j in range(4)]}")
torch.save({"s":s,"ah":ah},RUN+"notes/kedit_dirs.pt")
