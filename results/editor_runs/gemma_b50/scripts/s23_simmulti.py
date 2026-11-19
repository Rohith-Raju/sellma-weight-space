import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
exec(open(RUN+"scripts/s12_setup.py").read())
Lr=23; S,V,q,xd,parts=attn_layer(Lr); Wo=parts["Wo"]; gpa=parts["gpa"]
nl=[names.index(f"nl{j}") for j in range(4)]; N=len(key)
multi=torch.tensor([" – " in byid[qq].question.options["A"] for qq in qids])
Vs=torch.stack(V,2); att0=torch.softmax(S,-1); an_part=rms(((att0[...,None]*Vs).sum(2)).reshape(N,-1)@Wo.T,gpa)
def lens(x): return (x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*(1+nw))@E.T
def run(S2):
    att2=torch.softmax(S2,-1); raw2=((att2[...,None]*Vs).sum(2)).reshape(N,-1)@Wo.T
    return lens(F[:,Lr]+(rms(raw2,gpa)-an_part)).argmax(1)
def show(tag,g):
    print(f"{tag}: all {(g==key).float().mean():.3f} | multi {(g[multi]==key[multi]).float().mean():.3f} by key {[round((g[multi&(key==j)]==j).float().mean().item(),2) for j in range(4)]} fracA {(g[multi]==0).float().mean():.2f} | single {(g[~multi]==key[~multi]).float().mean():.3f} by key {[round((g[(~multi)&(key==j)]==j).float().mean().item(),2) for j in range(4)]}")
show("base",run(S))
print("--- add offset to nl0 score (heads 2,3) for multi items only")
for d in [0.5,1,1.5,2,2.5,3,4]:
    S2=S.clone(); S2[multi,3,nl[0]]+=d; S2[multi,2,nl[0]]+=d*0.45; show(f"add {d}",run(S2))
print("--- replace nl0 score by constant for multi items (head 3 only)")
for c in [11,12,12.5,13,13.5,14,14.5,15]:
    S2=S.clone(); S2[multi,3,nl[0]]=c; show(f"const {c}",run(S2))
s3=S[:,3][:,nl]
print("multi: head3 nl0 score key/nonkey mean, sd:",s3[multi&(key==0),0].mean().item(),s3[multi&(key!=0),0].mean().item(),s3[multi,0].std().item())
print("single:",s3[(~multi)&(key==0),0].mean().item(),s3[(~multi)&(key!=0),0].mean().item())
