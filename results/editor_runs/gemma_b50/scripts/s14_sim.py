import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
exec(open(RUN+"scripts/s12_setup.py").read())
Lr=23; h=3
S,V,q,xd,parts=attn_layer(Lr)
att=torch.softmax(S,-1)            # [N,H,18]
Wo=parts["Wo"]; gpa=parts["gpa"]
nl=[names.index(f"nl{j}") for j in range(4)]
N=len(key)
# partial reconstruction of raw attention output
Vs=torch.stack(V,2)                # [N,H,18,hd]
mix=(att[...,None]*Vs).sum(2)      # [N,H,hd]
raw_part=mix.reshape(N,-1)@Wo.T    # [N,2560]
# truth
Wd=w.float32(wb.layer_param(Lr)); gf=w.float32(wb.layer_param(Lr,"post_feedforward_layernorm.weight"))
acts=r1.mlp_acts[[i1[qq] for qq in qids]][:,Lr]
mo=rms(acts@Wd.T,gf)
# mlp reads residual after attention: x_mid = F[22]+attn_normed ; truth attn_normed = F[23]-F[22]-mo
an_true=F[:,Lr]-F[:,Lr-1]-mo
an_part=rms(raw_part,gpa)
print("cos(partial recon, true attn out):",torch.nn.functional.cosine_similarity(an_part,an_true,dim=-1).mean().item(), "norm true",an_true.norm(dim=-1).mean().item(),"norm part",an_part.norm(dim=-1).mean().item())
rms_part=raw_part.pow(2).mean(-1,keepdim=True).sqrt()
def lens(x): return (x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*(1+nw))@E.T
base_lens=lens(F[:,Lr]); print("L23 lens acc (true):",(base_lens.argmax(1)==key).float().mean().item(),"agree cho",(base_lens.argmax(1)==cho).float().mean().item())
# model with partial attn: residual = F22 + an_part + mo(approx same)
part_lens=lens(F[:,Lr-1]+an_part+mo); print("L23 lens acc (partial recon):",(part_lens.argmax(1)==key).float().mean().item(),"agree cho",(part_lens.argmax(1)==cho).float().mean().item())
# head-3 write directions per option
Wo_h=Wo[:,h*hd:(h+1)*hd]
vm=torch.stack([V[nl[j]][:,h].mean(0) for j in range(4)])     # [4,hd] mean value at nl_j
wr=vm@Wo_h.T                                                 # [4,2560] mean raw write when attending nl_j
Ec=(E-E.mean(0)); Ecn=Ec/Ec.norm(dim=1,keepdim=True)
print("head3 mean write (raw) projected on unit centered letter dirs (rows src nl_j):\n",(wr@Ecn.T))
print("norm of writes",wr.norm(dim=1))
wc=wr-wr.mean(0)
print("centered writes on letter dirs:\n",(wc@Ecn.T)); 
print("cos(centered write_j, Ec_j):",[torch.nn.functional.cosine_similarity(wc[j],Ec[j],dim=0).item() for j in range(4)])
# value-space direction specific to nl0
m=vm[0]-vm[1:].mean(0); mh=m/m.norm()
print("proj of v(nl_j) on mh: mean",[ (V[nl[j]][:,h]@mh).mean().item() for j in range(4)],"sd",[ (V[nl[j]][:,h]@mh).std().item() for j in range(4)], "others:",{n:round((V[i][:,h]@mh).mean().item(),2) for i,n in enumerate(names+["self"]) if not n.startswith("nl")})
torch.save({"mh":mh,"vm":vm,"wr":wr},RUN+"notes/head3_dirs.pt")
# calibrate: target extra A-write so that (write_0 . Ecn_0) centered equals mean of j=1..3
diag=(wc@Ecn.T).diag(); target=diag[1:].mean()-diag[0]
p0=(vm[0]@mh)
print("diag",diag,"target extra",target.item(),"p0",p0.item())
for mult in [0,0.5,1,1.5,2,3]:
    beta=mult*target/p0
    dWo=beta*torch.outer(Ecn[0],mh)                # [2560,hd]
    draw=(mix[:,h]@dWo.T)
    an_new=rms(raw_part+draw,gpa)
    # apply delta to truth
    newres=F[:,Lr]+(an_new-an_part)
    g=lens(newres).argmax(1)
    accs=[(g[key==j]==j).float().mean().item() for j in range(4)]
    print(f"mult {mult}: sim L23 lens acc {(g==key).float().mean():.3f} fracA {(g==0).float().mean():.3f} acc by key {[round(a,2) for a in accs]} |dWo|={dWo.norm():.3f} |Wo_h|={Wo_h.norm():.3f}")
print("======== dual-vector version")
M=torch.stack([V[i][:,h].mean(0) for i in range(len(names)+1)])     # [18,hd]
Pinv=torch.linalg.pinv(M)             # [hd,18]
md=Pinv[:,names.index("nl0")]         # m.M[nl0]=1, others 0
print("|md|",md.norm().item(),"proj means per source:",{n:round((V[i][:,h]@md).mean().item(),2) for i,n in enumerate(names+["self"])})
print("proj sd per source:",{n:round((V[i][:,h]@md).std().item(),2) for i,n in enumerate(names+["self"])})
torch.save({"md":md,"M":M,"Ecn":Ecn,"target":target},RUN+"notes/head3_dual.pt")
for mult in [0,0.5,1,1.5,2,3,4,6]:
    dWo=mult*target*torch.outer(Ecn[0],md)
    draw=(mix[:,h]@dWo.T)
    an_new=rms(raw_part+draw,gpa)
    newres=F[:,Lr]+(an_new-an_part)
    lg=lens(newres); g=lg.argmax(1)
    accs=[(g[key==j]==j).float().mean().item() for j in range(4)]
    print(f"mult {mult}: sim L23 lens acc {(g==key).float().mean():.3f} fracA {(g==0).float().mean():.3f} acc by key {[round(a,2) for a in accs]} |dWo|={dWo.norm():.3f} meanAlogit-shift {(lg[:,0]-base_lens[:,0]).mean():.2f}")
print("======== score offset on nl0 for head 3 (and optionally head 2 which shares keys), plus OV gain")
def sim(delta,mult,both=False,verbose=True):
    S2=S.clone(); S2[:,3,nl[0]]+=delta
    if both: S2[:,2,nl[0]]+=delta
    att2=torch.softmax(S2,-1)
    mix2=(att2[...,None]*Vs).sum(2)
    raw2=mix2.reshape(N,-1)@Wo.T
    dWo=mult*target*torch.outer(Ecn[0],md)
    raw2=raw2+mix2[:,h]@dWo.T
    newres=F[:,Lr]+(rms(raw2,gpa)-an_part)
    g=lens(newres).argmax(1)
    accs=[(g[key==j]==j).float().mean().item() for j in range(4)]
    if verbose: print(f"delta {delta} mult {mult} both={both}: acc {(g==key).float().mean():.3f} fracA {(g==0).float().mean():.3f} by key {[round(a,2) for a in accs]} att(nl0|key0) {att2[key==0,3,nl[0]].mean():.2f} att(nl0|key!=0) {att2[key!=0,3,nl[0]].mean():.2f}")
for both in [False,True]:
    for delta in [0,0.5,1,1.5,2,2.5,3,4]:
        sim(delta,0,both)
for delta in [1,1.5,2,3]:
    for mult in [1,2,4]:
        sim(delta,mult,True)
print("======== grid: offsets on nl0 (rows) and nl3 (cols) for head 3 only / both heads 2+3")
def sim2(d0,d3,both):
    S2=S.clone(); S2[:,3,nl[0]]+=d0; S2[:,3,nl[3]]+=d3
    if both: S2[:,2,nl[0]]+=d0; S2[:,2,nl[3]]+=d3
    att2=torch.softmax(S2,-1); mix2=(att2[...,None]*Vs).sum(2); raw2=mix2.reshape(N,-1)@Wo.T
    g=lens(F[:,Lr]+(rms(raw2,gpa)-an_part)).argmax(1)
    return (g==key).float().mean().item()
for both in [False,True]:
    print("both",both,"cols d3:",[0,-0.5,-1,-1.5,-2,-3])
    for d0 in [0,0.5,1,1.5,2,2.5]:
        print(d0,[round(sim2(d0,d3,both),3) for d3 in [0,-0.5,-1,-1.5,-2,-3]])
