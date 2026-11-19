import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
r=wb.load_pass(1); S=np.load(RUN+"notes/pass1_summary.npz")
key=S["key"];ch=S["ch"];sec=S["sec"];rot=S["rot"]
ki=torch.tensor(["ABCD".index(k) for k in key]); ci=torch.tensor(["ABCD".index(k) for k in ch])
lg=torch.load(RUN+"notes/pass1_lens.pt")
mek=torch.tensor(sec=="MEK"); las=torch.tensor(sec=="LAS")
# 1. offset analysis on lens logits per layer
for l in [21,22,23,24,25,27,29,31,33]:
    out=[]
    for b in [0,0.5,1,1.5,2,3,4,6]:
        g=lg[:,l].clone(); g[:,0]+=b; am=g.argmax(-1)
        out.append(f"{b}:{(am[mek]==ki[mek]).float().mean():.3f}/{(am[las]==ki[las]).float().mean():.3f}")
    print("L",l,"A-offset acc MEK/LAS:"," ".join(out))
# 2. where does the decision come from: per-layer delta of residual, split MLP vs attention
w=wb.weights()
nw=w.float32("model.language_model.norm.weight")
ids=[236776, 236799, 236780, 236796]; E=wb.unembedding_rows(ids)
res=r.residual
N=res.shape[0]
onehot=torch.nn.functional.one_hot(ki,4).float()
print("layer | key-direction projection: mlp-out, attn-out (mean over items, by shown key letter), using raw residual direction E_key - mean E_others")
D=E-E.mean(0,keepdim=True)   # centered letter dirs
for l in range(17,34):
    W=w.float32(wb.layer_param(l)); g=w.float32(wb.layer_param(l,"post_feedforward_layernorm.weight"))
    raw=r.mlp_acts[:,l]@W.T
    mo=raw*torch.rsqrt(raw.pow(2).mean(-1,keepdim=True)+1e-6)*(1+g)
    prev=res[:,l-1]
    ao=res[:,l]-prev-mo
    # scale by final-norm of current layer residual for comparability
    sc=torch.rsqrt(res[:,l].pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw)
    pm=(mo*sc)@D.T; pa=(ao*sc)@D.T    # [N,4] contribution to centered letter logits
    km=[pm[ki==j,j].mean().item() for j in range(4)]; ka=[pa[ki==j,j].mean().item() for j in range(4)]
    bm=[pm[ki!=j,j].mean().item() for j in range(4)]; ba=[pa[ki!=j,j].mean().item() for j in range(4)]
    print(f"L{l} MLP key {[round(x,2) for x in km]} nonkey {[round(x,2) for x in bm]} | ATT key {[round(x,2) for x in ka]} nonkey {[round(x,2) for x in ba]} | rms raw {raw.pow(2).mean(-1).sqrt().mean():.2f}")
