import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
r=wb.load_pass(1); S=np.load(RUN+"notes/pass1_summary.npz")
key=S["key"];ch=S["ch"];sec=S["sec"];rot=S["rot"]
w=wb.weights()
nw=w.float32("model.language_model.norm.weight")
ids=[wb.token_ids(l)[0] for l in "ABCD"]; print(ids)
E=wb.unembedding_rows(ids)  # [4,2560]
print("E norms",E.norm(dim=1), "cos", torch.nn.functional.cosine_similarity(E[:,None],E[None],dim=-1))
def lens(x):
    xn=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw)
    return xn@E.T
res=r.residual  # [N,34,2560]
lg=lens(res)    # [N,34,4]
# verify final
pr=torch.softmax(lg[:,33],-1)
print("final lens argmax agrees with chosen:", (np.array(list("ABCD"))[lg[:,33].argmax(-1).numpy()]==ch).mean())
print("residual norm by layer", res.norm(dim=-1).mean(0).round().tolist())
ki=torch.tensor(["ABCD".index(k) for k in key]); ci=torch.tensor(["ABCD".index(k) for k in ch])
for s in ["MEK","LAS"]:
    m=torch.tensor(sec==s)
    print("==",s)
    for l in range(34):
        g=lg[m,l]; k=ki[m]
        am=g.argmax(-1)
        acc=(am==k).float().mean().item()
        agree=(am==ci[m]).float().mean().item()
        cen=g-g.mean(-1,keepdim=True)
        # mean centered logit per letter (bias), and when key
        bias=cen.mean(0)
        keyl=torch.stack([cen[k==j,j].mean() for j in range(4)])
        nonkey=torch.stack([cen[k!=j,j].mean() for j in range(4)])
        frac=torch.bincount(am,minlength=4).float()/len(am)
        print(f"L{l:2d} acc {acc:.3f} agreeFinal {agree:.3f} argmax-frac {[round(x,2) for x in frac.tolist()]} key-logit {[round(x,1) for x in keyl.tolist()]} nonkey {[round(x,1) for x in nonkey.tolist()]}")
torch.save(lg,RUN+"notes/pass1_lens.pt")
