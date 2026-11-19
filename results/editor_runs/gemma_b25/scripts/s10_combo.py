import numpy as np, torch, json
exec(open("editor_runs/gemma_b25/scripts/s09_slope.py").read().split("for L in [16,20]:")[0].replace("print({k","#"))
mok0=(lp.argmax(1)==key).float().mean()
print("model acc",mok0)
for L in [14,16,18,20]:
    s=cvscore(L); sc=s-s.mean(1,keepdim=True)
    for name,tgt in [("L23lens",lens(last[:,23])),("final",lp)]:
        t=tgt-tgt.mean(1,keepdim=True)
        res=[]
        for lam in [0,5,10,20,40,80,1e4]:
            c=t+lam*sc; ok=(c.argmax(1)==key)
            res.append((lam,round(ok.float().mean().item(),3),round(ok[key==0].float().mean().item(),2)))
        # A-only boost
        res2=[]
        for lam in [5,10,20,40]:
            c=t.clone(); c[:,0]+=lam*sc[:,0]; ok=(c.argmax(1)==key); res2.append((lam,round(ok.float().mean().item(),3)))
        print("probe L%d + %s:"%(L,name),res," | A-only:",res2)
