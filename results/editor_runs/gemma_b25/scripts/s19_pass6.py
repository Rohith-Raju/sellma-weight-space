import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from lib import *
PT=torch.load("editor_runs/gemma_b25/notes/neuron_pos_t.pt")
# control slopes
for L in [14,15,16]:
    d=PT[L]["d"]
    sl=lambda y,x: float((y*x).sum()/(x*x).sum())
    print(L,"slope dA on mean(dB,dC,dD) %.2f | dB on mean(dA,dC,dD) %.2f | dB on mean(dC,dD) %.2f | dC on mean(dB,dD) %.2f | dD on mean(dB,dC) %.2f | dA on mean(dC,dD) %.2f"%(
      sl(d[0],d[1:].mean(0)),sl(d[1],d[[0,2,3]].mean(0)),sl(d[1],d[2:].mean(0)),sl(d[2],d[[1,3]].mean(0)),sl(d[3],d[1:3].mean(0)),sl(d[0],d[2:].mean(0))))
cfg=eval(sys.argv[1])
tt={L:PT[L]["t"][0] for L in PT}
e=scale_edit(cfg["layers"],cfg["k"],cfg["gamma"],tt=tt)
print({k:v for k,v in wb.edit_size(e).items() if k!="tensors"})
items=evalset()
ans=wb.answer(items,edit=e)
report(items,ans,str(cfg))
