import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from lib import *
PT=torch.load("editor_runs/gemma_b25/notes/neuron_pos_t.pt")
# comparative neurons: code correctness at B/C/D but not at A  -> attenuate
tt={}
for L in PT:
    t=PT[L]["t"]; rest=t[1:].mean(0)
    comp=rest.abs()-t[0].abs()*1.0
    comp[(rest.abs()<3)]=0
    tt[L]=comp.clamp(min=0)
    if L in (13,14,15,16,17,18): print(L,"n comparative candidates",int((tt[L]>0).sum()))
e=scale_edit([13,14,15,16,17,18],256,-0.5,tt=tt)
print({k:v for k,v in wb.edit_size(e).items() if k!="tensors"})
items=evalset()
ans=wb.answer(items,edit=e)
report(items,ans,"attenuate comparative L13-18 k256 g-0.5")
