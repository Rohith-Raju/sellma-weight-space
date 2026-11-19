# usage: evalpass.py <edit.pt> <mode>   mode: A = MEK290+LAS222 original order ; B = ORD 290 + MEK 111 reversed + LAS 111 reversed
import sys, torch, pickle, numpy as np
from bench.editor import Workbench
wb = Workbench.open("gemma_b10")
P=wb.practice_questions(); byid={p.id:p for p in P}
m1=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
E=torch.load(sys.argv[1]); mode=sys.argv[2]
name=wb.layer_param(23,"self_attn.q_proj.weight")
edit={name:{"rows":E["rows"],"values":E["delta"]}} if "rows" in E else E["edit"]
las_ids=[q for (q,o) in m1 if o=="ABCD" and byid[q].question.section=="LAS"]
mek_ids=[q for (q,o) in m1 if o=="ABCD" and byid[q].question.section=="MEK"]
if mode=="A":
    items=[byid[q] for q in mek_ids+las_ids]; orders=[None]*len(items)
else:
    ords=[p for p in P if p.question.section=="ORD"]
    items=ords+[byid[q] for q in mek_ids[::3][:97]]+[byid[q] for q in las_ids[::2][:111]]
    items=items[:512]; orders=[None]*len(ords)+["DCBA"]*(len(items)-len(ords))
print(len(items)); assert len(items)<=512
ans=wb.answer(items,edit=edit,options_orders=orders)
print("passes left",wb.passes_left)
