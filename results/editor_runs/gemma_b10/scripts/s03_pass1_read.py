# PASS 1 (read): MEK+LAS prompts in original and reversed option order, all layers.
from bench.editor import Workbench
import torch, pickle
wb = Workbench.open("gemma_b10")
P = wb.practice_questions()
mek=[p for p in P if p.question.section=="MEK"]
las=[p for p in P if p.question.section=="LAS"]
import random
random.seed(0)
# LAS: choose 222, stratified by key letter
las_sorted=sorted(las,key=lambda p:(p.question.answer,p.id))
las_sel=las_sorted[::1]
random.shuffle(las_sel)
las_sel=las_sel[:222]
items=mek+las_sel
meta=[];texts=[]
for p in items:
    for order in [None,"DCBA"]:
        texts.append(wb.render_prompt(p,order) if order else wb.render_prompt(p))
        meta.append((p.id,order or "ABCD"))
nt=[len(wb.tokenizer(t,add_special_tokens=False).input_ids) for t in texts]
print(len(texts),max(nt),sum(nt))
assert len(texts)==1024 and max(nt)<=4096
r=wb.read(texts,layers=list(range(34)),top_k=100)
pickle.dump(meta,open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","wb"))
print("done",wb.passes_left)
