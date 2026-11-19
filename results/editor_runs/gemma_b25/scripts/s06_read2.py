# PASS 2 (read): MEK prompts cut at the end of each option text; all layers.
from bench.editor import Workbench
import json
wb = Workbench.open("gemma_b25")
P = wb.practice_questions()
mek=[p for p in P if p.question.section=="MEK"]
sel=[p for p in mek if p.paper=="validation"]+[p for p in mek if p.paper=="drill"][:206]
meta=[];texts=[]
for p in sel:
    t=wb.render_prompt(p)
    for l in "ABCD":
        opt=f"\n{l}. {p.question.options[l]}\n"
        i=t.index(opt); assert t.count(opt)==1
        texts.append(t[:i+len(opt)-1]); meta.append((p.id,l))
print(len(texts)); print(repr(texts[1][-80:]))
json.dump(meta,open("editor_runs/gemma_b25/notes/read2_meta.json","w"))
r=wb.read(texts,layers=list(range(34)),top_k=5)
print(r.residual.shape, wb.passes_left)
