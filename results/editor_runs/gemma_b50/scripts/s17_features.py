import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
import re
def tab(name,items,f):
    from collections import defaultdict
    d=defaultdict(list)
    for p in items: d[f(p)].append(p.baseline.is_correct)
    print(name,{k:(round(sum(v)/len(v),2),len(v)) for k,v in sorted(d.items(),key=lambda kv:str(kv[0]))})
mek=[p for p in P if p.question.section=="MEK"]; las=[p for p in P if p.question.section=="LAS"]; ord_=[p for p in P if p.question.section=="ORD"]
tab("MEK n gaps",mek,lambda p: p.question.question.count("____"))
tab("MEK n parts in option",mek,lambda p: len(re.split(r" [–-] ",p.question.options["A"])))
tab("MEK key x ngaps",mek,lambda p:(p.question.answer,min(p.question.question.count("____"),2)))
tab("MEK sentence len",mek,lambda p: min(len(p.question.question)//150,4))
tab("LAS passage tokens (k)",las,lambda p: min(len(wb.token_ids(wb.render_prompt(p)))//500,6))
tab("LAS neg word",las,lambda p: bool(re.search(r"\b(inte|INTE|ej|minst|saknar)\b",p.question.question)))
tab("LAS option len",las,lambda p: min(sum(len(o) for o in p.question.options.values())//100,5))
tab("LAS key is longest",las,lambda p: max(p.question.options,key=lambda k:len(p.question.options[k]))==p.question.answer)
tab("LAS chosen is longest",las,lambda p: max(p.question.options,key=lambda k:len(p.question.options[k]))==p.baseline.chosen_letter)
tab("MEK key is longest",mek,lambda p: max(p.question.options,key=lambda k:len(p.question.options[k]))==p.question.answer)
tab("ORD key",ord_,lambda p:p.question.answer)
tab("ORD multiword stem",ord_,lambda p:" " in p.question.question.strip())
tab("LAS key",las,lambda p:p.question.answer)
# LAS question index within passage/paper
tab("LAS year",las,lambda p:p.id[:4]); tab("MEK year",mek,lambda p:p.id[:4]); tab("ORD year",ord_,lambda p:p.id[:4])
import random; random.seed(1)
for p in random.sample([p for p in mek if not p.baseline.is_correct],8):
    q=p.question; print("\n",p.id,"|",q.question,"\n ",q.options,"key",q.answer,"chosen",p.baseline.chosen_letter)
