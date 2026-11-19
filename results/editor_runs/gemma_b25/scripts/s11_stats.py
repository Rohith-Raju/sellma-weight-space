from bench.editor import Workbench
import numpy as np
from collections import Counter, defaultdict
wb = Workbench.open("gemma_b25")
P = wb.practice_questions()
def tab(name,items,f):
    d=defaultdict(list)
    for p in items: d[f(p)].append(p.baseline.is_correct)
    print(name,{k:(round(np.mean(v),2),len(v)) for k,v in sorted(d.items())})
mek=[p for p in P if p.question.section=="MEK"]; las=[p for p in P if p.question.section=="LAS"]; ord_=[p for p in P if p.question.section=="ORD"]
tab("MEK gaps",mek,lambda p:p.question.question.count("____"))
tab("MEK nwords opt A",mek,lambda p:min(4,len(p.question.options["A"].split())))
for paper in ["drill","validation"]:
    for s,it in [("MEK",mek),("LAS",las),("ORD",ord_)]:
        tab(s+" "+paper+" by key",[p for p in it if p.paper==paper],lambda p:p.question.answer)
tab("LAS ntok",las,lambda p:len(wb.token_ids(wb.render_prompt(p)))//500)
tab("LAS year",las,lambda p:p.id[:4]); tab("MEK year",mek,lambda p:p.id[:4]); tab("ORD year",ord_,lambda p:p.id[:4])
tab("LAS key is longest opt",las,lambda p:max(p.question.options,key=lambda l:len(p.question.options[l]))==p.question.answer)
tab("LAS chosen is longest opt",las,lambda p:max(p.question.options,key=lambda l:len(p.question.options[l]))==p.baseline.chosen_letter)
tab("LAS qtype inte",las,lambda p:(" inte " in p.question.question.lower()))
print(Counter(p.id.split("/")[0] for p in P))
for p in mek[:3]+[x for x in mek if not x.baseline.is_correct][:6]:
    print(p.id,p.question.question,p.question.options,"key",p.question.answer,"chose",p.baseline.chosen_letter)
