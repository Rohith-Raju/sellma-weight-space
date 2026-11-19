from bench.editor import Workbench
wb = Workbench.open("gemma_b10")
P = wb.practice_questions()
import random
random.seed(1)
for sec,n in [("ORD",50),("MEK",30)]:
    it=[p for p in P if p.question.section==sec and not p.baseline.is_correct]
    print("=====",sec,len(it))
    for p in it[:n]:
        q=p.question
        text=q.question.replace("\n"," ")[:230]
        opts=" | ".join(("*" if k==q.answer else "")+("!" if k==p.baseline.chosen_letter else "")+k+" "+v for k,v in q.options.items())
        print("-",text,"::",opts)
