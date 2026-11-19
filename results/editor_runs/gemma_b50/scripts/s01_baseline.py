from bench.editor import Workbench
from collections import Counter, defaultdict
import numpy as np
wb = Workbench.open("gemma_b50")
P = wb.practice_questions()
print(len(P), wb.budget, wb.passes_left)
for paper in ["drill","validation"]:
    for sec in ["ORD","LAS","MEK"]:
        it=[p for p in P if p.paper==paper and p.question.section==sec]
        print(paper, sec, len(it), sum(p.baseline.is_correct for p in it)/max(1,len(it)))
for sec in ["ORD","LAS","MEK"]:
    it=[p for p in P if p.question.section==sec]
    print(sec,"nopts",Counter(len(p.question.options) for p in it))
    print(" key",sorted(Counter(p.question.answer for p in it).items()))
    print(" chosen",sorted(Counter(p.baseline.chosen_letter for p in it).items()))
    w=[p for p in it if not p.baseline.is_correct]
    print(" wrong: key",sorted(Counter(p.question.answer for p in w).items())," chosen",sorted(Counter(p.baseline.chosen_letter for p in w).items()))
    print(" prob_on_letters mean",np.mean([p.baseline.prob_on_letters for p in it]), "min",np.min([p.baseline.prob_on_letters for p in it]))
    pc=[p.baseline.prob_correct for p in w]
    print(" wrong: prob_correct quantiles",np.quantile(pc,[.1,.25,.5,.75,.9]))
    mx=[max(p.baseline.letter_probs.values())/p.baseline.prob_on_letters for p in w]
    print(" wrong: max prob quantiles",np.quantile(mx,[.1,.25,.5,.75,.9]))
    mx=[max(p.baseline.letter_probs.values())/p.baseline.prob_on_letters for p in it if p.baseline.is_correct]
    print(" right: max prob quantiles",np.quantile(mx,[.1,.25,.5,.75,.9]))
    # rank of correct among wrong
    rk=Counter()
    for p in w:
        s=sorted(p.baseline.letter_probs.items(), key=lambda x:-x[1]); rk[[k for k,_ in s].index(p.question.answer)+1]+=1
    print(" wrong: rank of key",sorted(rk.items()))
print("=====PROMPT ORD"); print(wb.render_prompt(P[0]))
m=[p for p in P if p.question.section=="MEK"][0]
print("=====PROMPT MEK"); print(wb.render_prompt(m))
l=[p for p in P if p.question.section=="LAS"][0]
t=wb.render_prompt(l); print("=====PROMPT LAS"); print(t[:600]); print("....."); print(t[-900:])
print(len(wb.token_ids(t)))
