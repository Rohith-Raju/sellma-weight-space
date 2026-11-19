from bench.editor import Workbench
from collections import Counter, defaultdict
import numpy as np
wb = Workbench.open("gemma_b25")
P = wb.practice_questions()
print(wb.budget, wb.passes_left)
for sec in ["ORD","LAS","MEK"]:
    for paper in ["drill","validation"]:
        it=[p for p in P if p.question.section==sec and p.paper==paper]
        print(sec,paper,len(it), "acc %.3f"%np.mean([p.baseline.is_correct for p in it]),
              "nopt",Counter(len(p.question.options) for p in it),
              "pletters %.3f"%np.mean([p.baseline.prob_on_letters for p in it]),
              "pcorrect %.3f"%np.mean([p.baseline.prob_correct for p in it]))
    it=[p for p in P if p.question.section==sec]
    print(" key",sorted(Counter(p.question.answer for p in it).items()))
    print(" chosen",sorted(Counter(p.baseline.chosen_letter for p in it).items()))
    w=[p for p in it if not p.baseline.is_correct]
    print(" wrong: chosen",sorted(Counter(p.baseline.chosen_letter for p in w).items())," key",sorted(Counter(p.question.answer for p in w).items()))
    # confidence of wrong: prob of chosen normalised, rank of correct
    mx=[max(p.baseline.letter_probs.values())/p.baseline.prob_on_letters for p in w]
    rk=[sorted(p.baseline.letter_probs, key=lambda l:-p.baseline.letter_probs[l]).index(p.question.answer) for p in w]
    print(" wrong: mean top share %.3f"%np.mean(mx)," rank of correct",sorted(Counter(rk).items()))
    mxc=[max(p.baseline.letter_probs.values())/p.baseline.prob_on_letters for p in it if p.baseline.is_correct]
    print(" right: mean top share %.3f"%np.mean(mxc))
    # by accuracy per key letter
    print(" acc by key",{l:round(np.mean([p.baseline.is_correct for p in it if p.question.answer==l]),3) for l in "ABCDE" if any(p.question.answer==l for p in it)})
print(wb.render_prompt(P[0]))
mek=[p for p in P if p.question.section=="MEK"][0]
print(wb.render_prompt(mek))
las=[p for p in P if p.question.section=="LAS"][0]
t=wb.render_prompt(las); print(t[:600]); print("....."); print(t[-900:])
print(np.mean([len(wb.token_ids(wb.render_prompt(p))) for p in P if p.question.section=="LAS"]))
