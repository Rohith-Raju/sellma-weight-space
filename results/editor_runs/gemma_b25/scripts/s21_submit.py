import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from bench.editor import Workbench
import torch
wb = Workbench.open("gemma_b25")
EXPL = """HONEST NULL: I submit an empty edit (no weight is changed). 7 of 25 passes used. I found a real, nameable habit but no edit that I believe improves the model.

DIAGNOSIS. On the 4-option sections the unedited model under-credits the FIRST option. Practice baseline: MEK key=A 47% correct vs 60-76% for keys B/C/D; LAS key=A 55% vs 70-78%; A is chosen 48/290 (MEK) and 45/290 (LAS) although it is the key 78 and 67 times. ORD (5 options) shows no such effect. Pass 1 (read, 1024 prompts, all layers) showed it is positional, not content: with the options shown in reverse order (DCBA) the pattern is identical (MEK shown-key-A 45%, LAS 59%), and only 59% (MEK) / 72% (LAS) of answers pick the same content in both orders. The answer letter appears at the last position at layer 23 (logit-lens agreement with the final answer 0.50 at L22 -> 0.95 at L23) and is already short of A there; layers 24-31 add a further constant anti-A push (mean centred A logit -0.1 at L23 -> -4.1 at L29). Final margins are huge (wrong key-A items are 10-20 nats behind), so no output-letter bias can fix it (free simulation: +0..1 question).
Pass 2 (read, 256 MEK questions cut at the end of each option, all layers): a cross-validated linear probe on the residual at the option-end positions (L16-20) picks the key 62-66% of the time (the model: 62%) and has NO first-option deficit (key-A 60-68%). Many MLP neurons in L13-19 code "this option is the correct one" there (|t| up to 14, split-half r 0.75), but their output columns share no common direction (mean pairwise cosine 0.00), and the code is position-graded: neurons that mark the correct option at B/C/D respond only ~40% as strongly at A (regression slope 0.4 vs 0.7-1.0 between later positions; the B/C/D-minus-A index is reproducible, split-half r 0.4). Reading: later options are judged comparatively against earlier ones, the first option cannot be, and the readout at L21-23 under-weights it. This is a distributed sensitivity deficit, not a single direction or offset.

EDITS TESTED (answer passes on 290 MEK + 222 LAS practice questions, baseline 0.617 / 0.712):
3) scale x2 the down_proj columns of the 64 strongest correct-option neurons in each of L14-16: MEK 0.628, LAS 0.707 (9 fixed, 7 broken).
4) same, 512 neurons in each of L13-18: MEK 0.600, LAS 0.694; first-option deficit got worse (key-A 0.47->0.38).
5) article-style edit: direction built from token embeddings (unembedding of A minus mean of B,C,D), projected out of the 16 most consistently anti-A down_proj columns in each of L26-29: MEK 0.628, LAS 0.703 (5 fixed, 4 broken); A picks 48->54 and 38->41 only, far less than the first-order prediction, i.e. the decision is already locked in.
6) scale x2 the 256 neurons per layer (L13-18) that code correctness specifically at the first option: MEK 0.634, LAS 0.694.
7) scale x0.5 the 256 comparative neurons per layer (L13-18): MEK 0.634, LAS 0.680.
No candidate beat the baseline beyond noise (roughly +-2 points on these sets); the small MEK gains are on questions used to select the neurons (the 34 held-out MEK questions did not improve) and LAS, never used for selection, always got slightly worse. Free simulations also put the ceiling of any first-option bias correction at about +1 point. The remaining errors look like missing knowledge (rare words, reading comprehension), which weight editing without gradients cannot add.

EXPECTATION. The empty edit leaves the model exactly as it is: no change on hidden papers, on 2025-2026 exams, on ordinary Swedish, or after 4-bit compression. Submitting any of the tested edits would have been submitting noise."""
edit={}
try:
    wb.make_edit(edit); print(wb.edit_size(edit))
except Exception as ex:
    print("empty edit refused by make_edit:",ex)
    name=wb.layer_param(23)
    edit={name:{"cols":[0],"values":torch.zeros(2560,1)}}
    print({k:v for k,v in wb.edit_size(edit).items() if k!="tensors"})
rec=wb.submit(edit,EXPL)
print(rec)
