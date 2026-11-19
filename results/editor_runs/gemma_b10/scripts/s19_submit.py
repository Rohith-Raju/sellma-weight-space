import torch
from bench.editor import Workbench
wb=Workbench.open("gemma_b10")
name=wb.layer_param(23,"self_attn.q_proj.weight")
edit={name:{"rows":[768],"values":torch.zeros(1,2560)}}
print(wb.edit_size(edit)["weights_touched"],wb.edit_size(edit)["weights_changed"])
expl="""NULL SUBMISSION: this edit is all zeros (one row of layers.23.self_attn.q_proj, delta = 0) and changes no weight. I found no edit I believe raises accuracy; the workbench does not accept an empty dict, so a zero delta stands in for it. 5 of 10 passes used.

FAILURE DIAGNOSED. A position prior over the answer options, not a knowledge gap that an edit could fill. Unedited practice scores: ORD .831, LAS .703, MEK .617. In the 4-option sections the first option is under-chosen (A chosen 86 times in 580 questions, key is A 145 times; accuracy when the key is A is .47 in MEK and .55 in LAS, against .60-.78 for the other letters) and B/D are over-chosen.

EVIDENCE.
Pass 1 (read, 1024 prompts: all 290 MEK and 222 LAS, each in original and in reversed option order, all layers). The deficit follows the shown position, not the content: with options reversed, accuracy for the option shown first is still .45 (MEK) / .49 (LAS). Only 59% of MEK and 73% of LAS answers pick the same option in both orders. Logit lens on the last position: the letter decision appears at layer 23 (lens agrees with the final answer in 37% of prompts at L20, 95% at L23) and is written by the attention of layer 23 (key-vs-rest logit +3.5 from attention, about 0 from the MLP); later layers amplify it to margins of 11-14 nats. Simulated for free, an additive letter correction gives at most +1 point at the final logits and at the L23/L24 lens, and the fitted correction does not transfer between exam dates.
Pass 2 (read, 48 MEK prompts cut at 21 positions each, all layers), with the attention logits of layer 23 recomputed from the weights. Head 3 of layer 23 (head 7 more weakly) is the read-out: its OV writes the letter of the token it attends to (option label, newline after the option, letter in the instruction) and its strongest option is the final answer in 88-94% of prompts. Its query has a content-independent prior: +2.5 logits to the double newline after the last option, +1.3 to the instruction letter D, -0.8 to the label A, against a match signal of 2.4-3.8 logits.

EDITS TESTED (not submitted). A rank-1 change to the q_proj rows of heads 3 and 7 of layer 23 (rows 768-1023 and 1792-2047, 1.25M weights touched, 0.029% of the model, relative norm 0.3%): subtract from the mean answer-position query its component in the span of the centred mean keys of the four option positions, gated by the mean answer-position input with the generic residual mean projected out. No gradients; built from residuals of passes 1-2 and the weights. In the free attention model on held-out questions it flattens the prior and keeps the match signal.
Pass 3 (answer, 290 MEK + 222 LAS, full strength): 336 -> 332 correct of 512. D chosen 148 -> 108, B 146 -> 173; 16 gained, 20 lost.
Pass 4 (answer, same questions, half strength): 336 -> 339; 11 gained, 8 lost.
Pass 5 (answer, 290 ORD + 97 MEK reversed + 111 LAS reversed, half strength): 375 -> 374; ORD 241 -> 240.

CONCLUSION AND EXPECTATION. The prior is real and can be moved by a small edit, but moving it is accuracy-neutral: it only decides the questions on which the model has no usable evidence, and the answers that flip are right about one time in three before and after. That is a redistribution of answer letters, which is exactly the kind of edit that would show no improvement on unseen papers. The remaining errors I inspected (rare words in ORD, wrong fillers in MEK) are confident, with median margins above 10 nats, and look like missing knowledge. I expect this zero edit to score identically to the unedited model, before and after 4-bit compression."""
rec=wb.submit(edit,expl)
print(rec["passes_used"],rec["size"]["weights_touched"])
