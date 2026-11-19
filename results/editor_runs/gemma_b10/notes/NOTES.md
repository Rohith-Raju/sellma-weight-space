# gemma_b10 notes (5 of 10 passes used; submitted a null edit)
Baseline (free): ORD .831, LAS .703, MEK .617. Key=A accuracy only .47 (MEK) / .55 (LAS); A chosen 45-48 of 290, D and B over-chosen.
Pass 1 read  (1024 full prompts, MEK+LAS, original + reversed order, all layers): bias follows the shown position, not content
  (key shown first: .45-.47 MEK in both orders). Only 59% (MEK) / 73% (LAS) of answers survive reversing the options.
  Logit lens: letter decision appears at layer 23, written by attention (key-vs-rest +3.5), later layers only amplify.
  Additive letter-bias corrections simulated free at final logits and at L23/L24 lens: <= +1 pt, not stable across exam dates.
Pass 2 read  (48 MEK prompts cut at 21 positions): head 23.3 (and 23.7) copies the letter from option-label tokens, the newline after
  each option and the letters in the instruction; its attention argmax = final answer in 88-94%. Its query carries a content-independent
  prior: +2.5 logits to the "\n\n" after the last option, +1.3 to instruction " D", -0.8 to label "A" (match signal is 2.4-3.8).
Pass 3 answer (512 MEK+LAS, rank-1 q_proj edit on heads 3,7 removing that prior): 336 -> 332. D chosen 148 -> 108, B 146 -> 173.
Pass 4 answer (same questions, half strength): 336 -> 339 (11 gained, 8 lost).
Pass 5 answer (ORD 290 + reversed MEK 97 + reversed LAS 111, half strength): 375 -> 374.
Conclusion: the position prior is real and causally editable but accuracy-neutral: it only decides questions where the model has no
usable evidence (flipped answers are right about 1/3 of the time before and after). No edit I believe in -> null submission.
