# G2 verification report — option-A habit in the unedited model

Verifier: independent recomputation, 2026-11-19. Own code only:
`verification/scripts/G2_verify.py` (all claims) and `verification/scripts/G2_rotation_counts.py` (side check on the source of "123 vs 9").
Run with `uv run python -I verification/scripts/G2_verify.py`.

Raw inputs: `gpu_results/runs/20261119T060704Z_gemma-4b-full_dev/answers.jsonl` and
`gpu_results/runs/20261119T061105Z_gemma-4b-q4_dev/answers.jsonl` (1,350 rows each), cross-checked against
`exams/questions.jsonl` and `exams/split.json`.

Integrity checks (both runs): 1,350 unique ids, identical to the dev split; answer key and section equal
`questions.jsonl` on all 1,350; `chosen_letter` equals the argmax of `letter_probs` on all 1,350; `is_correct`
consistent on all 1,350; 450 ORD / 450 LAS / 450 MEK.

## Verdict table

| # | Claim | Claimed | Recomputed | Verdict | Note |
|---|---|---|---|---|---|
| 1a | Full, dev LAS+MEK: key counts A/B/C/D | 219 / 221 / 230 / 230 | 219 / 221 / 230 / 230 (n=900) | CONFIRMED | |
| 1b | Full: letters picked A/B/C/D | 140 / 252 / 253 / 255 | 140 / 252 / 253 / 255 | CONFIRMED | |
| 1c | Full: accuracy by key A/B/C/D | 51 / 79 / 73 / 70 % | 51.14 (112/219) / 78.73 (174/221) / 72.61 (167/230) / 70.43 (162/230) % | CONFIRMED | |
| 1d | Full: key=A and wrong, picked C/D/B | 40 / 37 / 30 | 40 / 37 / 30 (107 wrong) | CONFIRMED | |
| 2a | MEK key=A right, full → 4-bit | 47% → 39% | 47.01% (55/117) → 39.32% (46/117) | CONFIRMED | |
| 2b | MEK key=B/C/D right | 71% → 72% | 71.47% (238/333) → 72.37% (241/333) | CONFIRMED | |
| 2c | MEK A picked; A is key | 71 → 53; 117× | 71 → 53; 117 | CONFIRMED | |
| 2d | LAS key=A right | 56% → 48% | 55.88% (57/102) → 48.04% (49/102) | CONFIRMED | |
| 2e | LAS key=B/C/D right | 76% → 73% | 76.15% (265/348) → 73.28% (255/348) | CONFIRMED | |
| 2f | LAS A picked; A is key | 69 → 65; 102× | 69 → 65; 102 | CONFIRMED | |
| 3 | Full and 4-bit same letter, dev | 1,156 of 1,350 | 1,156 of 1,350 (ORD 408/450, LAS 385/450, MEK 363/450) | CONFIRMED | 85.6% overall; 83.1% on LAS+MEK (748/900) |
| 4a | Sign test 22 vs 4 | p≈0.00053 | 0.000533521 | CONFIRMED | exact two-sided binomial |
| 4b | Sign test 22 vs 2 | p≈3.6e-5 | 3.5882e-05 | CONFIRMED | |
| 4c | Accuracy key=A vs others, z | ≈6.29 | 6.2874 (pooled); 6.02 unpooled | CONFIRMED | 112/219 vs 503/681; pooled SE is what was used |
| 4d | Sign test 123 vs 9 | p≈1e-26 | 1.00419e-26 | CONFIRMED (arithmetic) | counts do not match the unedited run: see problem 1 |

## Wording check

- "Option-A habit" (claim 1). Supported, but the precise description is an *avoidance* of A: A is the key 219
  times and is picked only 140 times (B/C/D are each picked 252–255 times against 221–230 keys). Accuracy when
  the key is A is 51.1% vs 73.9% for the other keys (z = 6.29). The wording must not be read as "prefers A".
- "Habit stronger at 4-bit" (claim 2). Direction holds in both sections, and pooled over LAS+MEK: key=A accuracy
  51.1% → 43.4% (112 → 95 of 219), key=B/C/D 73.9% → 72.8% (503 → 496 of 681), A picked 140 → 118.
  Paired on the 219 key=A questions: full right / 4-bit wrong 24, the reverse 7, exact sign p = 0.0033; on the
  681 other questions 43 vs 36, p = 0.50. So the extra loss at 4-bit is concentrated on key=A questions and is
  significant pooled. Per section it is weaker evidence than the percentages suggest: LAS "A picked 69 → 65" is a
  4-question change, and each section alone has only 102–117 key=A questions. Claim it pooled, not per section.
- The 4-bit model does not simply shift uniformly away from A: its picks become C-heavy (C 253 → 301, D 255 → 219,
  B 252 → 262), so quantisation also changes the B/C/D balance, not only A. "Stronger A habit" is true but incomplete.
- ORD (5 options, not in headline) shows no A penalty at full precision (key=A 81.7% vs others 82.9%), so the habit
  is specific to LAS/MEK; consistent with the claim being scoped to LAS+MEK.

## Other problems

1. **"123 vs 9" is not the unedited model's count.** In `gpu_results/experiments/analysis/rotation_proof.json`
   the unedited runs (unedited_1, unedited_2, unedited_after) give wrong-only-at-A 124 vs right-only-at-A 9,
   which is what G7.1 and G8.5 state. 123 vs 9 is the *random control* run. For 124 vs 9 the exact p is
   5.4e-27, not 1e-26. The conclusion is unchanged, but G2.4 should read "124 vs 9 → p≈5e-27" (or be labelled as
   the random-control figure).
2. **Claim 4's sign-test counts are not G2 data.** 22 vs 4, 22 vs 2 and 123 vs 9 come from rotation experiments
   (G5.4, G7), not from the original-order dev runs. Only the p-value arithmetic was checked here; the counts
   themselves are for the G5/G7 verifiers (I only spot-checked 124/9 and 123/9 as above).
3. **z-test caveat.** z = 6.29 treats the 900 questions as independent; LAS questions share passages, so the true
   uncertainty is somewhat larger. The effect is far too large for this to change the conclusion.
4. No discrepancies found between the raw answers files and the claimed counts; nothing in claim groups 1–3 is wrong.
