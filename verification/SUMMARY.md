# Verification summary (2026-11-19)

Nine independent verifiers (subagents on Opus and Fable) recomputed every claim in `CLAIMS.md` from the raw
saved data with their own code; one re-scored random samples on the GPU from scratch with its own scorer.
Reports: `report_G1.md` … `report_G8.md`, `report_R.md`; scripts in `scripts/`.

## Result

No claim was reversed. About 120 numbers were checked; all but one recompute exactly. The independent GPU
re-run matched every saved answer (40 standard-run questions, 30 × 4 rotations unedited and with editor 3's
change, 20 × 4 Belebele English; probabilities within 1e-8).

## Corrections to carry into the report

| Where | Was | Correct |
|---|---|---|
| G2.4 sign test | 123 vs 9, p ≈ 1e-26 | **124 vs 9, p ≈ 5e-27** (123 was the random-control run) |
| G3.5 perplexity | "did not rise for any edit" | "no meaningful rise" (4 Qwen edits +0.0014 at most, within controls' spread) |
| G5.4 editor 1 quote | A chosen 86 of 580 | 93 of 580 (86 is of the editor's 512-question pass set); conclusion unchanged |
| G5 editor 3 note | 21 fixed, 17 broken | 18 fixed, 14 broken (net +4 unchanged) |
| G5.5 / G6 trial_b50_k | 0.015% of the model, 655,360 weights | 655,360 touched (0.015%); **366,231 changed (0.0085%)** |
| G8.1g random control on maths | within 0.2 points | within one question per position (max 0.25) |
| G8.2d Belebele sv mean | 82.3 | 82.35 (rounds to 82.4) |

## Wording and interpretation to tighten

1. **Loop-rate rule is too noisy to use as evidence.** 50 prompts, 2 points per prompt; random controls also
   exceed the +2-point limit (+4 to +8). Qwen's loop rate is saturated (88% / 100%). Treat loop-rate "damage"
   flags as uninformative at this sample size; perplexity is the usable damage check.
2. **Full vs 4-bit perplexity are not comparable** (different computation); never read "4-bit is lower" as better.
3. **Held-out evidence for editor 3's change** is hidden 27 gained / 0 lost at position A and test 11 / 0; the dev
   figure 83 / 0 includes papers the editor practised on. The gain at A is partly paid for at B/C/D (dev: B 4/26,
   C 5/22); net over all positions +1 point.
4. **Edit vs option averaging vs prompt note are statistically indistinguishable on overall score** (held-out
   differences p ≥ 0.15). What is proven is the position effect, not which method scores best.
5. **Group edit:** step 50 beat the unedited model only on the tie-break (mean probability 0.72602 vs 0.72557);
   validation accuracy never exceeded 68.0%. Hidden figures in results.md cover all 480 questions incl. ORD.
6. **Gemma's loop rate rose with every single-question edit** (10% → 14–22%) and with the group change (10% → 22%),
   but given point 1 this is weak evidence of harm.
7. **Option-A habit at 4-bit:** stronger pooled over LÄS+MEK (key=A paired 24 worse / 7 better, p = 0.003);
   per-section evidence is thin. Compression also shifts picks toward C.
8. **"A is the only penalised letter"** holds for the reading tests; the Högskoleprovet sets are LÄS+MEK.

## Provenance notes
- `analysis/other_exams_v1_first_run.py` is the script version that produced `unedited.json`, `editor3.json`,
  `random.json` (identical to `other_exams.py` except that it did not record chosen letters).
- Edited 4-bit GGUFs were deleted after scoring; their checksums survive only in `run.json`.
- 9 large activation files (editors' read passes) stay on the GPU disk; listed in `gpu_results/LARGE_FILES_LEFT_ON_GPU.txt`.
