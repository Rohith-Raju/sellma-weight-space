# Report G3: single-question rank-1 edits (archive/single_question_edits)

Verifier: independent subagent (Opus), 2026-11-19. Recomputed headline accuracy (LAS+MEK) from answers.jsonl,
scored against exams/questions.jsonl. Scripts: scripts/G3_inventory.py, scripts/G3_claims.py.

## Completeness
- Gemma: all 10 cases have search.json; all full-precision runs of the 5 fixed edits finished. 2021-03-13 MEK 28
  targeted_q4_test cut off at 103/180 (no claim uses it).
- Qwen: 7 of 10 reached; all runs finished except 2014-04-05 MEK 24 control_test (all 180 answers present,
  cut off in the damage step; no finished_at, no damage.json); its control_dev_subset never started.
- Baselines: all finished.

| # | Claim | Claimed | Recomputed | Verdict |
|---|---|---|---|---|
| 1a | Gemma 10 cases / 5 fixed / 5 not | 10/5/5 | 10/5/5 (unfixed failed at every size 1%–100%) | CONFIRMED |
| 1b–f | Gemma fixed edits (size, test headline) | 5%/66.7, 5%/65.0, 1%/70.0, 10%/71.7, 1%/68.3 | identical (80, 78, 84, 86, 82 of 120) | CONFIRMED |
| 1g | Gemma controls | 70.0 | 70.0 x5 | CONFIRMED |
| 1h | Gemma baseline | 70.0 | 70.0 | CONFIRMED |
| 2a | Qwen 7 reached, all fixed | 7 | 7 | CONFIRMED |
| 2b | Qwen edited test headlines | 92.5, 91.7, 93.3, 93.3, 89.2, 83.3, 92.5 | identical | CONFIRMED |
| 2c | Qwen controls | 93.3 | 93.3 x7 (one is the cut-off run, answers complete) | CONFIRMED |
| 2d | Qwen baseline | 93.3 | 93.3 | CONFIRMED |
| 3 | Average change | Gemma -1.7, Qwen -2.5 | -1.67, -2.50 (Qwen without the 20% edit: -1.5) | CONFIRMED |
| 4 | Letter shift | Qwen A 29->45%; Gemma D 24->38% | 29.2->45.0; 24.2->38.3 | CONFIRMED |
| 5 | Perplexity did not rise for any edit | no rise | Gemma all 5 lower; Qwen 4 of 7 higher by at most +0.0014, within controls' spread | WRONG literally; correct wording "no meaningful rise" |

## Other problems
1. Gemma's loop rate rose with every edit: 10% unedited, 14-22% edited (controls 8-16%); edited 4-bit 24% and 28%
   vs 20% unedited 4-bit. Not mentioned in the claims. Qwen's loop rate (84-94% vs 88%) is noise-level.
2. Most headline differences are within noise (0-12 questions of 120). Only Qwen's 20% edit (-12) and possibly
   Gemma 2017-04-01 (-6) stand out; the dev subset agrees (Qwen 83.5 vs 90.5; Gemma 63.0 vs 69.5).
3. Run setup checks out: one edit per run, matching search.json; controls same nominal size, fixed seed.
4. Gemma combined/ (all 5 edits together): test 68.3 (control 70.0), loop rate 24%.
