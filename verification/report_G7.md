# G7 verification report: rotation tests on Högskoleprovet

Verifier scripts (read-only on all existing files):
- `verification/scripts/G7_verify.py` — recomputes every number from `rotation_proof.json`, `heldout_position.jsonl`, `rotation_first_run_dev.jsonl`, with own exact two-sided sign tests; cross-checks against the standard runs and both split files.
- `verification/scripts/G7_ties_and_averaging.py` — exact-tie count; tests the option-averaging function (copied verbatim) against an independent reference on 20,000 simulated questions.

Run: `uv run python -I verification/scripts/G7_verify.py` (and the same for the second script).

**Result: every G7 number is CONFIRMED. No bug found that biases the results. Three caveats on interpretation are listed below the table.**

## Verdict table

| # | Claim | Claimed | My value | Verdict | Note |
|---|---|---|---|---|---|
| 1 | Three unedited repeats identical | 0 of 3,600 differ | 0 of 3,600; margins also identical to the last bit | CONFIRMED | unedited_1 vs unedited_2 and vs unedited_after; also 0 flips vs the first saved run |
| 1 | Unedited right in all 4 / in none | 405 / 109 | 405 / 109 | CONFIRMED | |
| 1 | Unedited by position A/B/C/D | 53.6 / 75.9 / 72.0 / 70.8 | 53.6 / 75.9 / 72.0 / 70.8 (482, 683, 648, 637 of 900) | CONFIRMED | |
| 1 | Editor 3 by position | 62.8 / 73.4 / 70.1 / 69.9 | 62.8 / 73.4 / 70.1 / 69.9 (565, 661, 631, 629) | CONFIRMED | |
| 1 | Editor 3 all-4 / never | 439 / 115 | 439 / 115 | CONFIRMED | |
| 1 | Position-dependent | 386 → 346 | 386 → 346 | CONFIRMED | |
| 1 | Wrong only at A | 124 → 76 | 124 → 76 | CONFIRMED | |
| 1 | Right only at A | 9 → 22 | 9 → 22 | CONFIRMED | |
| 1 | Right-at-A gained/lost, p | 83 / 0, p≈2e-25 | 83 / 0, p = 2.07e-25 | CONFIRMED | exact two-sided sign test |
| 1 | All-4 gained/lost, p | 43 / 9, p≈2e-6 | 43 / 9, p = 2.04e-6 | CONFIRMED | |
| 1 | Random control at A | 2 / 0 | 2 / 0 (p = 0.5) | CONFIRMED | control flips 6 of 3,600 answers in total; editor 3 flips 156 |
| 2 | Hidden: original, note, averaging, editor 3, editor 3 + averaging | 232, 230, 238, 236, 238 (of 320) | 232, 230, 238, 236, 238 | CONFIRMED | original-order values match the independent standard runs question by question (0 mismatches) |
| 2 | Test: same five | 84, 88, 90, 85, 90 (of 120) | 84, 88, 90, 85, 90 | CONFIRMED | same cross-check, 0 mismatches |
| 2 | A-right hidden | 57.2 → 65.6 | 57.2 → 65.6 (183 → 210) | CONFIRMED | |
| 2 | A-right test | 57.5 → 66.7 | 57.5 → 66.7 (69 → 80) | CONFIRMED | |
| 2 | Wrong only at A | 39 → 23 hidden, 14 → 8 test | 39 → 23, 14 → 8 | CONFIRMED | |
| 2 | Right in none | 39 → 38 hidden, 12 → 15 test | 39 → 38, 12 → 15 | CONFIRMED | |
| 2 | Right-at-A gained/lost | 27 / 0 hidden, 11 / 0 test | 27 / 0 (p = 1.5e-8), 11 / 0 (p = 0.00098) | CONFIRMED | |
| 3 | Averaging sums each option text's probability over the 4 rotations | as described | code does this; 0 disagreements with an independent reference on 20,000 simulated questions | CONFIRMED (code and counts); raw recomputation CANNOT CHECK | per-rotation probabilities were not saved, so the 238 / 90 counts can only be recounted from saved true/false flags, not rebuilt from probabilities |

## Caveats on interpretation (numbers are right; the reading needs care)

1. **The "hidden" held-out result is not independent of the dev result.** The 320 hidden questions are a subset of the 900 dev questions in claim 1. The hidden figures in claim 2 (A 57.2 → 65.6, gained 27 / lost 0) are exactly the hidden part of the dev figures. Only the 120 test questions are separate from the dev table.
2. **The dev result in claim 1 includes the papers editor 3 practised on.** Of the 900 dev questions, 480 are drill and 100 validation, which the editor workbench allows the editor to use (`PRACTICE_PAPERS = ("drill", "validation")`); 320 are hidden. Split of the +83 at A: drill 45 / 0 (51.7 → 61.0), validation 11 / 0 (51.0 → 62.0), hidden 27 / 0 (57.2 → 65.6). The effect is the same size on papers the editor never saw, so the claim holds, but the truly unseen evidence is hidden 27 / 0 plus test 11 / 0.
3. **The gain at A is paid for at B, C and D, and total accuracy barely moves.** Not stated in G7 but visible in its own numbers. Dev, editor 3 vs unedited: B gained 4 / lost 26 (p = 6e-5), C 5 / 22 (p = 0.0015), D 4 / 12 (p = 0.08). Sum of right answers over all four positions: 2,450 → 2,486 of 3,600 (+1.0 point). In original order on held-out: hidden 232 → 236 (gained 8, lost 4, p = 0.39), test 84 → 85 (3 / 2, p = 1). Option averaging vs original: hidden gained 20 / lost 14 (p = 0.39), test 9 / 3 (p = 0.15). Prompt note vs original: hidden 11 / 13, test 4 / 0 (p = 0.125). None of the held-out original-order differences between the five conditions is statistically significant.

## Code review

Scripts reviewed: `gpu_results/analysis/rotations.py`, `rotation_proof.py`, `heldout_position.py` (plus `proof_check.py`, `rotations_edit.py`, and `src/bench/prompts.py`, `src/bench/backends/hf.py`, `src/bench/editor/evaluate.py` as used by them).

**Rotation logic — correct.** Options are rotated cyclically by 0–3; the new key is `"ABCD"[rotated.index(correct_text)]`. Each question gets the correct answer exactly once at each of A, B, C, D (verified: every saved row has all four letters). Shift 0 is the original order; I checked that the rotation result at the original key letter equals the standard run's result for all 900 dev, 320 hidden and 120 test questions, unedited and edited: 0 mismatches. This confirms the rotation runs use the same prompt path and weights as the standard runs.

**Duplicate option texts — a latent bug, not triggered.** `rotated.index(correct)` takes the first match, and the averaging dictionary is keyed by option text, so two identical option texts would mis-track the key and merge two options. No question in dev, hidden or test LAS+MEK has duplicate option texts (0, also 0 after stripping whitespace and lower-casing), so no result is affected. `rotation_proof.py` counts and prints this (0); `heldout_position.py` and `rotations.py` do not check.

**Option averaging (claim 3) — correct.** Each rotation's four letter probabilities are normalised to sum to 1, appended to the list for the option text at that letter, and the text with the largest sum over the 4 rotations is chosen and compared with the correct text. Each text appears once at each letter. Ties go to the earlier original option (no effect seen). Consistency in the saved data: no question right in all 4 rotations is wrong under averaging, and none right in no rotation is right under averaging.

**Prompt note — correct, one design point.** The note is prepended to the single user message, before the instruction, on a copy of the messages (the original prompt is not mutated). It is scored in the original order against the original key. It was run only on the unedited model, as claimed. The note says the options are in random order, which is not literally true of the original order; that is a wording choice, not a bug.

**Edits applied and restored — confirmed.** `applied()` saves copies of the touched entries and writes them back in a `finally` block. Evidence it was applied: 156 of 3,600 answers change under editor 3; the edited original-order answers match the separately produced `experiments/editor/trial_b50_k/edited/{hidden,test}` runs exactly. Evidence it was restored: `unedited_after` (run after both the edit and the random control) is identical to `unedited_1` in every answer and every margin. In `heldout_position.py` the unedited condition runs before the edit, so no restore is needed there. The edit used is `editor_runs/trial_b50_k/final_edit` (layer-23 k_proj rows, "Editor 3 trial (pass 9)"), consistent with the label "editor 3".

**Random control — correct.** Same tensor, same nonzero pattern, Gaussian values scaled to the same norm, fixed seed.

**Held-out sets — correct.** The hidden rows are exactly the LAS+MEK questions of `hidden_wrong ∪ hidden_right` in the group split (320 = 160 LAS + 160 MEK, 8 exams, 16 passes); the test rows are exactly the LAS+MEK questions of the test split in `exams/split.json` (120, exams 2025-10-19, 2026-04-18, 2026-10-18). No overlap between hidden and drill/validation questions, papers or exam dates; no overlap between test and any group-split question. Keys in the saved rows match `questions.jsonl`. No duplicate rows.

**Minor points, no effect on conclusions.**
- Tie-breaking: `max("ABCD", key=...)` picks the earliest letter on an exact tie. There are 5 exact ties among the 3,600 unedited question-positions (1 counted right at A, 2 right at B, 2 wrong at D), 1 under editor 3, 9 under the random control. At most 1 of the 482 unedited A-right answers depends on this.
- `rotation_proof.py` uses raw letter probabilities, `heldout_position.py` normalised ones; the argmax is the same.
- `heldout_position.py` holds all records in memory and writes the file only at the end; a crash would have lost everything, but the run completed (880 rows present: 2 conditions × 440 questions).
- The sign-test function in the producing scripts (`mcnemar`, `p2`) is a correct exact two-sided binomial test; my independent implementation gives the same p-values.
- `logs/rotation_proof.json` and `logs/rotation_test.jsonl` are byte-identical to the copies in `experiments/analysis/`.
