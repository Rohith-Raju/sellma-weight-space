# Verification report: G4 (group edit, gemma-4b-full)

Verifier scripts: `verification/scripts/G4_verify.py` (everything), `verification/scripts/G4_q4_ci_seeds.py`
(seed sensitivity of the bootstrap interval). Run with `uv run python -I <script>` from the project root.
All numbers recomputed from `answers.jsonl`, `split.json`, `trained/training_log.jsonl`, `exams/split.json`,
`exams/questions.jsonl` and the standard runs in `gpu_results/runs/`; `results.md` and `comparisons/` were only
read to compare. Nothing existing was modified. Could not re-run the model (Mac copy only), so "the saved
answers are what the model produced" is taken as given.

## Claims

| # | Claim | Claimed | My value | Verdict | Note |
|---|---|---|---|---|---|
| 1a | Group sizes | drill 205 wrong / 515 right; validation 41 / 109; hidden 117 / 363 | 205 / 515; 41 / 109; 117 / 363 | CONFIRMED | Recounted from the dev run `20261119T060704Z_gemma-4b-full_dev` by exam date; 0 questions misfiled; 1,350 ids, all unique, equal to the dev split. |
| 1b | Papers per group | 8 hidden, 3 validation, 12 drill | 8 / 3 / 12 | CONFIRMED | The stated rule (dates sorted by sha256, first round(23/3)=8 hidden, next 3 validation) reproduces the split exactly. |
| 1c | No paper in two groups | (asked to check) | 0 shared dates; 0 shared ids; 0 of 145 reading passages and 0 identical question texts in more than one group; no overlap with test or gemma_extra dates | CONFIRMED | |
| 1d | Validation never trained on | (asked to check) | validation ∩ drill = 0 ids; hidden ∩ drill = 0; training settings record n_wrong 205, n_right 515, n_validation 150; the code refuses to start on an overlap | CONFIRMED | From ids and recorded settings; the training batches themselves are not logged, so this is not an observation of what was fed to the optimiser. |
| 2a | Stopped after 300 steps, reason no_improvement | 300, no_improvement | 300, no_improvement | CONFIRMED | Replaying the rule on the log: best at step 50, then 10 evaluations (75…300) without a better one → stop at step 300. |
| 2b | Chose step 50 | 50 | 50 | CONFIRMED | Consistent with the rule only through the tie-break, see "Other problems" 1. Saved delta really is step 50: trained/validation answers give 68/100 and mean prob 0.726022, trained/drill 4 fixed / 509 kept, identical to the step-50 log row. |
| 2c | Validation headline 68.0% at steps 0 and 50 | 68.0 / 68.0 | 68.0 / 68.0 (68 of 100) | CONFIRMED | Also 68.0 at steps 75, 100 and 175. It never exceeded the unedited model's 68.0. |
| 2d | Log row step 150: 6 fixed, 501 kept, val 67.0 | 6 / 501 / 67.0 | 6 / 501 / 67.0 | CONFIRMED | Full log (step: fixed, kept, val headline): 0: 0, 515, 68 · 25: 3, 511, 67 · 50: 4, 509, 68 · 75: 5, 504, 68 · 100: 8, 501, 68 · 125: 3, 501, 66 · 150: 6, 501, 67 · 175: 13, 499, 68 · 200: 14, 501, 67 · 225: 13, 501, 67 · 250: 8, 502, 66 · 275: 13, 498, 67 · 300: 16, 499, 67. I did not see the other rows "as reported", only this example. |
| 3a | Hidden, trained: fixes 6/117, keeps 359/363 | 6/117, 359/363 | 6/117, 359/363 (4 broken) | CONFIRMED | |
| 3b | Hidden, control: fixes 1/117, keeps 363 | 1/117, 363/363 | 1/117, 363/363 | CONFIRMED | |
| 3c | Test headline 70.0 → 70.8, control 70.0 | 70.0 / 70.8 / 70.0 | 84/120 = 70.0; 85/120 = 70.83; 84/120 = 70.0 | CONFIRMED | One MEK question wrong→right, none broken. Unedited re-run identical to the standard test run. |
| 4a | Hidden net +2 questions | +2 | +2 (6 fixed − 4 broken; 363→365 of 480, +0.42 pp) | CONFIRMED | |
| 4b | CI for hidden net includes zero | −0.8 to +1.7 pp, includes zero | −0.83 to +1.67 pp with my own paired bootstrap (stdlib RNG, 4 seeds; numpy, 20 seeds: same every time) | CONFIRMED | Exact sign test on 6 vs 4: p = 0.75. On LAS+MEK only: 232→233 of 320, CI −1.56 to +2.19. |
| 4c | Test verdict NO IMPROVEMENT | NO IMPROVEMENT | comparison.json says NO IMPROVEMENT; my headline metric change +0.0008, CI −0.0118 to +0.0172 (includes zero) | CONFIRMED | The verdict itself is produced by project code; I recomputed its main input, not the full rule. Loop rate 10% → 22% from damage.json (rule 3b failed). |
| 4d | 4-bit hidden 72.9% vs 73.3% unedited 4-bit | 72.9 / 73.3 | 350/480 = 72.92; 352/480 = 73.33 | CONFIRMED | These are over all 480 hidden questions including ORD, not headline. LAS+MEK only: 219/320 = 68.4 vs 220/320 = 68.8. |
| 4e | 4-bit hidden net −2 | −2 | −2 (3 fixed of 128, 5 broken of 352) | CONFIRMED | Interval includes zero; its ends depend on the seed (see "Other problems" 4). |

## Other problems

1. **Step 50 was chosen on a tie-break, not on validation accuracy.** Validation headline accuracy was 68.0% for
   the unedited model (step 0) and never went higher. Step 50 beat step 0 only on the second key, mean
   probability of the correct option over all 150 validation questions: 0.72602 vs 0.72557, a difference of
   0.0005. Steps 75, 100 and 175 also had 68.0%. So "chosen by the validation papers" is true by the rule, but
   validation gave no sign that any step was better than no change. The patience count restarted at step 50
   because of that tie-break; without it training would have stopped at step 250 and returned the unedited model.
2. **results.md training table shows no validation numbers.** It says "validation = (no validation field found
   in the log)" and prints n/a in every row, although every log row has `validation.headline_accuracy`. The
   report code does not read the nested field. The numbers in claim 2 are right but are not in results.md.
3. **Docstring disagrees with the run.** `src/bench/group_edit.py` module text says patience is 6; the default
   and the recorded setting are 10. The run used 10.
4. **4-bit net interval is not stable to the seed.** results.md gives −1.7 to +0.8; over 20 seeds I get
   −1.67 or −1.46 at the low end and +0.62 or +0.83 at the high end (values move in steps of 1/480). Zero is
   inside in every case. The full-precision interval (−0.83 to +1.67) did not move.
5. **Mixed denominators in the claims.** "Headline" is LAS+MEK for test (70.0 → 70.8) but the hidden figures
   (fixed/kept, net +2, 72.9 vs 73.3) are over all sections including ORD. Each number is right for its own
   denominator.
6. **Trained vs random on hidden wrong: lower CI end is exactly 0.0** (+4.3 pp, 0.0 to +8.5; I get the same).
   results.md treats this as "includes zero", which is the cautious reading; a sign test on 6 vs 1 gives p = 0.125.
7. **Damage on test.** Loop rate rises from 10% to 22% with the trained change (control 8%), on 50 prompts.
   Not part of the G4 claims, but it is one of the failed rules and was not in the claim text.
8. **Not checkable here:** that the saved answers came from the stated weights and delta (no model on the Mac),
   and which questions each training batch contained.
