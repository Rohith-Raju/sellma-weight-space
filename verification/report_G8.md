# Report G8: other exams (Belebele Swedish/English, MMLU maths)

Verifier: independent subagent (Opus), 2026-11-19. Scripts: scripts/G8_recompute.py, scripts/G8_datasets.py.

| # | Claim | Claimed | Recomputed | Verdict |
|---|---|---|---|---|
| 1a-c | Unedited by position (sv, en, maths) and all-4 | 76.4/82.4/83.8/84.6 (345); 83.4/88.4/86.4/87.2 (394); 45.8/46.0/34.5/34.5 (46) | same | CONFIRMED |
| 1d-f | Editor 3 by position | 80.6/81.2/83.8/83.8 (358); 85.8/87.8/85.8/86.0 (392); 57.0/40.0/32.5/30.2 (47) | same | CONFIRMED |
| 1g | Random = unedited (maths within 0.2) | equal / within 0.2 | sv, en identical; maths 3 of 1,600 cells differ, max 0.25 points | CONFIRMED (wording: within one question per position) |
| 2a-c | Gains/losses per position | as claimed | same (sv A p=9.5e-7) | CONFIRMED |
| 2d | Means over positions | 81.8->82.3, 86.4->86.4, 40.2->39.9 | 81.80->82.35, 86.35->86.35, 40.19->39.94 | CONFIRMED (82.35 rounds to 82.4) |
| 3a-c | Letter picks | as claimed | same (counts 444/500/511/545 -> 482/482/501/535 etc.) | CONFIRMED |
| 4a | sv and en same 500 questions | yes | same ids, order, keys 500/500 | CONFIRMED |
| 4b | Revisions | 7899cdf…, c30699e… | recorded, resolvable, reproduce the saved id lists | CONFIRMED |
| 4c | Licences | CC BY-SA 4.0, MIT | API: cc-by-sa-4.0, mit | CONFIRMED |
| 5a | wrong-only-at-A / right-only-at-A | 124/9, 39/3, 14/2, 31/3, 15/2, 24/61 | same | CONFIRMED |
| 5b | A the only penalised letter on reading tests | yes | only A significant (HP sets are LÄS+MEK, not reading only) | CONFIRMED (caveat) |
| 5c | Maths favours A and B | yes | A 24/61, B 5/33, both significant | CONFIRMED |

Consistency: *_with_choices runs match the first runs in every cell (unedited, editor 3).

## Code review (other_exams.py): correct answer tracking, scoring, prompts. Notes (no claim changes):
1. Provenance: the first-run files (unedited.json, editor3.json, random.json) came from the first version of the
   script, now saved as analysis/other_exams_v1_first_run.py (identical except for recording chosen letters).
2. English prompts are not whitespace-stripped (12 questions, 55 option sets have stray whitespace).
3. One Belebele item (German_judge…#1) has duplicate options (4, 2, 5, 2); position tracking counts the identical
   copy as wrong; C and D understated by 0.2 points each in all conditions; A figures unaffected.
4. Source passages contain "" artifacts from the dataset; identical across conditions.
5. Docstring path says work/ (now analysis/); output path hard-coded.
