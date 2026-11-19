# G9 verification: numbers in RESEARCH_NOTE.md and report/report_full.md (2026-11-19)

Independent verifier (subagent, Opus). Every value recomputed from raw per-question files in `results/` with own code
(`verification/scripts/G9_kinds.py`, `G9_q4.py`, `G9_single.py`, `G9_older.py`). Authors' scripts read only for file formats.

Sanity checks: in `rotation_choices.json`, chosen original letter, chosen position and top probability agree in all 8,160 tries;
it matches `rotation_proof.json` on all 3,600 dev answers and `heldout_position.jsonl` on all 440 x 4 held-out answers;
4-bit shards: chosen position vs right/wrong flag, 0 mismatches.

## WRONG (to fix)
1. Report §4 "None of these differences is statistically clear (p >= 0.15)" (test papers): exact McNemar unedited vs prompt note +4/-0, p = 0.125; unedited vs averaging +9/-3, p = 0.146. (Weight edit vs the two fixes: smallest p 0.227, correct.)
2. Report §4 "For the other 79, nothing changed": 73 still "avoids option A", 6 became "unsure"; at option A, 77 of 79 picked the same wrong option. Only 42 of the 45 are still right in all four orders.
3. Note §4 caption "Only the 'avoids option A' kind shrinks": "Doesn't know" also shrinks, 58 -> 53.
4. Group edit "fixed 6 of 117, broke 4 of 363, net +2": counts all 480 questions incl. ORD. LAS+MEK only: 5 of 88 fixed, 4 of 232 broken, net +1. "Within chance" holds.

## Wording
- Report §5 "same wrong answer as before on 329 of 335": true only at option A; add "at option A".
- 9 of 12 single edits clearly worse: confirmed with 5 bootstrap seeds; Gemma 2022-10-23 MEK 26 is borderline (upper bound -0.0002; a normal interval would include 0).

## CONFIRMED (selection)
Five kinds 405->439, 124->76, 262->270, 51->62, 0->0, 58->53; 0 of 109 never-right became right in all four; 115 never-right after;
329/335, 110/110, 40/40; 0.8% (0.83% over 4 orders); Qwen 68/109; option-A pick rates 482/900->565/900, 135/2700->216/2700,
482/617->565/781 (78.1->72.3), test 69/89->80/111, 4-bit 212/269->260/356; per-position table; 83/0, 124->76, random 2/0, +1 point;
27/0, 11/0; 4-bit A 252->212, B 338->329, C 321->338, D 319->294, wrong-only-at-A 53->82, 48/0, 38/0, random 0/1, 260 vs 252;
weight sizes Gemma 2.0-11.8 M (0.047-0.274%), Qwen 4.9-24.2 M (0.051-0.251%), group 75,124,630 (1.747%), Gemma total 4,300,079,472;
test baselines 93.3/90.8/70.0/65.8; test methods 84/88/90/85 of 120; Belebele/MMLU 21/0, 12/0, 45/0, 81.8->82.35, 86.4->86.4, 40.2->39.9,
maths A picks 28.9->39.2%; damage +0.05%, +0.03%, -0.05%, 75 speeches, 40,154 words; editor passes used 5/10, 7/25, 11/50; 870 = 720+150;
SweSAT verbal averages 62.40 / 63.02 / 84.79 (arithmetic).

## CANNOT CHECK
Qwen total 9,653,104,368 (no shapes in results/); bf16 count 366,231 and the 4-bit weight check (needs GGUF files; internally consistent,
314,961/366,231 = 86.0%, projection 1.0007); all external paper numbers.
