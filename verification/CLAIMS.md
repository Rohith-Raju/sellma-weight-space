# Claims to verify

Every number reported to the owner on 2026-11-19. Each verifier recomputes its group **from the raw saved
data** (not from summaries written by the same code, where avoidable) and marks each claim CONFIRMED,
WRONG (with the correct value), or CANNOT CHECK (with the reason).

Data locations. Mac copy: `<repo>/gpu_results/` (= `/root/sellma/` on the GPU
machine, minus 9 large activation files). Questions and split: `<repo>/exams/`
(questions.jsonl, split.json, raw/ PDFs). GPU machine: `ssh sellma-gpu`, project `/root/sellma`.
Headline = LÄS (LAS) + MEK only. "dev" = the 23 older exam papers (1,350 questions; 900 of them LAS+MEK).

## G1. Data, split, standard runs (unedited models)
1. 28 exams, 57 verbal passes found, 55 kept, 1,650 questions; set aside: 2018-04-14 pass 5 (two booklet versions).
2. Every kept pass has 10 ORD, 10 LAS, 10 MEK; ORD has 5 options, LAS and MEK 4.
3. Split: test = 2025-10-19, 2026-04-18, 2026-10-18 (180 questions: 60/60/60); gemma_extra = 2024-10-20, 2025-04-05 (120); dev = the other 23 (1,350).
4. Answer keys match the official PDFs (spot-check at least 30 questions over 3+ exams incl. one test exam).
5. Standard runs (headline accuracy %, ORD/LAS/MEK %, perplexity, loop rate %):
   - qwen-9b-full dev 88.4 | 89.6/91.1/85.8 | 7.869 | 88; test 93.3 | 88.3/95.0/91.7
   - qwen-9b-q4 dev 87.1 | 88.0/90.0/84.2 | 7.768 | 100; test 90.8 | 85.0/91.7/90.0
   - gemma-4b-full dev 68.3 | 82.7/71.6/65.1 | 20.83 | 10; test 70.0 | 88.3/68.3/71.7; gemma_extra 72.5 | 87.5/67.5/77.5
   - gemma-4b-q4 dev 65.7 | 80.4/67.6/63.8 | 15.10 | 20; test 65.8 | 90.0/63.3/68.3; gemma_extra 60.0 | 85.0/57.5/62.5
6. Every standard run used the right weights: qwen full shards db6f444b43…, 31c7d7e2dd…, 7ec36ba3a4…, b62b0c4cd7…; gemma full eb5fd5e97d…, fdde0e5aa5…; qwen q4 d4579f70b8…; gemma q4 fe8598adf4….
7. Mean prob_on_letters ≥ 0.999 in every standard run.

## G2. Option-A habit in the unedited model (original order)
1. Gemma full, dev LAS+MEK (900): A is correct 219 times, B 221, C 230, D 230; Gemma picks A 140, B 252, C 253, D 255; accuracy when key is A 51%, B 79%, C 73%, D 70%. When key=A and wrong it picked C 40, D 37, B 30.
2. Gemma full vs 4-bit, dev: MEK key=A right 47% → 39%, key=B/C/D 71% → 72%, A picked 71 → 53 (A is key 117×); LAS key=A 56% → 48%, B/C/D 76% → 73%, A picked 69 → 65 (A is key 102×).
3. Full and 4-bit choose the same letter on 1,156 of 1,350 dev questions.
4. Sign tests: 22 vs 4 → p≈0.00053; 22 vs 2 → p≈3.6e-5; accuracy key=A vs others z≈6.29; 123 vs 9 → p≈1e-26.

## G3. Single-question rank-1 edits (archive/single_question_edits)
1. Gemma: 10 failure cases, 5 fixed, 5 could not fix. Fixed: 2015-03-28 LAS 13 (5%, test headline 66.7), 2017-04-01 MEK 24 (5%, 65.0), 2021-03-13 MEK 28 (1%, 70.0), 2022-05-07 LAS 16 (10%, 71.7), 2022-10-23 MEK 26 (1%, 68.3); every random control 70.0; baseline 70.0.
2. Qwen: 7 edits reached, all fixed; test headline after edit 92.5, 91.7, 93.3, 93.3, 89.2, 83.3 (the 20% edit), 92.5; controls 93.3; baseline 93.3.
3. Average change: Gemma about −1.7 points, Qwen about −2.5.
4. Letter shift: Qwen 2021-05-08 MEK 25 edit: A share 29% → 45% (test LAS+MEK); Gemma 2017-04-01 MEK 24: D 24% → 38%.
5. Perplexity did not rise for any edit.

## G4. Group edit (experiments/group/gemma-4b-full)
1. Split: drill_wrong 205, drill_right 515, validation_wrong 41, validation_right 109, hidden_wrong 117, hidden_right 363; 8 hidden, 3 validation, 12 drill papers.
2. Training stopped after 300 steps (no_improvement), chose step 50; validation headline 68.0% at steps 0 and 50; training-log rows as reported (e.g. step 150: 6 fixed, 501 kept, val 67.0).
3. Full precision hidden: trained fixes 6/117 (control 1/117), keeps 359/363 (control 363); test headline 70.0 → 70.8 (control 70.0).
4. results.md: hidden net +2 questions, CI includes zero; test verdict NO IMPROVEMENT; 4-bit hidden 72.9% vs 73.3% unedited 4-bit, net −2.

## G5. AI editors (editor_runs/gemma_b10, gemma_b25, gemma_b50)
1. Passes used 5/10, 7/25, 11/50; all three submitted an all-zero edit (0 weights changed).
2. No rule breaks: no gradients/autograd/optimisers, no reads of exams/, experiments/, runs/, archive/, data/damage, no model loading, no Workbench.create, no ledger edits (check all scripts and ledgers).
3. Trial-edit practice results: b10 pass3 336→332 (fixed 16, broke 20), pass4 336→339 (11/8); b25 pass3 337→339 (9/7), pass4 337→328 (15/24); b50 pass5 350→351, pass6 350→351, pass9 350→354 (whole practice set 624→628 per the editor).
4. Editors' diagnoses, as quoted: b10 "A chosen 86 times in 580 questions, key A 145"; b25 MEK key=A 47%, LAS 55%; b50 rotation: key A→D fixed 22 broke 4, B→A broke 22 fixed 2.
5. Trial edits registered for testing: weights changed trial_b10_q_full 606,820; trial_b10_q_half 438,051; trial_b25_mlp 3,932,160; trial_b50_k 366,231 (share 0.015%, layer 23 k_proj rows).

## G6. Trial edits at full precision and 4-bit (experiments/editor/trial_*)
Hidden LAS+MEK (320, key=A 74): unedited full all 72.5 / A-right 51.4 / BCD 78.9 / A picked 47; unedited 4-bit 68.8 / 41.9 / 76.8 / 38;
trial_b50_k full 73.8 / 60.8 / 77.6 / 58; 4-bit 70.0 / 50.0 / 76.0 / 47; its control full 72.8 / 51.4 / 79.3 / 47;
trial_b10_q_full full 70.3 / 54.1 / 75.2 / 51; 4-bit 65.0 / 44.6 / 71.1 / 42;
trial_b25_mlp full 72.5 / 52.7 / 78.5 / 48; 4-bit 65.6 / 39.2 / 73.6 / 39.
Gain in A-right from trial_b50_k: +9.4 full, +8.1 at 4-bit. Test LAS+MEK: A-right 47.2 → 55.6 (full).
Damage: trial_b50_k within limits; b10/b25 trials flagged (loop rate +4 points).
4-bit hidden net for trial_b50_k: +4 questions, CI includes zero.

## G7. Rotation tests on Högskoleprovet
1. Dev (rotation_proof.json): three unedited repeats identical on all 3,600 answers; unedited right in all 4 positions 405, in none 109; by position A 53.6, B 75.9, C 72.0, D 70.8; editor 3 change: A 62.8, B 73.4, C 70.1, D 69.9; all-4 439; never 115; position-dependent 386 → 346; wrong-only-at-A 124 → 76; right-only-at-A 9 → 22; right-at-A gained 83 lost 0 (p≈2e-25); all-4 gained 43 lost 9 (p≈2e-6); random control at A 2/0.
2. Held-out (heldout_position.jsonl): hidden original 232, prompt note 230, option averaging 238, editor 3 236, editor 3 + averaging 238; test 84, 88, 90, 85, 90; A-right hidden 57.2 → 65.6, test 57.5 → 66.7; wrong-only-at-A 39 → 23 (hidden), 14 → 8 (test); none 39 → 38, 12 → 15; held-out right-at-A gained/lost 27/0 (hidden), 11/0 (test).
3. Option averaging is computed by summing each option text's probability over the 4 rotations (check the code does this correctly).

## G8. Other exams (experiments/analysis/other_exams)
1. Unedited by position: Belebele sv A 76.4 B 82.4 C 83.8 D 84.6 (all-4 345); en A 83.4 B 88.4 C 86.4 D 87.2 (394); MMLU maths A 45.8 B 46.0 C 34.5 D 34.5 (46). Editor 3: sv 80.6/81.2/83.8/83.8 (358); en 85.8/87.8/85.8/86.0 (392); maths 57.0/40.0/32.5/30.2 (47). Random equals unedited (maths within 0.2).
2. Gains/losses with editor 3: sv A +21/−0, B 0/6, C 2/2, D 1/5; en A +12/−0, B 1/4, C 0/3, D 2/8; maths A +45/−0, B 2/26, C 2/10, D 0/17. Means over positions: sv 81.8 → 82.3, en 86.4 → 86.4, maths 40.2 → 39.9.
3. Letter picks: sv unedited A 22.2 B 25.0 C 25.6 D 27.2, edited 24.1/24.1/25.1/26.8; en 23.1/25.9/25.1/26.0 → 24.8/25.4/24.6/25.2; maths 28.9/29.5/20.2/21.4 → 39.2/24.3/18.5/18.0.
4. Belebele sv and en use the same 500 questions; dataset revisions belebele 7899cdf…, mmlu c30699e…; licences CC BY-SA 4.0 and MIT.
5. a_habit_all_tests: wrong-only-at-A vs right-only-at-A: HP dev 124/9, hidden 39/3, test 14/2, Belebele sv 31/3, en 15/2, maths 24/61 (reversed); A is the only penalised letter on reading tests; maths favours A and B.
