# Editor edit: trial_b50_k

One weight edit of `gemma-4b-full` made by an AI editor (`editor_runs/trial_b50_k/final_edit.safetensors`), scored on papers the editor never saw (the hidden papers of the group experiment), against a random edit touching exactly the same weights with the same size. See `src/bench/editor/evaluate.py` for the method.

## Verdict

**Overall label: no evidence**

The editor with a budget of 1 forward passes (used 0) changed 655,360 weights in 1 tensors (0.0152% of the model), in layers 23. On the practice papers (870 questions the editor could use) accuracy goes from 71.7% to 72.2%, +0.5 percentage points (95% CI -0.8 to +1.7); the random edit: +0.0 percentage points. This is what the editor optimised, so it only shows how well the edit fits those questions.

On the hidden papers (480 questions the editor never saw) the edit fixes 9 of 117 wrong answers and breaks 4 of 363 right ones: net +5 questions, +1.0 percentage points accuracy (95% CI -0.4 to +2.5). The interval includes zero, so the net change cannot be told apart from no change. The random edit of the same shape and size nets +1 questions; edited minus random is +0.8 percentage points (95% CI -0.6 to +2.3). The interval includes zero, so the edit does no better than a random edit might by chance.

On the test split the decision rule says NO IMPROVEMENT (headline change +0.0092, 95% CI -0.0164 to +0.0383; rules not passed: 1_metric_improves, 2_beats_control). Damage checks: perplexity +0.05%, loop rate -2.0 points: within the limits. On gemma_extra (indicative only) accuracy changes by -2.5 percentage points (95% CI -5.8 to +0.0); the interval includes zero, so it cannot be told apart from no change.

After compression to 4-bit, against the unedited 4-bit model, the hidden-paper net change is +4 questions, +0.8 percentage points (95% CI -0.4 to +2.1); the interval includes zero, so it cannot be told apart from no change. There was no full-precision gain to carry over.

Overall: no evidence that the edit makes the model better at Swedish beyond the questions the editor worked on.

## The edit

* Budget: 1 forward passes; used 0
* Format: rows
* Weights touched (nonzero in the change): 655,360 in 1 tensors, 0.0152% of the model's 4,300,079,472 weights; 366,231 of them still differ after rounding to bfloat16
* Layers: 23; modules: self_attn.k_proj (1)
* Size as reported by the editor: `{"weights_touched": 655360, "weights_changed": 366231, "model_weights": 4300079472, "share_of_model": 0.0001524064855701811}`

The editor's explanation:

> Editor 3 trial (pass 9): layer-23 k_proj rows of heads 2 and 3 raising attention to the first option newline, gamma 3.5.

| Tensor | Layer | Kind | Rows / columns touched | Weights touched | Share of tensor | ‖ΔW‖ / ‖W‖ | Norm landed in bf16 (edit / random) |
|---|---|---|---|---|---|---|---|
| `self_attn.k_proj` | 23 | rows | 256 rows: 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, … | 655,360 | 25.0% | 0.005404 | 100.5% / 101.2% |

Random control: seed 1; same tensors, same rows/columns and nonzero pattern; Gaussian values scaled to the same Frobenius norm per tensor.

## Full precision

The unedited hidden-paper run disagrees with split.json on 0 questions.

### Practice papers (drill + validation; what the editor optimised)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 870 | 71.7% → 72.2% | 18/246 | 14/624 | +4 questions, +0.5 pp (95% CI -0.8 to +1.7) | +0.0044 [-0.0065, +0.0155] |
| control | 870 | 71.7% → 71.7% | 1/246 | 1/624 | +0 questions, +0.0 pp (95% CI -0.3 to +0.3) | -0.0002 [-0.0007, +0.0002] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 246 | 18 | 1 | +6.9 pp (95% CI +4.1 to +10.2) |
| right before: kept | 624 | 610 | 623 | -2.1 pp (95% CI -3.2 to -1.1) |
| **all** | 870 | 628 | 624 | +0.5 pp (95% CI -0.8 to +1.7) |

By section (practice):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 83.1% → 83.8% | 2/49 | 0/241 | +2 |
| edited | LAS | 70.3% → 69.0% | 5/86 | 9/204 | -4 |
| edited | MEK | 61.7% → 63.8% | 11/111 | 5/179 | +6 |
| control | ORD | 83.1% → 83.1% | 0/49 | 0/241 | +0 |
| control | LAS | 70.3% → 70.3% | 0/86 | 0/204 | +0 |
| control | MEK | 61.7% → 61.7% | 1/111 | 1/179 | +0 |

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 75.6% → 76.7% | 9/117 | 4/363 | +5 questions, +1.0 pp (95% CI -0.4 to +2.5) | +0.0097 [-0.0030, +0.0231] |
| control | 480 | 75.6% → 75.8% | 1/117 | 0/363 | +1 questions, +0.2 pp (95% CI +0.0 to +0.6) | -0.0003 [-0.0008, +0.0003] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 117 | 9 | 1 | +6.8 pp (95% CI +2.6 to +12.0) |
| right before: kept | 363 | 359 | 363 | -1.1 pp (95% CI -2.2 to -0.3) |
| **all** | 480 | 368 | 364 | +0.8 pp (95% CI -0.6 to +2.3) |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 81.9% → 82.5% | 1/29 | 0/131 | +1 |
| edited | LAS | 73.8% → 75.6% | 5/42 | 2/118 | +3 |
| edited | MEK | 71.2% → 71.9% | 3/46 | 2/114 | +1 |
| control | ORD | 81.9% → 81.9% | 0/29 | 0/131 | +0 |
| control | LAS | 73.8% → 73.8% | 0/42 | 0/118 | +0 |
| control | MEK | 71.2% → 71.9% | 1/46 | 0/114 | +1 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | +0.0092 [-0.0164, +0.0383] | +0.56 [-1.67, +2.78] | 3 | 2 | 2.8 pp | +0.05% | -2.0 points |
| test | random vs unedited | NO IMPROVEMENT | +0.0005 [+0.0000, +0.0014] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | -0.05% | +8.0 points |
| test | edited vs random | — | +0.0088 [-0.0170, +0.0376] | +0.56 [-1.67, +2.78] | 3 | 2 | 2.8 pp | +0.10% | -10.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | -0.0283 [-0.0683, +0.0021] | -2.50 [-5.83, +0.00] | 0 | 3 | 3.3 pp | n/a | n/a |
| gemma_extra | random vs unedited | NO IMPROVEMENT (indicative) | -0.0000 [-0.0004, +0.0003] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | n/a | n/a |
| gemma_extra | edited vs random | — | -0.0283 [-0.0682, +0.0020] | -2.50 [-5.83, +0.00] | 0 | 3 | 3.3 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | -0.0000 [-0.0000, +0.0000] |
| LAS | 60 | +1.67 [-3.33, +8.33] | +0.0122 [-0.0225, +0.0555] |
| MEK | 60 | +0.00 [-5.00, +5.00] | +0.0063 [-0.0323, +0.0506] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +2.8 | -1.1 | -0.6 | -1.1 | +0.0 |
| test | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| gemma_extra | edited | +3.3 | -1.7 | +0.0 | -1.7 | +0.0 |
| gemma_extra | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

## 4-bit (Q4_K_M)

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 73.3% → 74.2% | 7/128 | 3/352 | +4 questions, +0.8 pp (95% CI -0.4 to +2.1) | +0.0086 [-0.0033, +0.0217] |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 82.5% → 82.5% | 0/28 | 0/132 | +0 |
| edited | LAS | 70.6% → 71.9% | 2/47 | 0/113 | +2 |
| edited | MEK | 66.9% → 68.1% | 5/53 | 3/107 | +2 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | +0.0396 [+0.0043, +0.0784] | +2.78 [+0.00, +5.56] | 6 | 1 | 2.8 pp | +0.03% | +0.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | +0.0243 [-0.0309, +0.0786] | +1.67 [-2.50, +5.83] | 4 | 2 | 4.2 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | +0.0070 [-0.0005, +0.0214] |
| LAS | 60 | +5.00 [+0.00, +11.67] | +0.0472 [+0.0009, +0.1077] |
| MEK | 60 | +3.33 [-3.33, +10.00] | +0.0320 [-0.0179, +0.0906] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +2.8 | -0.6 | +0.6 | -2.8 | +0.0 |
| gemma_extra | edited | +4.2 | -3.3 | +0.8 | -1.7 | +0.0 |

## Unedited runs used

* full hidden: `experiments/group/gemma-4b-full/baseline/hidden`
* full drill: `experiments/group/gemma-4b-full/baseline/drill`
* full validation: `experiments/group/gemma-4b-full/baseline/validation`
* full test: `experiments/group/gemma-4b-full/baseline/test`
* full gemma_extra: `experiments/group/gemma-4b-full/baseline/gemma_extra`
* q4 hidden: `experiments/group/gemma-4b-full/baseline/q4_hidden`
* q4 test: `runs/20261119T061642Z_gemma-4b-q4_test`
* q4 gemma_extra: `runs/20261119T061922Z_gemma-4b-q4_gemma_extra`
