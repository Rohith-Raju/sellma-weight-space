# Editor edit: trial_b10_q_full

One weight edit of `gemma-4b-full` made by an AI editor (`editor_runs/trial_b10_q_full/final_edit.safetensors`), scored on papers the editor never saw (the hidden papers of the group experiment), against a random edit touching exactly the same weights with the same size. See `src/bench/editor/evaluate.py` for the method.

## Verdict

**Overall label: harm**

The editor with a budget of 1 forward passes (used 0) changed 1,254,160 weights in 1 tensors (0.0292% of the model), in layers 23. On the practice papers (870 questions the editor could use) accuracy goes from 71.7% to 71.3%, -0.5 percentage points (95% CI -2.0 to +1.0); the random edit: +0.0 percentage points. This is what the editor optimised, so it only shows how well the edit fits those questions.

On the hidden papers (480 questions the editor never saw) the edit fixes 10 of 117 wrong answers and breaks 17 of 363 right ones: net -7 questions, -1.5 percentage points accuracy (95% CI -3.5 to +0.6). The interval includes zero, so the net change cannot be told apart from no change. The random edit of the same shape and size nets +0 questions; edited minus random is -1.5 percentage points (95% CI -3.5 to +0.6). The interval includes zero, so the edit does no better than a random edit might by chance.

On the test split the decision rule says NO IMPROVEMENT (headline change -0.0070, 95% CI -0.0350 to +0.0171; rules not passed: 1_metric_improves, 2_beats_control, 3b_loop_rate). Damage checks: perplexity -0.03%, loop rate +4.0 points: outside the limits, so the edit damages the model. On gemma_extra (indicative only) accuracy changes by -0.8 percentage points (95% CI -5.0 to +2.5); the interval includes zero, so it cannot be told apart from no change.

After compression to 4-bit, against the unedited 4-bit model, the hidden-paper net change is -12 questions, -2.5 percentage points (95% CI -4.8 to -0.2); the interval lies below zero, a real loss. There was no full-precision gain to carry over.

Overall: the edit does harm (it damages the model); it is not an improvement.

## The edit

* Budget: 1 forward passes; used 0
* Format: rows
* Weights touched (nonzero in the change): 1,254,160 in 1 tensors, 0.0292% of the model's 4,300,079,472 weights; 606,820 of them still differ after rounding to bfloat16
* Layers: 23; modules: self_attn.q_proj (1)
* Size as reported by the editor: `{"weights_touched": 1254160, "weights_changed": 606820, "model_weights": 4300079472, "share_of_model": 0.00029165972586471306}`

The editor's explanation:

> Editor 1 trial (pass 3): rank-1 change to layer-23 q_proj rows of heads 3 and 7 removing the position prior, full strength. Not submitted by the editor; scored only to see what compression does to an edit.

| Tensor | Layer | Kind | Rows / columns touched | Weights touched | Share of tensor | ‖ΔW‖ / ‖W‖ | Norm landed in bf16 (edit / random) |
|---|---|---|---|---|---|---|---|
| `self_attn.q_proj` | 23 | rows | 512 rows: 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, … | 1,254,160 | 23.9% | 0.003148 | 100.8% / 102.0% |

Random control: seed 1; same tensors, same rows/columns and nonzero pattern; Gaussian values scaled to the same Frobenius norm per tensor.

## Full precision

The unedited hidden-paper run disagrees with split.json on 0 questions.

### Practice papers (drill + validation; what the editor optimised)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 870 | 71.7% → 71.3% | 19/246 | 23/624 | -4 questions, -0.5 pp (95% CI -2.0 to +1.0) | -0.0041 [-0.0173, +0.0087] |
| control | 870 | 71.7% → 71.7% | 1/246 | 1/624 | +0 questions, +0.0 pp (95% CI -0.3 to +0.3) | +0.0001 [-0.0002, +0.0004] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 246 | 19 | 1 | +7.3 pp (95% CI +4.1 to +10.6) |
| right before: kept | 624 | 601 | 623 | -3.5 pp (95% CI -5.1 to -2.1) |
| **all** | 870 | 620 | 624 | -0.5 pp (95% CI -2.0 to +1.0) |

By section (practice):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 83.1% → 83.1% | 1/49 | 1/241 | +0 |
| edited | LAS | 70.3% → 70.3% | 7/86 | 7/204 | +0 |
| edited | MEK | 61.7% → 60.3% | 11/111 | 15/179 | -4 |
| control | ORD | 83.1% → 83.1% | 0/49 | 0/241 | +0 |
| control | LAS | 70.3% → 70.0% | 0/86 | 1/204 | -1 |
| control | MEK | 61.7% → 62.1% | 1/111 | 0/179 | +1 |

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 75.6% → 74.2% | 10/117 | 17/363 | -7 questions, -1.5 pp (95% CI -3.5 to +0.6) | -0.0163 [-0.0365, +0.0029] |
| control | 480 | 75.6% → 75.6% | 0/117 | 0/363 | +0 questions, +0.0 pp (95% CI +0.0 to +0.0) | -0.0000 [-0.0003, +0.0002] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 117 | 10 | 0 | +8.5 pp (95% CI +4.3 to +13.7) |
| right before: kept | 363 | 346 | 363 | -4.7 pp (95% CI -6.9 to -2.8) |
| **all** | 480 | 356 | 363 | -1.5 pp (95% CI -3.5 to +0.6) |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 81.9% → 81.9% | 2/29 | 2/131 | +0 |
| edited | LAS | 73.8% → 75.0% | 5/42 | 3/118 | +2 |
| edited | MEK | 71.2% → 65.6% | 3/46 | 12/114 | -9 |
| control | ORD | 81.9% → 81.9% | 0/29 | 0/131 | +0 |
| control | LAS | 73.8% → 73.8% | 0/42 | 0/118 | +0 |
| control | MEK | 71.2% → 71.2% | 0/46 | 0/114 | +0 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | -0.0070 [-0.0350, +0.0171] | -0.56 [-2.22, +1.11] | 1 | 2 | 2.2 pp | -0.03% | +4.0 points |
| test | random vs unedited | NO IMPROVEMENT | +0.0008 [-0.0001, +0.0025] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | -0.03% | +6.0 points |
| test | edited vs random | — | -0.0078 [-0.0357, +0.0156] | -0.56 [-2.22, +1.11] | 1 | 2 | 2.2 pp | -0.00% | -2.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | -0.0140 [-0.0685, +0.0358] | -0.83 [-5.00, +2.50] | 2 | 3 | 5.8 pp | n/a | n/a |
| gemma_extra | random vs unedited | NO IMPROVEMENT (indicative) | -0.0001 [-0.0005, +0.0001] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | n/a | n/a |
| gemma_extra | edited vs random | — | -0.0139 [-0.0683, +0.0359] | -0.83 [-5.00, +2.50] | 2 | 3 | 5.8 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | +0.0000 [-0.0000, +0.0000] |
| LAS | 60 | +1.67 [+0.00, +5.00] | +0.0126 [-0.0001, +0.0365] |
| MEK | 60 | -3.33 [-8.33, +0.00] | -0.0266 [-0.0815, +0.0115] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +1.1 | +1.1 | +0.0 | -2.2 | +0.0 |
| test | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| gemma_extra | edited | +0.8 | +2.5 | +2.5 | -5.8 | +0.0 |
| gemma_extra | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

## 4-bit (Q4_K_M)

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 73.3% → 70.8% | 9/128 | 21/352 | -12 questions, -2.5 pp (95% CI -4.8 to -0.2) | -0.0195 [-0.0390, -0.0000] |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 82.5% → 82.5% | 1/28 | 1/132 | +0 |
| edited | LAS | 70.6% → 67.5% | 3/47 | 8/113 | -5 |
| edited | MEK | 66.9% → 62.5% | 5/53 | 12/107 | -7 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | -0.0043 [-0.0448, +0.0362] | -0.56 [-3.33, +2.22] | 3 | 4 | 5.6 pp | -0.09% | -2.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | +0.0080 [-0.0419, +0.0549] | +0.83 [-2.50, +4.17] | 3 | 2 | 3.3 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | +0.0003 [-0.0005, +0.0015] |
| LAS | 60 | +3.33 [+0.00, +8.33] | +0.0318 [-0.0011, +0.0803] |
| MEK | 60 | -5.00 [-11.67, +1.67] | -0.0405 [-0.1102, +0.0234] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +0.6 | +3.3 | +2.2 | -5.6 | -0.6 |
| gemma_extra | edited | +0.8 | -0.8 | +3.3 | -3.3 | +0.0 |

## Unedited runs used

* full hidden: `experiments/group/gemma-4b-full/baseline/hidden`
* full drill: `experiments/group/gemma-4b-full/baseline/drill`
* full validation: `experiments/group/gemma-4b-full/baseline/validation`
* full test: `experiments/group/gemma-4b-full/baseline/test`
* full gemma_extra: `experiments/group/gemma-4b-full/baseline/gemma_extra`
* q4 hidden: `experiments/group/gemma-4b-full/baseline/q4_hidden`
* q4 test: `runs/20261119T061642Z_gemma-4b-q4_test`
* q4 gemma_extra: `runs/20261119T061922Z_gemma-4b-q4_gemma_extra`
