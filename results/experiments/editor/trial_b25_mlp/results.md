# Editor edit: trial_b25_mlp

One weight edit of `gemma-4b-full` made by an AI editor (`editor_runs/trial_b25_mlp/final_edit.safetensors`), scored on papers the editor never saw (the hidden papers of the group experiment), against a random edit touching exactly the same weights with the same size. See `src/bench/editor/evaluate.py` for the method.

## Verdict

**Overall label: harm**

The editor with a budget of 1 forward passes (used 0) changed 3,932,160 weights in 6 tensors (0.0914% of the model), in layers 13, 14, 15, 16, 17, 18. On the practice papers (870 questions the editor could use) accuracy goes from 71.7% to 71.8%, +0.1 percentage points (95% CI -1.3 to +1.5); the random edit: -0.6 percentage points. This is what the editor optimised, so it only shows how well the edit fits those questions.

On the hidden papers (480 questions the editor never saw) the edit fixes 9 of 117 wrong answers and breaks 7 of 363 right ones: net +2 questions, +0.4 percentage points accuracy (95% CI -1.2 to +2.1). The interval includes zero, so the net change cannot be told apart from no change. The random edit of the same shape and size nets -1 questions; edited minus random is +0.6 percentage points (95% CI -1.2 to +2.5). The interval includes zero, so the edit does no better than a random edit might by chance.

On the test split the decision rule says NO IMPROVEMENT (headline change +0.0086, 95% CI -0.0237 to +0.0443; rules not passed: 1_metric_improves, 2_beats_control, 3b_loop_rate). Damage checks: perplexity -2.51%, loop rate +4.0 points: outside the limits, so the edit damages the model. On gemma_extra (indicative only) accuracy changes by +0.0 percentage points (95% CI -3.3 to +3.3); the interval includes zero, so it cannot be told apart from no change.

After compression to 4-bit, against the unedited 4-bit model, the hidden-paper net change is -12 questions, -2.5 percentage points (95% CI -4.4 to -0.6); the interval lies below zero, a real loss. There was no full-precision gain to carry over.

Overall: the edit does harm (it damages the model); it is not an improvement.

## The edit

* Budget: 1 forward passes; used 0
* Format: cols
* Weights touched (nonzero in the change): 3,932,160 in 6 tensors, 0.0914% of the model's 4,300,079,472 weights; 3,932,160 of them still differ after rounding to bfloat16
* Layers: 13, 14, 15, 16, 17, 18; modules: mlp.down_proj (6)
* Size as reported by the editor: `{"weights_touched": 3932160, "weights_changed": 3932160, "model_weights": 4300079472, "share_of_model": 0.0009144389134210867}`

The editor's explanation:

> Editor 2 trial (pass 6): x2 on 256 first-option correctness neurons per layer, MLP down_proj columns, layers 13-18.

| Tensor | Layer | Kind | Rows / columns touched | Weights touched | Share of tensor | ‖ΔW‖ / ‖W‖ | Norm landed in bf16 (edit / random) |
|---|---|---|---|---|---|---|---|
| `mlp.down_proj` | 13 | cols | 256 columns: 19, 29, 120, 141, 171, 173, 175, 208, 219, 332, 392, 473, 512, 519, 622, 626, 633, 642, 690, 717, 784, 810, 833, 842, … | 655,360 | 2.5% | 0.1571 | 100.0% / 100.0% |
| `mlp.down_proj` | 14 | cols | 256 columns: 9, 26, 73, 140, 192, 197, 224, 285, 389, 431, 514, 543, 562, 572, 614, 640, 653, 770, 791, 836, 846, 889, 964, 990, … | 655,360 | 2.5% | 0.1577 | 100.0% / 100.0% |
| `mlp.down_proj` | 15 | cols | 256 columns: 25, 72, 88, 152, 240, 249, 325, 349, 369, 417, 483, 597, 619, 692, 723, 741, 798, 820, 821, 837, 881, 917, 940, 962, … | 655,360 | 2.5% | 0.1593 | 100.0% / 100.0% |
| `mlp.down_proj` | 16 | cols | 256 columns: 7, 82, 85, 111, 180, 241, 249, 260, 283, 344, 412, 555, 790, 845, 865, 915, 936, 990, 1010, 1045, 1062, 1075, 1108, 1174, … | 655,360 | 2.5% | 0.1563 | 100.0% / 100.0% |
| `mlp.down_proj` | 17 | cols | 256 columns: 14, 62, 74, 155, 205, 282, 302, 333, 341, 347, 367, 373, 461, 503, 533, 543, 566, 625, 655, 682, 684, 746, 753, 833, … | 655,360 | 2.5% | 0.1587 | 100.0% / 100.0% |
| `mlp.down_proj` | 18 | cols | 256 columns: 36, 134, 143, 167, 243, 257, 275, 282, 313, 443, 491, 497, 528, 541, 565, 580, 659, 713, 723, 796, 799, 812, 842, 925, … | 655,360 | 2.5% | 0.159 | 100.0% / 100.0% |

Random control: seed 1; same tensors, same rows/columns and nonzero pattern; Gaussian values scaled to the same Frobenius norm per tensor.

## Full precision

The unedited hidden-paper run disagrees with split.json on 0 questions.

### Practice papers (drill + validation; what the editor optimised)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 870 | 71.7% → 71.8% | 18/246 | 17/624 | +1 questions, +0.1 pp (95% CI -1.3 to +1.5) | +0.0019 [-0.0097, +0.0139] |
| control | 870 | 71.7% → 71.1% | 6/246 | 11/624 | -5 questions, -0.6 pp (95% CI -1.5 to +0.3) | -0.0027 [-0.0101, +0.0046] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 246 | 18 | 6 | +4.9 pp (95% CI +2.0 to +8.1) |
| right before: kept | 624 | 607 | 613 | -1.0 pp (95% CI -2.2 to +0.3) |
| **all** | 870 | 625 | 619 | +0.7 pp (95% CI -0.6 to +2.0) |

By section (practice):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 83.1% → 83.8% | 3/49 | 1/241 | +2 |
| edited | LAS | 70.3% → 68.3% | 5/86 | 11/204 | -6 |
| edited | MEK | 61.7% → 63.4% | 10/111 | 5/179 | +5 |
| control | ORD | 83.1% → 82.8% | 1/49 | 2/241 | -1 |
| control | LAS | 70.3% → 68.6% | 0/86 | 5/204 | -5 |
| control | MEK | 61.7% → 62.1% | 5/111 | 4/179 | +1 |

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 75.6% → 76.0% | 9/117 | 7/363 | +2 questions, +0.4 pp (95% CI -1.2 to +2.1) | +0.0035 [-0.0112, +0.0181] |
| control | 480 | 75.6% → 75.4% | 6/117 | 7/363 | -1 questions, -0.2 pp (95% CI -1.7 to +1.2) | -0.0037 [-0.0162, +0.0087] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 117 | 9 | 6 | +2.6 pp (95% CI -2.6 to +7.7) |
| right before: kept | 363 | 356 | 356 | +0.0 pp (95% CI -1.7 to +1.7) |
| **all** | 480 | 365 | 362 | +0.6 pp (95% CI -1.2 to +2.5) |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 81.9% → 83.1% | 3/29 | 1/131 | +2 |
| edited | LAS | 73.8% → 72.5% | 2/42 | 4/118 | -2 |
| edited | MEK | 71.2% → 72.5% | 4/46 | 2/114 | +2 |
| control | ORD | 81.9% → 81.9% | 1/29 | 1/131 | +0 |
| control | LAS | 73.8% → 75.0% | 2/42 | 0/118 | +2 |
| control | MEK | 71.2% → 69.4% | 3/46 | 6/114 | -3 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | +0.0086 [-0.0237, +0.0443] | +0.56 [-1.67, +2.79] | 3 | 2 | 1.1 pp | -2.51% | +4.0 points |
| test | random vs unedited | NO IMPROVEMENT | -0.0219 [-0.0504, -0.0001] | -1.67 [-3.89, +0.00] | 0 | 3 | 1.1 pp | -0.24% | +0.0 points |
| test | edited vs random | — | +0.0305 [-0.0043, +0.0686] | +2.22 [-0.56, +5.00] | 5 | 1 | 2.2 pp | -2.27% | +4.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | -0.0022 [-0.0507, +0.0468] | +0.00 [-3.33, +3.33] | 2 | 2 | 1.7 pp | n/a | n/a |
| gemma_extra | random vs unedited | NO IMPROVEMENT (indicative) | -0.0135 [-0.0509, +0.0197] | -0.83 [-3.33, +1.67] | 1 | 2 | 1.7 pp | n/a | n/a |
| gemma_extra | edited vs random | — | +0.0113 [-0.0369, +0.0641] | +0.83 [-2.50, +4.17] | 3 | 2 | 2.5 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | +0.0002 [+0.0000, +0.0005] |
| LAS | 60 | -1.67 [-5.00, +0.00] | -0.0166 [-0.0498, +0.0000] |
| MEK | 60 | +3.33 [-3.33, +10.00] | +0.0337 [-0.0198, +0.0960] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +0.6 | -1.1 | -0.6 | +1.1 | +0.0 |
| test | control | -0.6 | +1.1 | -0.6 | +0.0 | +0.0 |
| gemma_extra | edited | +1.7 | +0.0 | -0.8 | -0.8 | +0.0 |
| gemma_extra | control | +0.0 | +0.0 | +1.7 | -1.7 | +0.0 |

## 4-bit (Q4_K_M)

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 73.3% → 70.8% | 5/128 | 17/352 | -12 questions, -2.5 pp (95% CI -4.4 to -0.6) | -0.0232 [-0.0411, -0.0063] |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 82.5% → 81.2% | 1/28 | 3/132 | -2 |
| edited | LAS | 70.6% → 65.6% | 3/47 | 11/113 | -8 |
| edited | MEK | 66.9% → 65.6% | 1/53 | 3/107 | -2 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | -0.0159 [-0.0477, +0.0133] | +0.56 [-2.22, +3.33] | 4 | 3 | 1.7 pp | -3.32% | +6.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | +0.0154 [-0.0100, +0.0515] | +1.67 [+0.00, +4.17] | 2 | 0 | 1.7 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +5.00 [+0.00, +11.67] | +0.0449 [+0.0000, +0.1014] |
| LAS | 60 | -1.67 [-8.33, +3.33] | -0.0187 [-0.0813, +0.0322] |
| MEK | 60 | -1.67 [-5.00, +0.00] | -0.0132 [-0.0383, +0.0019] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | -1.1 | +1.7 | +0.0 | +0.0 | -0.6 |
| gemma_extra | edited | +0.8 | +0.0 | +0.8 | -1.7 | +0.0 |

## Unedited runs used

* full hidden: `experiments/group/gemma-4b-full/baseline/hidden`
* full drill: `experiments/group/gemma-4b-full/baseline/drill`
* full validation: `experiments/group/gemma-4b-full/baseline/validation`
* full test: `experiments/group/gemma-4b-full/baseline/test`
* full gemma_extra: `experiments/group/gemma-4b-full/baseline/gemma_extra`
* q4 hidden: `experiments/group/gemma-4b-full/baseline/q4_hidden`
* q4 test: `runs/20261119T061642Z_gemma-4b-q4_test`
* q4 gemma_extra: `runs/20261119T061922Z_gemma-4b-q4_gemma_extra`
