# Editor edit: trial_b10_q_half

One weight edit of `gemma-4b-full` made by an AI editor (`editor_runs/trial_b10_q_half/final_edit.safetensors`), scored on papers the editor never saw (the hidden papers of the group experiment), against a random edit touching exactly the same weights with the same size. See `src/bench/editor/evaluate.py` for the method.

## Verdict

**Overall label: harm**

The editor with a budget of 1 forward passes (used 0) changed 1,254,160 weights in 1 tensors (0.0292% of the model), in layers 23. On the practice papers (870 questions the editor could use) accuracy goes from 71.7% to 72.1%, +0.3 percentage points (95% CI -0.7 to +1.4); the random edit: -0.1 percentage points. This is what the editor optimised, so it only shows how well the edit fits those questions.

On the hidden papers (480 questions the editor never saw) the edit fixes 7 of 117 wrong answers and breaks 12 of 363 right ones: net -5 questions, -1.0 percentage points accuracy (95% CI -2.9 to +0.6). The interval includes zero, so the net change cannot be told apart from no change. The random edit of the same shape and size nets +0 questions; edited minus random is -1.0 percentage points (95% CI -2.9 to +0.6). The interval includes zero, so the edit does no better than a random edit might by chance.

On the test split the decision rule says NO IMPROVEMENT (headline change -0.0092, 95% CI -0.0296 to +0.0057; rules not passed: 1_metric_improves, 2_beats_control, 3b_loop_rate). Damage checks: perplexity -0.02%, loop rate +4.0 points: outside the limits, so the edit damages the model. On gemma_extra (indicative only) accuracy changes by -1.7 percentage points (95% CI -4.2 to +0.0); the interval includes zero, so it cannot be told apart from no change.

After compression to 4-bit, against the unedited 4-bit model, the hidden-paper net change is -5 questions, -1.0 percentage points (95% CI -2.7 to +0.4); the interval includes zero, so it cannot be told apart from no change. There was no full-precision gain to carry over.

Overall: the edit does harm (it damages the model); it is not an improvement.

## The edit

* Budget: 1 forward passes; used 0
* Format: rows
* Weights touched (nonzero in the change): 1,254,160 in 1 tensors, 0.0292% of the model's 4,300,079,472 weights; 438,051 of them still differ after rounding to bfloat16
* Layers: 23; modules: self_attn.q_proj (1)
* Size as reported by the editor: `{"weights_touched": 1254160, "weights_changed": 438051, "model_weights": 4300079472, "share_of_model": 0.00029165972586471306}`

The editor's explanation:

> Editor 1 trial (pass 4): the same change at half strength.

| Tensor | Layer | Kind | Rows / columns touched | Weights touched | Share of tensor | ‖ΔW‖ / ‖W‖ | Norm landed in bf16 (edit / random) |
|---|---|---|---|---|---|---|---|
| `self_attn.q_proj` | 23 | rows | 512 rows: 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, … | 1,254,160 | 23.9% | 0.001574 | 101.3% / 103.5% |

Random control: seed 1; same tensors, same rows/columns and nonzero pattern; Gaussian values scaled to the same Frobenius norm per tensor.

## Full precision

The unedited hidden-paper run disagrees with split.json on 0 questions.

### Practice papers (drill + validation; what the editor optimised)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 870 | 71.7% → 72.1% | 13/246 | 10/624 | +3 questions, +0.3 pp (95% CI -0.7 to +1.4) | +0.0037 [-0.0051, +0.0124] |
| control | 870 | 71.7% → 71.6% | 1/246 | 2/624 | -1 questions, -0.1 pp (95% CI -0.6 to +0.2) | +0.0000 [-0.0003, +0.0004] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 246 | 13 | 1 | +4.9 pp (95% CI +2.4 to +7.7) |
| right before: kept | 624 | 614 | 622 | -1.3 pp (95% CI -2.4 to -0.3) |
| **all** | 870 | 627 | 623 | +0.5 pp (95% CI -0.7 to +1.6) |

By section (practice):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 83.1% → 82.8% | 0/49 | 1/241 | -1 |
| edited | LAS | 70.3% → 71.4% | 6/86 | 3/204 | +3 |
| edited | MEK | 61.7% → 62.1% | 7/111 | 6/179 | +1 |
| control | ORD | 83.1% → 83.1% | 0/49 | 0/241 | +0 |
| control | LAS | 70.3% → 70.0% | 0/86 | 1/204 | -1 |
| control | MEK | 61.7% → 61.7% | 1/111 | 1/179 | +0 |

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 75.6% → 74.6% | 7/117 | 12/363 | -5 questions, -1.0 pp (95% CI -2.9 to +0.6) | -0.0086 [-0.0239, +0.0061] |
| control | 480 | 75.6% → 75.6% | 0/117 | 0/363 | +0 questions, +0.0 pp (95% CI +0.0 to +0.0) | -0.0003 [-0.0008, +0.0000] |

Edited minus random (counts are questions answered correctly):

| Questions | n | Edited | Random | Difference (CI) |
|---|---|---|---|---|
| wrong before: fixed | 117 | 7 | 0 | +6.0 pp (95% CI +1.7 to +10.3) |
| right before: kept | 363 | 351 | 363 | -3.3 pp (95% CI -5.2 to -1.7) |
| **all** | 480 | 358 | 363 | -1.0 pp (95% CI -2.9 to +0.6) |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 81.9% → 81.9% | 1/29 | 1/131 | +0 |
| edited | LAS | 73.8% → 73.8% | 3/42 | 3/118 | +0 |
| edited | MEK | 71.2% → 68.1% | 3/46 | 8/114 | -5 |
| control | ORD | 81.9% → 81.9% | 0/29 | 0/131 | +0 |
| control | LAS | 73.8% → 73.8% | 0/42 | 0/118 | +0 |
| control | MEK | 71.2% → 71.2% | 0/46 | 0/114 | +0 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | -0.0092 [-0.0296, +0.0057] | -1.11 [-2.78, +0.00] | 0 | 2 | 1.1 pp | -0.02% | +4.0 points |
| test | random vs unedited | NO IMPROVEMENT | +0.0000 [-0.0000, +0.0001] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | -0.06% | +4.0 points |
| test | edited vs random | — | -0.0093 [-0.0296, +0.0057] | -1.11 [-2.78, +0.00] | 0 | 2 | 1.1 pp | +0.04% | +0.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | -0.0218 [-0.0612, +0.0069] | -1.67 [-4.17, +0.00] | 0 | 2 | 2.5 pp | n/a | n/a |
| gemma_extra | random vs unedited | NO IMPROVEMENT (indicative) | -0.0001 [-0.0010, +0.0005] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | n/a | n/a |
| gemma_extra | edited vs random | — | -0.0217 [-0.0610, +0.0069] | -1.67 [-4.17, +0.00] | 0 | 2 | 2.5 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | +0.0000 [-0.0000, +0.0000] |
| LAS | 60 | +0.00 [+0.00, +0.00] | +0.0001 [+0.0000, +0.0002] |
| MEK | 60 | -3.33 [-8.33, +0.00] | -0.0186 [-0.0598, +0.0113] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +0.6 | +0.6 | +0.0 | -1.1 | +0.0 |
| test | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| gemma_extra | edited | +0.8 | +0.8 | +0.8 | -2.5 | +0.0 |
| gemma_extra | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

## 4-bit (Q4_K_M)

### Hidden papers (never seen by the editor)

| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | Mean prob. correct change (CI) |
|---|---|---|---|---|---|---|
| edited | 480 | 73.3% → 72.3% | 5/128 | 10/352 | -5 questions, -1.0 pp (95% CI -2.7 to +0.4) | -0.0047 [-0.0168, +0.0078] |

By section (hidden):

| Edit | Section | Accuracy before → after | Fixed | Broken | Net |
|---|---|---|---|---|---|
| edited | ORD | 82.5% → 81.9% | 0/28 | 1/132 | -1 |
| edited | LAS | 70.6% → 68.1% | 1/47 | 5/113 | -4 |
| edited | MEK | 66.9% → 66.9% | 4/53 | 4/107 | +0 |

### Against the unedited model and the random edit (bench.report)

Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to edited vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | edited vs unedited | NO IMPROVEMENT | -0.0009 [-0.0377, +0.0374] | +0.00 [-2.78, +3.33] | 4 | 4 | 4.4 pp | +0.01% | +4.0 points |
| gemma_extra | edited vs unedited | NO IMPROVEMENT (indicative) | +0.0133 [-0.0134, +0.0427] | +0.00 [-2.50, +2.50] | 1 | 1 | 1.7 pp | n/a | n/a |

By section (test split, edited vs unedited):

| Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|
| ORD | 60 | +0.00 [+0.00, +0.00] | +0.0002 [-0.0004, +0.0012] |
| LAS | 60 | +3.33 [+0.00, +8.33] | +0.0310 [-0.0018, +0.0791] |
| MEK | 60 | -3.33 [-11.67, +5.00] | -0.0327 [-0.0959, +0.0256] |

Chosen letters: change in the share of answers choosing each letter, against the unedited model (percentage points). An edit that only pushes the model toward one letter moves answers without learning anything.

| Set | Edit | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | edited | +1.1 | +1.7 | +2.2 | -4.4 | -0.6 |
| gemma_extra | edited | +0.0 | +0.0 | +1.7 | -1.7 | +0.0 |

## Unedited runs used

* full hidden: `experiments/group/gemma-4b-full/baseline/hidden`
* full drill: `experiments/group/gemma-4b-full/baseline/drill`
* full validation: `experiments/group/gemma-4b-full/baseline/validation`
* full test: `experiments/group/gemma-4b-full/baseline/test`
* full gemma_extra: `experiments/group/gemma-4b-full/baseline/gemma_extra`
* q4 hidden: `experiments/group/gemma-4b-full/baseline/q4_hidden`
* q4 test: `runs/20261119T061642Z_gemma-4b-q4_test`
* q4 gemma_extra: `runs/20261119T061922Z_gemma-4b-q4_gemma_extra`
