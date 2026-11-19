# Group edit: gemma-4b-full

One small low-rank change, trained on many of the model's wrong answers at once (the drill papers), stopped when the score on the validation papers stopped improving, then tested on wrong answers it never saw (the hidden papers), against a random change of the same size. See `src/bench/group_experiment.py` for the method.

## Verdict

Training kept the change from step 50 of 300, chosen by the validation papers. It fixes 2.0% of the 205 drill questions the model got wrong and keeps 98.8% of the 515 it got right. These are the questions it was trained on, so this only shows how far training got.

At full precision, on the hidden papers (never used in training or for stopping) the trained change fixes 5.1% of the 117 wrong answers, the random change 0.9%: a difference of +4.3 percentage points (95% CI +0.0 to +8.5). The interval includes zero, so this is no evidence that the change generalises: it fixes no more unseen wrong answers than a random change might by chance. It breaks 4 of the 363 hidden questions the model got right (keeps 98.9%); the net change on the hidden papers is +2 questions, +0.4 percentage points accuracy (95% CI -0.8 to +1.7); the interval includes zero, so the net change cannot be told apart from no change. On the test split the decision rule says NO IMPROVEMENT (headline change +0.0008, 95% CI -0.0118 to +0.0167; rules not passed: 1_metric_improves, 2_beats_control, 3b_loop_rate).

After compression to 4-bit, the trained change fixes 3 of 128 wrong answers on the hidden papers (2.3%). There is no random-change run to compare with, so it is not known whether that is more than any change of this size would do. It breaks 5 of the 352 hidden questions the model got right (keeps 98.6%); the net change on the hidden papers is -2 questions, -0.4 percentage points accuracy (95% CI -1.7 to +0.8); the interval includes zero, so the net change cannot be told apart from no change. On the test split the decision rule says NO IMPROVEMENT (headline change +0.0082, 95% CI -0.0063 to +0.0257; rules not passed: 1_metric_improves, 2_beats_control).

## The split

From `runs/20261119T060704Z_gemma-4b-full_dev`. Hidden papers: 2015-03-28, 2016-04-09, 2017-10-21, 2018-10-21, 2019-10-20, 2020-10-25, 2021-05-08, 2022-10-23. Validation papers: 2018-04-14, 2022-03-12, 2023-10-22. Drill papers: the other 12.

| Group | Questions | LÄS | MEK | ORD |
|---|---|---|---|---|
| drill_wrong | 205 | 69 | 96 | 40 |
| drill_right | 515 | 171 | 144 | 200 |
| validation_wrong | 41 | 17 | 15 | 9 |
| validation_right | 109 | 33 | 35 | 41 |
| hidden_wrong | 117 | 42 | 46 | 29 |
| hidden_right | 363 | 118 | 114 | 131 |

## Training

* Chosen step: 50 of 300 steps run.
* Why it stopped: {'patience': 10, 'steps_run': 300, 'stop_reason': 'no_improvement'}
* Log: `experiments/group/gemma-4b-full/trained/training_log.jsonl`
* At the chosen step (scored by the bench): drill wrong fixed 2.0% (4/205), drill right kept 98.8% (509/515).

Training curve (from the log; validation = `(no validation field found in the log)`). The chosen step is marked.

| Step | Validation | Drill wrong fixed | Drill right kept |
|---|---|---|---|
| 0 | n/a | 0.0% | 100.0% |
| 25 | n/a | 1.5% | 99.2% |
| 50 (chosen) | n/a | 2.0% | 98.8% |
| 75 | n/a | 2.4% | 97.9% |
| 100 | n/a | 3.9% | 97.3% |
| 125 | n/a | 1.5% | 97.3% |
| 150 | n/a | 2.9% | 97.3% |
| 175 | n/a | 6.3% | 96.9% |
| 200 | n/a | 6.8% | 97.3% |
| 225 | n/a | 6.3% | 97.3% |
| 250 | n/a | 3.9% | 97.5% |
| 275 | n/a | 6.3% | 96.7% |
| 300 | n/a | 7.8% | 96.9% |

## Full precision

Wrong and right before the change: `runs/20261119T060704Z_gemma-4b-full_dev`.
The unedited re-run on drill disagrees with that on 0 questions.
The unedited re-run on validation disagrees with that on 0 questions.
The unedited re-run on hidden disagrees with that on 0 questions.

### Fixed, kept and broken

| Change | Papers | Wrong fixed | Right kept | Right broken | Net (accuracy, CI) |
|---|---|---|---|---|---|
| trained | drill | 2.0% (4/205) | 98.8% (509/515) | 6 | -2 questions, -0.3 pp (95% CI -1.1 to +0.6) |
| trained | validation | 2.4% (1/41) | 99.1% (108/109) | 1 | +0 questions, +0.0 pp (95% CI -2.0 to +2.0) |
| trained | hidden | 5.1% (6/117) | 98.9% (359/363) | 4 | +2 questions, +0.4 pp (95% CI -0.8 to +1.7) |
| control | drill | 0.0% (0/205) | 99.8% (514/515) | 1 | -1 questions, -0.1 pp (95% CI -0.4 to +0.0) |
| control | validation | 0.0% (0/41) | 100.0% (109/109) | 0 | +0 questions, +0.0 pp (95% CI +0.0 to +0.0) |
| control | hidden | 0.9% (1/117) | 100.0% (363/363) | 0 | +1 questions, +0.2 pp (95% CI +0.0 to +0.6) |

### Trained minus random change

The key number is the bold row: does the trained change fix more unseen wrong answers than a random change of the same size? Counts are questions answered correctly.

| Questions | n | Trained | Random | Difference (CI) |
|---|---|---|---|---|
| **Hidden wrong: fixed** | 117 | 6 | 1 | +4.3 pp (95% CI +0.0 to +8.5) |
| Hidden right: kept | 363 | 359 | 363 | -1.1 pp (95% CI -2.2 to -0.3) |
| Hidden all | 480 | 365 | 364 | +0.2 pp (95% CI -1.2 to +1.7) |
| Validation wrong: fixed | 41 | 1 | 0 | +2.4 pp (95% CI +0.0 to +7.3) |
| Validation right: kept | 109 | 108 | 109 | -0.9 pp (95% CI -2.8 to +0.0) |
| Validation all | 150 | 109 | 109 | +0.0 pp (95% CI -2.0 to +2.0) |
| Drill wrong: fixed | 205 | 4 | 0 | +2.0 pp (95% CI +0.5 to +3.9) |
| Drill right: kept | 515 | 509 | 514 | -1.0 pp (95% CI -1.9 to +0.0) |
| Drill all | 720 | 513 | 514 | -0.1 pp (95% CI -1.1 to +0.7) |

### Against the unedited model (bench.report)

Only the test split gives a verdict that counts; the others are indicative, and on sets without damage checks the damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to trained vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | trained vs unedited | NO IMPROVEMENT | +0.0008 [-0.0118, +0.0167] | +0.56 [+0.00, +1.67] | 1 | 0 | 0.6 pp | -0.62% | +12.0 points |
| test | control vs unedited | NO IMPROVEMENT | +0.0010 [+0.0000, +0.0028] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | -0.05% | -2.0 points |
| test | trained vs random | — | -0.0002 [-0.0132, +0.0157] | +0.56 [+0.00, +1.67] | 1 | 0 | 0.6 pp | -0.57% | +14.0 points |
| gemma_extra | trained vs unedited | NO IMPROVEMENT (indicative) | +0.0001 [-0.0240, +0.0283] | +0.00 [-2.50, +2.50] | 1 | 1 | 1.7 pp | n/a | n/a |
| gemma_extra | control vs unedited | NO IMPROVEMENT (indicative) | +0.0000 [-0.0004, +0.0004] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | n/a | n/a |
| gemma_extra | trained vs random | — | +0.0001 [-0.0241, +0.0285] | +0.00 [-2.50, +2.50] | 1 | 1 | 1.7 pp | n/a | n/a |
| hidden | trained vs unedited | NO IMPROVEMENT (indicative) | -0.0052 [-0.0165, +0.0060] | +0.42 [-0.83, +1.67] | 6 | 4 | 1.7 pp | n/a | n/a |
| hidden | control vs unedited | NO IMPROVEMENT (indicative) | +0.0001 [-0.0006, +0.0011] | +0.21 [+0.00, +0.62] | 1 | 0 | 0.2 pp | n/a | n/a |
| hidden | trained vs random | — | -0.0053 [-0.0168, +0.0061] | +0.21 [-1.04, +1.46] | 6 | 5 | 1.7 pp | n/a | n/a |
| validation | trained vs unedited | NO IMPROVEMENT (indicative) | -0.0013 [-0.0181, +0.0179] | +0.00 [-2.00, +2.00] | 1 | 1 | 2.0 pp | n/a | n/a |
| validation | control vs unedited | NO IMPROVEMENT (indicative) | -0.0002 [-0.0014, +0.0008] | +0.00 [+0.00, +0.00] | 0 | 0 | 0.0 pp | n/a | n/a |
| validation | trained vs random | — | -0.0011 [-0.0179, +0.0181] | +0.00 [-2.00, +2.00] | 1 | 1 | 2.0 pp | n/a | n/a |
| drill | trained vs unedited | NO IMPROVEMENT (indicative) | -0.0023 [-0.0112, +0.0061] | -0.28 [-1.11, +0.56] | 4 | 6 | 1.1 pp | n/a | n/a |
| drill | control vs unedited | NO IMPROVEMENT (indicative) | +0.0001 [-0.0005, +0.0006] | -0.14 [-0.42, +0.00] | 0 | 1 | 0.1 pp | n/a | n/a |
| drill | trained vs random | — | -0.0024 [-0.0113, +0.0062] | -0.14 [-0.97, +0.83] | 5 | 6 | 1.2 pp | n/a | n/a |

### By section (hidden papers)

| Change | Section | Wrong fixed | Right kept | Broken |
|---|---|---|---|---|
| trained | ORD | 3.4% (1/29) | 100.0% (131/131) | 0 |
| trained | LAS | 9.5% (4/42) | 98.3% (116/118) | 2 |
| trained | MEK | 2.2% (1/46) | 98.2% (112/114) | 2 |
| control | ORD | 0.0% (0/29) | 100.0% (131/131) | 0 |
| control | LAS | 0.0% (0/42) | 100.0% (118/118) | 0 |
| control | MEK | 2.2% (1/46) | 100.0% (114/114) | 0 |

### By section (test split, against the unedited model)

| Change | Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|---|
| trained | ORD | 60 | +0.00 [+0.00, +0.00] | -0.0002 [-0.0005, -0.0000] |
| trained | LAS | 60 | +0.00 [+0.00, +0.00] | -0.0013 [-0.0052, +0.0025] |
| trained | MEK | 60 | +1.67 [+0.00, +5.00] | +0.0030 [-0.0223, +0.0355] |
| control | ORD | 60 | +0.00 [+0.00, +0.00] | -0.0000 [-0.0000, +0.0000] |
| control | LAS | 60 | +0.00 [+0.00, +0.00] | +0.0000 [-0.0000, +0.0000] |
| control | MEK | 60 | +0.00 [+0.00, +0.00] | +0.0019 [+0.0001, +0.0054] |

### Chosen letters

Change in the share of answers choosing each letter, against the unedited model (percentage points). A change that only pushes the model toward one letter moves answers without learning anything.

| Set | Change | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | trained | +0.6 | -0.6 | +0.0 | +0.0 | +0.0 |
| test | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| gemma_extra | trained | +1.7 | +0.0 | +0.0 | -1.7 | +0.0 |
| gemma_extra | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| hidden | trained | +1.7 | -0.4 | +0.2 | -1.5 | +0.0 |
| hidden | control | +0.0 | +0.0 | -0.2 | +0.2 | +0.0 |
| validation | trained | +2.0 | -1.3 | +0.0 | -0.7 | +0.0 |
| validation | control | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |
| drill | trained | +1.1 | -0.1 | -0.1 | -0.8 | +0.0 |
| drill | control | -0.1 | +0.0 | +0.1 | +0.0 | +0.0 |

## 4-bit (Q4_K_M)

Wrong and right before the change: `experiments/group/gemma-4b-full/baseline/q4_hidden`.

### Fixed, kept and broken

| Change | Papers | Wrong fixed | Right kept | Right broken | Net (accuracy, CI) |
|---|---|---|---|---|---|
| trained | hidden | 2.3% (3/128) | 98.6% (347/352) | 5 | -2 questions, -0.4 pp (95% CI -1.7 to +0.8) |

### Trained minus random change

The key number is the bold row: does the trained change fix more unseen wrong answers than a random change of the same size? Counts are questions answered correctly.

| Questions | n | Trained | Random | Difference (CI) |
|---|---|---|---|---|
| (no random-change run at this precision) | | | | |

### Against the unedited model (bench.report)

Only the test split gives a verdict that counts; the others are indicative, and on sets without damage checks the damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. The decision rule does not apply to trained vs random, so no verdict is shown there.

| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | Right→wrong | Largest letter shift | Perplexity | Loop rate |
|---|---|---|---|---|---|---|---|---|---|
| test | trained vs unedited | NO IMPROVEMENT | +0.0082 [-0.0063, +0.0257] | +0.00 [-1.67, +1.67] | 1 | 1 | 1.1 pp | -0.40% | -2.0 points |
| gemma_extra | trained vs unedited | NO IMPROVEMENT (indicative) | +0.0383 [+0.0084, +0.0732] | +3.33 [+0.83, +6.67] | 4 | 0 | 1.7 pp | n/a | n/a |
| hidden | trained vs unedited | NO IMPROVEMENT (indicative) | -0.0004 [-0.0098, +0.0097] | -0.42 [-1.67, +0.62] | 3 | 5 | 1.0 pp | n/a | n/a |

### By section (hidden papers)

| Change | Section | Wrong fixed | Right kept | Broken |
|---|---|---|---|---|
| trained | ORD | 0.0% (0/28) | 99.2% (131/132) | 1 |
| trained | LAS | 2.1% (1/47) | 98.2% (111/113) | 2 |
| trained | MEK | 3.8% (2/53) | 98.1% (105/107) | 2 |

### By section (test split, against the unedited model)

| Change | Section | n | Accuracy change (pp, CI) | Metric change (CI) |
|---|---|---|---|---|
| trained | ORD | 60 | +0.00 [+0.00, +0.00] | +0.0015 [-0.0009, +0.0054] |
| trained | LAS | 60 | +0.00 [+0.00, +0.00] | +0.0098 [-0.0081, +0.0328] |
| trained | MEK | 60 | +0.00 [-5.00, +5.00] | +0.0066 [-0.0149, +0.0359] |

### Chosen letters

Change in the share of answers choosing each letter, against the unedited model (percentage points). A change that only pushes the model toward one letter moves answers without learning anything.

| Set | Change | A | B | C | D | E |
|---|---|---|---|---|---|---|
| test | trained | +0.6 | -0.6 | +1.1 | -1.1 | +0.0 |
| gemma_extra | trained | +1.7 | -1.7 | +0.8 | -0.8 | +0.0 |
| hidden | trained | +0.8 | -0.4 | +0.6 | -1.0 | +0.0 |
