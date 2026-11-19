# Comparison: hidden vs hidden

Split `dev`, 480 questions, metric `prob_correct`. Headline: LAS+MEK. Control: hidden.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0052, 95% CI [-0.0165, +0.0060] (must be above zero) |
| 2_beats_control | FAIL | headline edited minus control -0.0053, 95% CI [-0.0168, +0.0061] (must be above zero) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 320 | 72.8% | 72.5% | +0.31 [-1.56, +2.19] | -0.0052 [-0.0165, +0.0060] | 5 | 4 |
| All questions | 480 | 76.0% | 75.6% | +0.42 [-0.83, +1.67] | -0.0024 [-0.0101, +0.0052] | 6 | 4 |
| ORD | 160 | 82.5% | 81.9% | +0.62 [+0.00, +1.88] | +0.0031 [-0.0007, +0.0093] | 1 | 0 |
| LAS | 160 | 75.0% | 73.8% | +1.25 [-1.88, +4.38] | +0.0002 [-0.0182, +0.0188] | 4 | 2 |
| MEK | 160 | 70.6% | 71.2% | -0.62 [-2.50, +1.25] | -0.0106 [-0.0239, +0.0008] | 1 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.9 (16.6%) | -0.6 (27.2%) | +0.3 (31.2%) | -1.6 (25.0%) | +0.0 (0.0%) |
| All questions | +1.7 (17.3%) | -0.4 (25.4%) | +0.2 (27.9%) | -1.5 (22.1%) | +0.0 (7.3%) |
| ORD | +1.3 (18.8%) | +0.0 (21.9%) | +0.0 (21.2%) | -1.2 (16.2%) | +0.0 (21.9%) |
| LAS | +3.1 (18.1%) | +0.6 (25.6%) | -1.3 (34.4%) | -2.5 (21.9%) | +0.0 (0.0%) |
| MEK | +0.6 (15.0%) | -1.9 (28.7%) | +1.9 (28.1%) | -0.6 (28.1%) | +0.0 (0.0%) |

## Control

Headline edited minus control on `prob_correct`: -0.0053 [-0.0168, +0.0061].

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
