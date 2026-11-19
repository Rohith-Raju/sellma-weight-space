# Comparison: hidden vs hidden

Split `dev`, 480 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0053, 95% CI [-0.0168, +0.0061] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 320 | 72.8% | 72.8% | +0.00 [-1.88, +1.88] | -0.0053 [-0.0168, +0.0061] | 5 | 5 |
| All questions | 480 | 76.0% | 75.8% | +0.21 [-1.04, +1.46] | -0.0025 [-0.0103, +0.0053] | 6 | 5 |
| ORD | 160 | 82.5% | 81.9% | +0.62 [+0.00, +1.88] | +0.0032 [-0.0007, +0.0093] | 1 | 0 |
| LAS | 160 | 75.0% | 73.8% | +1.25 [-1.88, +4.38] | +0.0004 [-0.0181, +0.0191] | 4 | 2 |
| MEK | 160 | 70.6% | 71.9% | -1.25 [-3.75, +1.25] | -0.0110 [-0.0248, +0.0007] | 1 | 3 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.9 (16.6%) | -0.6 (27.2%) | +0.6 (31.2%) | -1.9 (25.0%) | +0.0 (0.0%) |
| All questions | +1.7 (17.3%) | -0.4 (25.4%) | +0.4 (27.9%) | -1.7 (22.1%) | +0.0 (7.3%) |
| ORD | +1.3 (18.8%) | +0.0 (21.9%) | +0.0 (21.2%) | -1.2 (16.2%) | +0.0 (21.9%) |
| LAS | +3.1 (18.1%) | +0.6 (25.6%) | -1.3 (34.4%) | -2.5 (21.9%) | +0.0 (0.0%) |
| MEK | +0.6 (15.0%) | -1.9 (28.7%) | +2.5 (28.1%) | -1.3 (28.1%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
