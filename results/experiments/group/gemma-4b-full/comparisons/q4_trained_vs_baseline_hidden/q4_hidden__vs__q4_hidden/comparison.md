# Comparison: q4_hidden vs q4_hidden

Split `dev`, 480 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0004, 95% CI [-0.0098, +0.0097] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 320 | 68.4% | 68.8% | -0.31 [-1.88, +1.25] | -0.0004 [-0.0098, +0.0097] | 3 | 4 |
| All questions | 480 | 72.9% | 73.3% | -0.42 [-1.67, +0.62] | -0.0017 [-0.0085, +0.0051] | 3 | 5 |
| ORD | 160 | 81.9% | 82.5% | -0.62 [-1.88, +0.00] | -0.0044 [-0.0115, +0.0003] | 0 | 1 |
| LAS | 160 | 70.0% | 70.6% | -0.62 [-3.12, +1.25] | -0.0027 [-0.0131, +0.0081] | 1 | 2 |
| MEK | 160 | 66.9% | 66.9% | +0.00 [-2.50, +2.50] | +0.0019 [-0.0137, +0.0190] | 2 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.3 (13.1%) | -0.6 (29.7%) | +0.6 (35.3%) | -1.3 (21.9%) | +0.0 (0.0%) |
| All questions | +0.8 (14.6%) | -0.4 (26.5%) | +0.6 (31.0%) | -1.0 (20.4%) | +0.0 (7.5%) |
| ORD | +0.0 (17.5%) | +0.0 (20.0%) | +0.6 (22.5%) | -0.6 (17.5%) | +0.0 (22.5%) |
| LAS | +0.6 (15.0%) | +0.0 (22.5%) | +0.6 (39.4%) | -1.2 (23.1%) | +0.0 (0.0%) |
| MEK | +1.9 (11.2%) | -1.2 (36.9%) | +0.6 (31.2%) | -1.3 (20.6%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
