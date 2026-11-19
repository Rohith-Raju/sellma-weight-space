# Comparison: drill vs drill

Split `dev`, 720 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0001, 95% CI [-0.0005, +0.0006] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 480 | 65.4% | 65.6% | -0.21 [-0.62, +0.00] | +0.0001 [-0.0005, +0.0006] | 0 | 1 |
| All questions | 720 | 71.4% | 71.5% | -0.14 [-0.42, +0.00] | +0.0002 [-0.0002, +0.0006] | 0 | 1 |
| ORD | 240 | 83.3% | 83.3% | +0.00 [+0.00, +0.00] | +0.0005 [-0.0000, +0.0012] | 0 | 0 |
| LAS | 240 | 70.8% | 71.2% | -0.42 [-1.25, +0.00] | +0.0001 [-0.0008, +0.0011] | 0 | 1 |
| MEK | 240 | 60.0% | 60.0% | +0.00 [+0.00, +0.00] | +0.0000 [-0.0006, +0.0006] | 0 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | -0.2 (15.4%) | +0.0 (28.7%) | +0.2 (27.7%) | +0.0 (28.1%) | +0.0 (0.0%) |
| All questions | -0.1 (16.2%) | +0.0 (25.8%) | +0.1 (25.3%) | +0.0 (26.0%) | +0.0 (6.7%) |
| ORD | +0.0 (17.9%) | +0.0 (20.0%) | +0.0 (20.4%) | +0.0 (21.7%) | +0.0 (20.0%) |
| LAS | -0.4 (15.0%) | +0.0 (27.9%) | +0.4 (30.0%) | +0.0 (27.1%) | +0.0 (0.0%) |
| MEK | +0.0 (15.8%) | +0.0 (29.6%) | +0.0 (25.4%) | +0.0 (29.2%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
