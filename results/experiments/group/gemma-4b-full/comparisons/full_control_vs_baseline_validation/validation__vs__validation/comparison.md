# Comparison: validation vs validation

Split `dev`, 150 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0002, 95% CI [-0.0014, +0.0008] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 100 | 68.0% | 68.0% | +0.00 [+0.00, +0.00] | -0.0002 [-0.0014, +0.0008] | 0 | 0 |
| All questions | 150 | 72.7% | 72.7% | +0.00 [+0.00, +0.00] | +0.0004 [-0.0007, +0.0017] | 0 | 0 |
| ORD | 50 | 82.0% | 82.0% | +0.00 [+0.00, +0.00] | +0.0016 [-0.0000, +0.0047] | 0 | 0 |
| LAS | 50 | 66.0% | 66.0% | +0.00 [+0.00, +0.00] | +0.0006 [-0.0000, +0.0017] | 0 | 0 |
| MEK | 50 | 70.0% | 70.0% | +0.00 [+0.00, +0.00] | -0.0010 [-0.0028, +0.0000] | 0 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (18.0%) | +0.0 (25.0%) | +0.0 (22.0%) | +0.0 (35.0%) | +0.0 (0.0%) |
| All questions | +0.0 (19.3%) | +0.0 (22.7%) | +0.0 (24.0%) | +0.0 (28.7%) | +0.0 (5.3%) |
| ORD | +0.0 (22.0%) | +0.0 (18.0%) | +0.0 (28.0%) | +0.0 (16.0%) | +0.0 (16.0%) |
| LAS | +0.0 (16.0%) | +0.0 (18.0%) | +0.0 (28.0%) | +0.0 (38.0%) | +0.0 (0.0%) |
| MEK | +0.0 (20.0%) | +0.0 (32.0%) | +0.0 (16.0%) | +0.0 (32.0%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
