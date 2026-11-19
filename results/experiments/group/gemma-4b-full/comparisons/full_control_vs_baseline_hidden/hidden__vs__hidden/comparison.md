# Comparison: hidden vs hidden

Split `dev`, 480 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0001, 95% CI [-0.0006, +0.0011] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 320 | 72.8% | 72.5% | +0.31 [+0.00, +0.94] | +0.0001 [-0.0006, +0.0011] | 1 | 0 |
| All questions | 480 | 75.8% | 75.6% | +0.21 [+0.00, +0.62] | +0.0001 [-0.0004, +0.0007] | 1 | 0 |
| ORD | 160 | 81.9% | 81.9% | +0.00 [+0.00, +0.00] | -0.0000 [-0.0001, +0.0000] | 0 | 0 |
| LAS | 160 | 73.8% | 73.8% | +0.00 [+0.00, +0.00] | -0.0002 [-0.0006, +0.0000] | 0 | 0 |
| MEK | 160 | 71.9% | 71.2% | +0.62 [+0.00, +1.88] | +0.0004 [-0.0008, +0.0023] | 1 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (14.7%) | +0.0 (27.8%) | -0.3 (30.6%) | +0.3 (26.9%) | +0.0 (0.0%) |
| All questions | +0.0 (15.6%) | +0.0 (25.8%) | -0.2 (27.5%) | +0.2 (23.8%) | +0.0 (7.3%) |
| ORD | +0.0 (17.5%) | +0.0 (21.9%) | +0.0 (21.2%) | +0.0 (17.5%) | +0.0 (21.9%) |
| LAS | +0.0 (15.0%) | +0.0 (25.0%) | +0.0 (35.6%) | +0.0 (24.4%) | +0.0 (0.0%) |
| MEK | +0.0 (14.4%) | +0.0 (30.6%) | -0.6 (25.6%) | +0.6 (29.4%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
