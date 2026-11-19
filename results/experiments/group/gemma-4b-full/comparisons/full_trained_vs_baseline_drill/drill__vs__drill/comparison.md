# Comparison: drill vs drill

Split `dev`, 720 questions, metric `prob_correct`. Headline: LAS+MEK. Control: drill.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0023, 95% CI [-0.0112, +0.0061] (must be above zero) |
| 2_beats_control | FAIL | headline edited minus control -0.0024, 95% CI [-0.0113, +0.0062] (must be above zero) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 480 | 65.2% | 65.6% | -0.42 [-1.67, +0.83] | -0.0023 [-0.0112, +0.0061] | 4 | 6 |
| All questions | 720 | 71.2% | 71.5% | -0.28 [-1.11, +0.56] | -0.0021 [-0.0078, +0.0035] | 4 | 6 |
| ORD | 240 | 83.3% | 83.3% | +0.00 [+0.00, +0.00] | -0.0016 [-0.0048, +0.0015] | 0 | 0 |
| LAS | 240 | 70.0% | 71.2% | -1.25 [-3.33, +0.42] | -0.0072 [-0.0215, +0.0064] | 1 | 4 |
| MEK | 240 | 60.4% | 60.0% | +0.42 [-1.25, +2.08] | +0.0025 [-0.0080, +0.0122] | 3 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.7 (17.3%) | -0.2 (28.5%) | -0.2 (27.3%) | -1.3 (26.9%) | +0.0 (0.0%) |
| All questions | +1.1 (17.5%) | -0.1 (25.7%) | -0.1 (25.0%) | -0.8 (25.1%) | +0.0 (6.7%) |
| ORD | +0.0 (17.9%) | +0.0 (20.0%) | +0.0 (20.4%) | +0.0 (21.7%) | +0.0 (20.0%) |
| LAS | +2.1 (17.5%) | +0.0 (27.9%) | +0.0 (29.6%) | -2.1 (25.0%) | +0.0 (0.0%) |
| MEK | +1.3 (17.1%) | -0.4 (29.2%) | -0.4 (25.0%) | -0.4 (28.7%) | +0.0 (0.0%) |

## Control

Headline edited minus control on `prob_correct`: -0.0024 [-0.0113, +0.0062].

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
