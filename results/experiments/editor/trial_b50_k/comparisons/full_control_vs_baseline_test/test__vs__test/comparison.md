# Comparison: test vs test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | PASS | headline change in prob_correct +0.0005, 95% CI [+0.0000, +0.0014] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -0.05% (limit +5%) |
| 3b_loop_rate | FAIL | loop rate change +8.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 70.0% | 70.0% | +0.00 [+0.00, +0.00] | +0.0005 [+0.0000, +0.0014] | 0 | 0 |
| All questions | 180 | 76.1% | 76.1% | +0.00 [+0.00, +0.00] | +0.0003 [+0.0000, +0.0009] | 0 | 0 |
| ORD | 60 | 88.3% | 88.3% | +0.00 [+0.00, +0.00] | -0.0000 [-0.0000, +0.0000] | 0 | 0 |
| LAS | 60 | 68.3% | 68.3% | +0.00 [+0.00, +0.00] | +0.0000 [-0.0000, +0.0000] | 0 | 0 |
| MEK | 60 | 71.7% | 71.7% | +0.00 [+0.00, +0.00] | +0.0009 [+0.0000, +0.0027] | 0 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (17.5%) | +0.0 (30.0%) | +0.0 (28.3%) | +0.0 (24.2%) | +0.0 (0.0%) |
| All questions | +0.0 (16.7%) | +0.0 (26.1%) | +0.0 (24.4%) | +0.0 (25.0%) | +0.0 (7.8%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (26.7%) | +0.0 (23.3%) |
| LAS | +0.0 (15.0%) | +0.0 (28.3%) | +0.0 (31.7%) | +0.0 (25.0%) | +0.0 (0.0%) |
| MEK | +0.0 (20.0%) | +0.0 (31.7%) | +0.0 (25.0%) | +0.0 (23.3%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 20.828 | 20.819 | -0.05% |
| Loop rate (50 prompts) | 10.0% | 18.0% | +8.0 points |
