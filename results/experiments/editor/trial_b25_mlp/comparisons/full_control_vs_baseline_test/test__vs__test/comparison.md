# Comparison: test vs test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0219, 95% CI [-0.0504, -0.0001] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -0.24% (limit +5%) |
| 3b_loop_rate | PASS | loop rate change +0.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 67.5% | 70.0% | -2.50 [-5.83, +0.00] | -0.0219 [-0.0504, -0.0001] | 0 | 3 |
| All questions | 180 | 74.4% | 76.1% | -1.67 [-3.89, +0.00] | -0.0146 [-0.0329, -0.0000] | 0 | 3 |
| ORD | 60 | 88.3% | 88.3% | +0.00 [+0.00, +0.00] | +0.0000 [-0.0000, +0.0000] | 0 | 0 |
| LAS | 60 | 66.7% | 68.3% | -1.67 [-5.00, +0.00] | -0.0166 [-0.0496, +0.0000] | 0 | 1 |
| MEK | 60 | 68.3% | 71.7% | -3.33 [-8.33, +0.00] | -0.0273 [-0.0731, +0.0052] | 0 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | -0.8 (16.7%) | +1.7 (31.7%) | -0.8 (27.5%) | +0.0 (24.2%) | +0.0 (0.0%) |
| All questions | -0.6 (16.1%) | +1.1 (27.2%) | -0.6 (23.9%) | +0.0 (25.0%) | +0.0 (7.8%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (26.7%) | +0.0 (23.3%) |
| LAS | -1.7 (13.3%) | +0.0 (28.3%) | +0.0 (31.7%) | +1.7 (26.7%) | +0.0 (0.0%) |
| MEK | +0.0 (20.0%) | +3.3 (35.0%) | -1.7 (23.3%) | -1.7 (21.7%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 20.828 | 20.778 | -0.24% |
| Loop rate (50 prompts) | 10.0% | 10.0% | +0.0 points |
