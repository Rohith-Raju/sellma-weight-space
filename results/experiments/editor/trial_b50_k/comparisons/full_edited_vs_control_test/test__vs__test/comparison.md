# Comparison: test vs test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0088, 95% CI [-0.0170, +0.0376] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change +0.10% (limit +5%) |
| 3b_loop_rate | PASS | loop rate change -10.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 70.8% | 70.0% | +0.83 [-2.50, +4.17] | +0.0088 [-0.0170, +0.0376] | 3 | 2 |
| All questions | 180 | 76.7% | 76.1% | +0.56 [-1.67, +2.78] | +0.0058 [-0.0114, +0.0252] | 3 | 2 |
| ORD | 60 | 88.3% | 88.3% | +0.00 [+0.00, +0.00] | -0.0000 [-0.0000, +0.0000] | 0 | 0 |
| LAS | 60 | 70.0% | 68.3% | +1.67 [-3.33, +8.33] | +0.0122 [-0.0226, +0.0555] | 2 | 1 |
| MEK | 60 | 71.7% | 71.7% | +0.00 [-5.00, +5.00] | +0.0053 [-0.0329, +0.0498] | 1 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +4.2 (21.7%) | -1.7 (28.3%) | -0.8 (27.5%) | -1.7 (22.5%) | +0.0 (0.0%) |
| All questions | +2.8 (19.4%) | -1.1 (25.0%) | -0.6 (23.9%) | -1.1 (23.9%) | +0.0 (7.8%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (26.7%) | +0.0 (23.3%) |
| LAS | +5.0 (20.0%) | -1.7 (26.7%) | -1.7 (30.0%) | -1.7 (23.3%) | +0.0 (0.0%) |
| MEK | +3.3 (23.3%) | -1.7 (30.0%) | +0.0 (25.0%) | -1.7 (21.7%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 20.819 | 20.839 | +0.10% |
| Loop rate (50 prompts) | 18.0% | 8.0% | -10.0 points |
