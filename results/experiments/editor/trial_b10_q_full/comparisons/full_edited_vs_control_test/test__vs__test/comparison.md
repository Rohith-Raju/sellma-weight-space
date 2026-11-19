# Comparison: test vs test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0078, 95% CI [-0.0357, +0.0156] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -0.00% (limit +5%) |
| 3b_loop_rate | PASS | loop rate change -2.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 69.2% | 70.0% | -0.83 [-4.17, +1.67] | -0.0078 [-0.0357, +0.0156] | 1 | 2 |
| All questions | 180 | 75.6% | 76.1% | -0.56 [-2.22, +1.11] | -0.0052 [-0.0241, +0.0102] | 1 | 2 |
| ORD | 60 | 88.3% | 88.3% | +0.00 [+0.00, +0.00] | +0.0000 [-0.0000, +0.0000] | 0 | 0 |
| LAS | 60 | 70.0% | 68.3% | +1.67 [+0.00, +5.00] | +0.0126 [-0.0001, +0.0365] | 1 | 0 |
| MEK | 60 | 68.3% | 71.7% | -3.33 [-8.33, +0.00] | -0.0281 [-0.0815, +0.0083] | 0 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.7 (19.2%) | +1.7 (31.7%) | +0.0 (28.3%) | -3.3 (20.8%) | +0.0 (0.0%) |
| All questions | +1.1 (17.8%) | +1.1 (27.2%) | +0.0 (24.4%) | -2.2 (22.8%) | +0.0 (7.8%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (26.7%) | +0.0 (23.3%) |
| LAS | +1.7 (16.7%) | +0.0 (28.3%) | +0.0 (31.7%) | -1.7 (23.3%) | +0.0 (0.0%) |
| MEK | +1.7 (21.7%) | +3.3 (35.0%) | +0.0 (25.0%) | -5.0 (18.3%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 20.822 | 20.821 | -0.00% |
| Loop rate (50 prompts) | 16.0% | 14.0% | -2.0 points |
