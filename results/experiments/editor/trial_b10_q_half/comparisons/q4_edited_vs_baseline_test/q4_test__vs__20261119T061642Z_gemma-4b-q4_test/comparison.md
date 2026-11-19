# Comparison: q4_test vs 20261119T061642Z_gemma-4b-q4_test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0009, 95% CI [-0.0377, +0.0374] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change +0.01% (limit +5%) |
| 3b_loop_rate | FAIL | loop rate change +4.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 65.8% | 65.8% | +0.00 [-4.17, +4.17] | -0.0009 [-0.0377, +0.0374] | 4 | 4 |
| All questions | 180 | 73.9% | 73.9% | +0.00 [-2.78, +3.33] | -0.0005 [-0.0254, +0.0250] | 4 | 4 |
| ORD | 60 | 90.0% | 90.0% | +0.00 [+0.00, +0.00] | +0.0002 [-0.0004, +0.0012] | 0 | 0 |
| LAS | 60 | 66.7% | 63.3% | +3.33 [+0.00, +8.33] | +0.0310 [-0.0018, +0.0791] | 2 | 0 |
| MEK | 60 | 65.0% | 68.3% | -3.33 [-11.67, +5.00] | -0.0327 [-0.0959, +0.0256] | 2 | 4 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.7 (18.3%) | +2.5 (32.5%) | +2.5 (29.2%) | -6.7 (20.0%) | +0.0 (0.0%) |
| All questions | +1.1 (17.2%) | +1.7 (27.8%) | +2.2 (25.6%) | -4.4 (20.6%) | -0.6 (8.9%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +1.7 (18.3%) | +0.0 (21.7%) | -1.7 (26.7%) |
| LAS | +1.7 (15.0%) | +0.0 (26.7%) | +3.3 (35.0%) | -5.0 (23.3%) | +0.0 (0.0%) |
| MEK | +1.7 (21.7%) | +5.0 (38.3%) | +1.7 (23.3%) | -8.3 (16.7%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 15.100 | 15.101 | +0.01% |
| Loop rate (50 prompts) | 20.0% | 24.0% | +4.0 points |
