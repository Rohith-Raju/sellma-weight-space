# Run summary: gemma-4b-full+editor[trial_b50_k]

Backend `hf`, split `dev`, 720 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 720 | 71.7% (68.3% to 74.8%) | 0.715 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 480 | 65.6% (61.3% to 69.7%) | 0.656 | 1.000 | 0.0% | no |
| ORD | 240 | 83.8% (78.6% to 87.9%) | 0.835 | 1.000 | 0.0% | no |
| LAS | 240 | 69.6% (63.5% to 75.1%) | 0.696 | 1.000 | 0.0% | no |
| MEK | 240 | 61.7% (55.4% to 67.6%) | 0.615 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.4% | 23.6% | 24.9% | 24.4% | 6.7% |
| Headline (LAS+MEK) | 21.2% | 25.4% | 27.3% | 26.0% | 0.0% |
| ORD | 18.8% | 20.0% | 20.0% | 21.2% | 20.0% |
| LAS | 20.4% | 23.8% | 30.0% | 25.8% | 0.0% |
| MEK | 22.1% | 27.1% | 24.6% | 26.2% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 75.0% | 0.758 | 0.0% |
| 2014-04-05 | 60 | 65.0% | 0.653 | 0.0% |
| 2014-10-25 | 60 | 70.0% | 0.700 | 0.0% |
| 2015-10-24 | 60 | 60.0% | 0.599 | 0.0% |
| 2016-10-29 | 60 | 81.7% | 0.800 | 0.0% |
| 2017-04-01 | 60 | 73.3% | 0.732 | 0.0% |
| 2019-04-06 | 60 | 71.7% | 0.717 | 0.0% |
| 2021-03-13 | 60 | 73.3% | 0.740 | 0.0% |
| 2021-10-24 | 60 | 56.7% | 0.564 | 0.0% |
| 2022-05-07 | 60 | 83.3% | 0.834 | 0.0% |
| 2023-03-25 | 60 | 68.3% | 0.679 | 0.0% |
| 2024-04-13 | 60 | 81.7% | 0.810 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
