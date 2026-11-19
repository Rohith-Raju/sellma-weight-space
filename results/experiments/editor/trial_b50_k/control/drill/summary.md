# Run summary: gemma-4b-full+editor[trial_b50_k:random]

Backend `hf`, split `dev`, 720 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 720 | 71.5% (68.1% to 74.7%) | 0.713 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 480 | 65.6% (61.3% to 69.7%) | 0.654 | 1.000 | 0.0% | no |
| ORD | 240 | 83.3% (78.1% to 87.5%) | 0.831 | 1.000 | 0.0% | no |
| LAS | 240 | 71.2% (65.2% to 76.6%) | 0.707 | 1.000 | 0.0% | no |
| MEK | 240 | 60.0% (53.7% to 66.0%) | 0.601 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.4% | 25.8% | 25.3% | 25.8% | 6.7% |
| Headline (LAS+MEK) | 15.6% | 28.7% | 27.7% | 27.9% | 0.0% |
| ORD | 17.9% | 20.0% | 20.4% | 21.7% | 20.0% |
| LAS | 15.4% | 27.9% | 29.6% | 27.1% | 0.0% |
| MEK | 15.8% | 29.6% | 25.8% | 28.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 73.3% | 0.734 | 0.0% |
| 2014-04-05 | 60 | 66.7% | 0.666 | 0.0% |
| 2014-10-25 | 60 | 66.7% | 0.672 | 0.0% |
| 2015-10-24 | 60 | 60.0% | 0.583 | 0.0% |
| 2016-10-29 | 60 | 80.0% | 0.788 | 0.0% |
| 2017-04-01 | 60 | 76.7% | 0.755 | 0.0% |
| 2019-04-06 | 60 | 71.7% | 0.718 | 0.0% |
| 2021-03-13 | 60 | 71.7% | 0.717 | 0.0% |
| 2021-10-24 | 60 | 56.7% | 0.573 | 0.0% |
| 2022-05-07 | 60 | 86.7% | 0.866 | 0.0% |
| 2023-03-25 | 60 | 66.7% | 0.671 | 0.0% |
| 2024-04-13 | 60 | 81.7% | 0.814 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
