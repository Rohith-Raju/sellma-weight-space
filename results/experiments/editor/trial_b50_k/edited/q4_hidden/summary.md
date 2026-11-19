# Run summary: gemma-4b-q4+editor[trial_b50_k]

Backend `llamacpp`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 74.2% (70.1% to 77.9%) | 0.738 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 70.0% (64.8% to 74.8%) | 0.696 | 1.000 | 0.0% | no |
| ORD | 160 | 82.5% (75.9% to 87.6%) | 0.823 | 1.000 | 0.0% | no |
| LAS | 160 | 71.9% (64.5% to 78.3%) | 0.714 | 1.000 | 0.0% | no |
| MEK | 160 | 68.1% (60.6% to 74.8%) | 0.678 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.2% | 26.2% | 29.8% | 20.2% | 7.5% |
| Headline (LAS+MEK) | 14.7% | 29.4% | 34.4% | 21.6% | 0.0% |
| ORD | 19.4% | 20.0% | 20.6% | 17.5% | 22.5% |
| LAS | 15.6% | 21.2% | 38.8% | 24.4% | 0.0% |
| MEK | 13.8% | 37.5% | 30.0% | 18.8% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 73.3% | 0.728 | 0.0% |
| 2016-04-09 | 60 | 75.0% | 0.751 | 0.0% |
| 2017-10-21 | 60 | 68.3% | 0.681 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.822 | 0.0% |
| 2019-10-20 | 60 | 76.7% | 0.756 | 0.0% |
| 2020-10-25 | 60 | 75.0% | 0.751 | 0.0% |
| 2021-05-08 | 60 | 65.0% | 0.649 | 0.0% |
| 2022-10-23 | 60 | 76.7% | 0.767 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
