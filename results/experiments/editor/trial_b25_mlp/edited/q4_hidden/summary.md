# Run summary: gemma-4b-q4+editor[trial_b25_mlp]

Backend `llamacpp`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 70.8% (66.6% to 74.7%) | 0.706 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 65.6% (60.3% to 70.6%) | 0.655 | 1.000 | 0.0% | no |
| ORD | 160 | 81.2% (74.5% to 86.5%) | 0.809 | 1.000 | 0.0% | no |
| LAS | 160 | 65.6% (58.0% to 72.5%) | 0.654 | 1.000 | 0.0% | no |
| MEK | 160 | 65.6% (58.0% to 72.5%) | 0.656 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 14.0% | 29.0% | 29.8% | 19.4% | 7.9% |
| Headline (LAS+MEK) | 12.2% | 32.8% | 34.1% | 20.9% | 0.0% |
| ORD | 17.5% | 21.2% | 21.2% | 16.2% | 23.8% |
| LAS | 13.8% | 26.2% | 38.1% | 21.9% | 0.0% |
| MEK | 10.6% | 39.4% | 30.0% | 20.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 75.0% | 0.746 | 0.0% |
| 2016-04-09 | 60 | 71.7% | 0.724 | 0.0% |
| 2017-10-21 | 60 | 61.7% | 0.616 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.833 | 0.0% |
| 2019-10-20 | 60 | 68.3% | 0.678 | 0.0% |
| 2020-10-25 | 60 | 73.3% | 0.723 | 0.0% |
| 2021-05-08 | 60 | 61.7% | 0.606 | 0.0% |
| 2022-10-23 | 60 | 71.7% | 0.725 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
