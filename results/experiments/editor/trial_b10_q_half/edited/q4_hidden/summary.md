# Run summary: gemma-4b-q4+editor[trial_b10_q_half]

Backend `llamacpp`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 72.3% (68.1% to 76.1%) | 0.725 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 67.5% (62.2% to 72.4%) | 0.677 | 1.000 | 0.0% | no |
| ORD | 160 | 81.9% (75.2% to 87.1%) | 0.820 | 1.000 | 0.0% | no |
| LAS | 160 | 68.1% (60.6% to 74.8%) | 0.687 | 1.000 | 0.0% | no |
| MEK | 160 | 66.9% (59.3% to 73.7%) | 0.667 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 14.4% | 29.2% | 30.8% | 18.1% | 7.5% |
| Headline (LAS+MEK) | 12.8% | 33.8% | 35.0% | 18.4% | 0.0% |
| ORD | 17.5% | 20.0% | 22.5% | 17.5% | 22.5% |
| LAS | 14.4% | 26.9% | 38.1% | 20.6% | 0.0% |
| MEK | 11.2% | 40.6% | 31.9% | 16.2% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 71.7% | 0.718 | 0.0% |
| 2016-04-09 | 60 | 70.0% | 0.706 | 0.0% |
| 2017-10-21 | 60 | 66.7% | 0.661 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.833 | 0.0% |
| 2019-10-20 | 60 | 71.7% | 0.731 | 0.0% |
| 2020-10-25 | 60 | 75.0% | 0.742 | 0.0% |
| 2021-05-08 | 60 | 61.7% | 0.620 | 0.0% |
| 2022-10-23 | 60 | 78.3% | 0.787 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
