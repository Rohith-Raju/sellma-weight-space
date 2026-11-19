# Run summary: gemma-4b-full+editor[trial_b25_mlp:random]

Backend `hf`, split `dev`, 720 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 720 | 71.1% (67.7% to 74.3%) | 0.712 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 480 | 65.0% (60.6% to 69.1%) | 0.651 | 1.000 | 0.0% | no |
| ORD | 240 | 83.3% (78.1% to 87.5%) | 0.834 | 0.999 | 0.0% | no |
| LAS | 240 | 69.2% (63.1% to 74.7%) | 0.695 | 1.000 | 0.0% | no |
| MEK | 240 | 60.8% (54.5% to 66.8%) | 0.608 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.2% | 26.8% | 25.4% | 24.7% | 6.8% |
| Headline (LAS+MEK) | 15.6% | 30.0% | 27.9% | 26.5% | 0.0% |
| ORD | 17.5% | 20.4% | 20.4% | 21.2% | 20.4% |
| LAS | 15.4% | 27.1% | 31.2% | 26.2% | 0.0% |
| MEK | 15.8% | 32.9% | 24.6% | 26.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 73.3% | 0.739 | 0.0% |
| 2014-04-05 | 60 | 65.0% | 0.643 | 0.0% |
| 2014-10-25 | 60 | 68.3% | 0.677 | 0.0% |
| 2015-10-24 | 60 | 58.3% | 0.591 | 0.0% |
| 2016-10-29 | 60 | 78.3% | 0.780 | 0.0% |
| 2017-04-01 | 60 | 76.7% | 0.760 | 0.0% |
| 2019-04-06 | 60 | 71.7% | 0.719 | 0.0% |
| 2021-03-13 | 60 | 70.0% | 0.704 | 0.0% |
| 2021-10-24 | 60 | 58.3% | 0.580 | 0.0% |
| 2022-05-07 | 60 | 85.0% | 0.859 | 0.0% |
| 2023-03-25 | 60 | 68.3% | 0.688 | 0.0% |
| 2024-04-13 | 60 | 80.0% | 0.807 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
