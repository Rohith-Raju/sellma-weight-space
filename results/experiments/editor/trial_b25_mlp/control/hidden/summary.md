# Run summary: gemma-4b-full+editor[trial_b25_mlp:random]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 75.4% (71.4% to 79.1%) | 0.753 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 72.2% (67.0% to 76.8%) | 0.722 | 1.000 | 0.0% | no |
| ORD | 160 | 81.9% (75.2% to 87.1%) | 0.815 | 1.000 | 0.0% | no |
| LAS | 160 | 75.0% (67.8% to 81.1%) | 0.747 | 1.000 | 0.0% | no |
| MEK | 160 | 69.4% (61.8% to 76.0%) | 0.698 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.6% | 27.7% | 28.5% | 21.0% | 7.1% |
| Headline (LAS+MEK) | 14.4% | 30.3% | 32.2% | 23.1% | 0.0% |
| ORD | 18.1% | 22.5% | 21.2% | 16.9% | 21.2% |
| LAS | 14.4% | 25.0% | 36.2% | 24.4% | 0.0% |
| MEK | 14.4% | 35.6% | 28.1% | 21.9% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 76.7% | 0.763 | 0.0% |
| 2016-04-09 | 60 | 71.7% | 0.720 | 0.0% |
| 2017-10-21 | 60 | 76.7% | 0.758 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.833 | 0.0% |
| 2019-10-20 | 60 | 70.0% | 0.704 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.768 | 0.0% |
| 2021-05-08 | 60 | 71.7% | 0.713 | 0.0% |
| 2022-10-23 | 60 | 76.7% | 0.767 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
