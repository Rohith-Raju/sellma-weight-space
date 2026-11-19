# Run summary: gemma-4b-q4+group[trained]

Backend `llamacpp`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 72.9% (68.8% to 76.7%) | 0.728 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 68.4% (63.2% to 73.3%) | 0.681 | 1.000 | 0.0% | no |
| ORD | 160 | 81.9% (75.2% to 87.1%) | 0.821 | 1.000 | 0.0% | no |
| LAS | 160 | 70.0% (62.5% to 76.6%) | 0.694 | 1.000 | 0.0% | no |
| MEK | 160 | 66.9% (59.3% to 73.7%) | 0.669 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 14.6% | 26.5% | 31.0% | 20.4% | 7.5% |
| Headline (LAS+MEK) | 13.1% | 29.7% | 35.3% | 21.9% | 0.0% |
| ORD | 17.5% | 20.0% | 22.5% | 17.5% | 22.5% |
| LAS | 15.0% | 22.5% | 39.4% | 23.1% | 0.0% |
| MEK | 11.2% | 36.9% | 31.2% | 20.6% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 73.3% | 0.733 | 0.0% |
| 2016-04-09 | 60 | 71.7% | 0.722 | 0.0% |
| 2017-10-21 | 60 | 68.3% | 0.666 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.824 | 0.0% |
| 2019-10-20 | 60 | 73.3% | 0.731 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.763 | 0.0% |
| 2021-05-08 | 60 | 63.3% | 0.634 | 0.0% |
| 2022-10-23 | 60 | 73.3% | 0.749 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
