# Run summary: gemma-4b-q4

Backend `llamacpp`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 73.3% (69.2% to 77.1%) | 0.729 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 68.8% (63.5% to 73.6%) | 0.682 | 1.000 | 0.0% | no |
| ORD | 160 | 82.5% (75.9% to 87.6%) | 0.825 | 1.000 | 0.0% | no |
| LAS | 160 | 70.6% (63.2% to 77.1%) | 0.697 | 1.000 | 0.0% | no |
| MEK | 160 | 66.9% (59.3% to 73.7%) | 0.667 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 13.8% | 26.9% | 30.4% | 21.5% | 7.5% |
| Headline (LAS+MEK) | 11.9% | 30.3% | 34.7% | 23.1% | 0.0% |
| ORD | 17.5% | 20.0% | 21.9% | 18.1% | 22.5% |
| LAS | 14.4% | 22.5% | 38.8% | 24.4% | 0.0% |
| MEK | 9.4% | 38.1% | 30.6% | 21.9% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 73.3% | 0.729 | 0.0% |
| 2016-04-09 | 60 | 73.3% | 0.725 | 0.0% |
| 2017-10-21 | 60 | 66.7% | 0.660 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.828 | 0.0% |
| 2019-10-20 | 60 | 75.0% | 0.746 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.763 | 0.0% |
| 2021-05-08 | 60 | 63.3% | 0.634 | 0.0% |
| 2022-10-23 | 60 | 75.0% | 0.750 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
