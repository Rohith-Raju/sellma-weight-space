# Run summary: gemma-4b-full+editor[trial_b10_q_half:random]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 75.6% (71.6% to 79.3%) | 0.757 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 72.5% (67.4% to 77.1%) | 0.725 | 1.000 | 0.0% | no |
| ORD | 160 | 81.9% (75.2% to 87.1%) | 0.820 | 1.000 | 0.0% | no |
| LAS | 160 | 73.8% (66.4% to 80.0%) | 0.738 | 1.000 | 0.0% | no |
| MEK | 160 | 71.2% (63.8% to 77.7%) | 0.711 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.6% | 25.8% | 27.7% | 23.5% | 7.3% |
| Headline (LAS+MEK) | 14.7% | 27.8% | 30.9% | 26.6% | 0.0% |
| ORD | 17.5% | 21.9% | 21.2% | 17.5% | 21.9% |
| LAS | 15.0% | 25.0% | 35.6% | 24.4% | 0.0% |
| MEK | 14.4% | 30.6% | 26.2% | 28.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 75.0% | 0.752 | 0.0% |
| 2016-04-09 | 60 | 70.0% | 0.705 | 0.0% |
| 2017-10-21 | 60 | 76.7% | 0.770 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.833 | 0.0% |
| 2019-10-20 | 60 | 73.3% | 0.729 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.769 | 0.0% |
| 2021-05-08 | 60 | 70.0% | 0.708 | 0.0% |
| 2022-10-23 | 60 | 80.0% | 0.788 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
