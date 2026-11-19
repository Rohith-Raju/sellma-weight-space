# Run summary: gemma-4b-q4+editor[trial_b10_q_full]

Backend `llamacpp`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 70.8% (66.6% to 74.7%) | 0.710 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 65.0% (59.6% to 70.0%) | 0.653 | 1.000 | 0.0% | no |
| ORD | 160 | 82.5% (75.9% to 87.6%) | 0.824 | 1.000 | 0.0% | no |
| LAS | 160 | 67.5% (59.9% to 74.3%) | 0.672 | 1.000 | 0.0% | no |
| MEK | 160 | 62.5% (54.8% to 69.6%) | 0.634 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.0% | 31.5% | 30.4% | 15.8% | 7.3% |
| Headline (LAS+MEK) | 13.1% | 36.9% | 34.7% | 15.3% | 0.0% |
| ORD | 18.8% | 20.6% | 21.9% | 16.9% | 21.9% |
| LAS | 14.4% | 29.4% | 38.8% | 17.5% | 0.0% |
| MEK | 11.9% | 44.4% | 30.6% | 13.1% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 70.0% | 0.708 | 0.0% |
| 2016-04-09 | 60 | 68.3% | 0.680 | 0.0% |
| 2017-10-21 | 60 | 66.7% | 0.660 | 0.0% |
| 2018-10-21 | 60 | 80.0% | 0.805 | 0.0% |
| 2019-10-20 | 60 | 70.0% | 0.706 | 0.0% |
| 2020-10-25 | 60 | 70.0% | 0.702 | 0.0% |
| 2021-05-08 | 60 | 63.3% | 0.636 | 0.0% |
| 2022-10-23 | 60 | 78.3% | 0.782 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
