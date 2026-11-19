# Run summary: gemma-4b-full+editor[trial_b10_q_full]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 74.2% (70.1% to 77.9%) | 0.741 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 70.3% (65.1% to 75.1%) | 0.702 | 1.000 | 0.0% | no |
| ORD | 160 | 81.9% (75.2% to 87.1%) | 0.817 | 1.000 | 0.0% | no |
| LAS | 160 | 75.0% (67.8% to 81.1%) | 0.749 | 1.000 | 0.0% | no |
| MEK | 160 | 65.6% (58.0% to 72.5%) | 0.656 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.7% | 31.0% | 28.3% | 17.3% | 6.7% |
| Headline (LAS+MEK) | 15.9% | 34.7% | 31.6% | 17.8% | 0.0% |
| ORD | 18.1% | 23.8% | 21.9% | 16.2% | 20.0% |
| LAS | 16.9% | 28.1% | 35.0% | 20.0% | 0.0% |
| MEK | 15.0% | 41.2% | 28.1% | 15.6% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 73.3% | 0.734 | 0.0% |
| 2016-04-09 | 60 | 68.3% | 0.680 | 0.0% |
| 2017-10-21 | 60 | 71.7% | 0.717 | 0.0% |
| 2018-10-21 | 60 | 85.0% | 0.852 | 0.0% |
| 2019-10-20 | 60 | 68.3% | 0.685 | 0.0% |
| 2020-10-25 | 60 | 78.3% | 0.776 | 0.0% |
| 2021-05-08 | 60 | 68.3% | 0.683 | 0.0% |
| 2022-10-23 | 60 | 80.0% | 0.799 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
