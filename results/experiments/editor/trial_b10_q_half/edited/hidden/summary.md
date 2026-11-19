# Run summary: gemma-4b-full+editor[trial_b10_q_half]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 74.6% (70.5% to 78.3%) | 0.748 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 70.9% (65.7% to 75.6%) | 0.714 | 1.000 | 0.0% | no |
| ORD | 160 | 81.9% (75.2% to 87.1%) | 0.818 | 1.000 | 0.0% | no |
| LAS | 160 | 73.8% (66.4% to 80.0%) | 0.743 | 1.000 | 0.0% | no |
| MEK | 160 | 68.1% (60.6% to 74.8%) | 0.685 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.5% | 29.8% | 27.5% | 19.2% | 7.1% |
| Headline (LAS+MEK) | 15.6% | 33.1% | 30.6% | 20.6% | 0.0% |
| ORD | 18.1% | 23.1% | 21.2% | 16.2% | 21.2% |
| LAS | 16.2% | 28.1% | 33.8% | 21.9% | 0.0% |
| MEK | 15.0% | 38.1% | 27.5% | 19.4% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 73.3% | 0.743 | 0.0% |
| 2016-04-09 | 60 | 71.7% | 0.714 | 0.0% |
| 2017-10-21 | 60 | 75.0% | 0.751 | 0.0% |
| 2018-10-21 | 60 | 85.0% | 0.844 | 0.0% |
| 2019-10-20 | 60 | 68.3% | 0.692 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.768 | 0.0% |
| 2021-05-08 | 60 | 68.3% | 0.688 | 0.0% |
| 2022-10-23 | 60 | 78.3% | 0.787 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
