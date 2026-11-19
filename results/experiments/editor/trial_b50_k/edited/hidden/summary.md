# Run summary: gemma-4b-full+editor[trial_b50_k]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 76.7% (72.7% to 80.2%) | 0.767 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 73.8% (68.7% to 78.3%) | 0.739 | 1.000 | 0.0% | no |
| ORD | 160 | 82.5% (75.9% to 87.6%) | 0.822 | 1.000 | 0.0% | no |
| LAS | 160 | 75.6% (68.4% to 81.6%) | 0.758 | 1.000 | 0.0% | no |
| MEK | 160 | 71.9% (64.5% to 78.3%) | 0.720 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 18.5% | 25.0% | 26.7% | 22.5% | 7.3% |
| Headline (LAS+MEK) | 18.1% | 27.2% | 29.4% | 25.3% | 0.0% |
| ORD | 19.4% | 20.6% | 21.2% | 16.9% | 21.9% |
| LAS | 19.4% | 23.8% | 33.8% | 23.1% | 0.0% |
| MEK | 16.9% | 30.6% | 25.0% | 27.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 76.7% | 0.768 | 0.0% |
| 2016-04-09 | 60 | 73.3% | 0.736 | 0.0% |
| 2017-10-21 | 60 | 76.7% | 0.768 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.833 | 0.0% |
| 2019-10-20 | 60 | 71.7% | 0.717 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.770 | 0.0% |
| 2021-05-08 | 60 | 73.3% | 0.736 | 0.0% |
| 2022-10-23 | 60 | 81.7% | 0.805 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
