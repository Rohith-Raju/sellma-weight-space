# Run summary: gemma-4b-full+editor[trial_b25_mlp]

Backend `hf`, split `dev`, 720 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 720 | 71.5% (68.1% to 74.7%) | 0.715 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 480 | 65.4% (61.1% to 69.5%) | 0.655 | 1.000 | 0.0% | no |
| ORD | 240 | 83.8% (78.6% to 87.9%) | 0.833 | 1.000 | 0.0% | no |
| LAS | 240 | 68.8% (62.6% to 74.3%) | 0.687 | 1.000 | 0.0% | no |
| MEK | 240 | 62.1% (55.8% to 68.0%) | 0.623 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 17.2% | 25.7% | 24.4% | 25.7% | 6.9% |
| Headline (LAS+MEK) | 17.3% | 28.5% | 26.5% | 27.7% | 0.0% |
| ORD | 17.1% | 20.0% | 20.4% | 21.7% | 20.8% |
| LAS | 17.1% | 26.7% | 28.7% | 27.5% | 0.0% |
| MEK | 17.5% | 30.4% | 24.2% | 27.9% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 73.3% | 0.747 | 0.0% |
| 2014-04-05 | 60 | 68.3% | 0.681 | 0.0% |
| 2014-10-25 | 60 | 66.7% | 0.659 | 0.0% |
| 2015-10-24 | 60 | 60.0% | 0.601 | 0.0% |
| 2016-10-29 | 60 | 78.3% | 0.779 | 0.0% |
| 2017-04-01 | 60 | 78.3% | 0.779 | 0.0% |
| 2019-04-06 | 60 | 75.0% | 0.746 | 0.0% |
| 2021-03-13 | 60 | 73.3% | 0.737 | 0.0% |
| 2021-10-24 | 60 | 58.3% | 0.575 | 0.0% |
| 2022-05-07 | 60 | 83.3% | 0.836 | 0.0% |
| 2023-03-25 | 60 | 65.0% | 0.650 | 0.0% |
| 2024-04-13 | 60 | 78.3% | 0.783 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
