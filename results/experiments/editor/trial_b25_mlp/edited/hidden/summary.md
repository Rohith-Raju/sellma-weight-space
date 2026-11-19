# Run summary: gemma-4b-full+editor[trial_b25_mlp]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 76.0% (72.0% to 79.6%) | 0.760 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 72.5% (67.4% to 77.1%) | 0.724 | 1.000 | 0.0% | no |
| ORD | 160 | 83.1% (76.6% to 88.1%) | 0.833 | 1.000 | 0.0% | no |
| LAS | 160 | 72.5% (65.1% to 78.8%) | 0.727 | 1.000 | 0.0% | no |
| MEK | 160 | 72.5% (65.1% to 78.8%) | 0.722 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.8% | 26.0% | 27.5% | 22.7% | 7.9% |
| Headline (LAS+MEK) | 15.0% | 27.8% | 31.6% | 25.6% | 0.0% |
| ORD | 17.5% | 22.5% | 19.4% | 16.9% | 23.8% |
| LAS | 15.6% | 25.0% | 34.4% | 25.0% | 0.0% |
| MEK | 14.4% | 30.6% | 28.7% | 26.2% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 78.3% | 0.777 | 0.0% |
| 2016-04-09 | 60 | 75.0% | 0.748 | 0.0% |
| 2017-10-21 | 60 | 78.3% | 0.778 | 0.0% |
| 2018-10-21 | 60 | 81.7% | 0.818 | 0.0% |
| 2019-10-20 | 60 | 73.3% | 0.731 | 0.0% |
| 2020-10-25 | 60 | 75.0% | 0.750 | 0.0% |
| 2021-05-08 | 60 | 65.0% | 0.662 | 0.0% |
| 2022-10-23 | 60 | 81.7% | 0.820 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
