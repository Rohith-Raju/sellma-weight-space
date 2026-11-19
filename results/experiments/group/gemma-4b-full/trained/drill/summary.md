# Run summary: gemma-4b-full+group[trained]

Backend `hf`, split `dev`, 720 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 720 | 71.2% (67.8% to 74.4%) | 0.711 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 480 | 65.2% (60.8% to 69.3%) | 0.652 | 1.000 | 0.0% | no |
| ORD | 240 | 83.3% (78.1% to 87.5%) | 0.830 | 1.000 | 0.0% | no |
| LAS | 240 | 70.0% (63.9% to 75.4%) | 0.701 | 1.000 | 0.0% | no |
| MEK | 240 | 60.4% (54.1% to 66.4%) | 0.604 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 17.5% | 25.7% | 25.0% | 25.1% | 6.7% |
| Headline (LAS+MEK) | 17.3% | 28.5% | 27.3% | 26.9% | 0.0% |
| ORD | 17.9% | 20.0% | 20.4% | 21.7% | 20.0% |
| LAS | 17.5% | 27.9% | 29.6% | 25.0% | 0.0% |
| MEK | 17.1% | 29.2% | 25.0% | 28.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 75.0% | 0.739 | 0.0% |
| 2014-04-05 | 60 | 65.0% | 0.654 | 0.0% |
| 2014-10-25 | 60 | 65.0% | 0.662 | 0.0% |
| 2015-10-24 | 60 | 58.3% | 0.588 | 0.0% |
| 2016-10-29 | 60 | 80.0% | 0.779 | 0.0% |
| 2017-04-01 | 60 | 76.7% | 0.755 | 0.0% |
| 2019-04-06 | 60 | 71.7% | 0.723 | 0.0% |
| 2021-03-13 | 60 | 75.0% | 0.740 | 0.0% |
| 2021-10-24 | 60 | 56.7% | 0.574 | 0.0% |
| 2022-05-07 | 60 | 83.3% | 0.835 | 0.0% |
| 2023-03-25 | 60 | 66.7% | 0.670 | 0.0% |
| 2024-04-13 | 60 | 81.7% | 0.817 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
