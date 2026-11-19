# Run summary: gemma-4b-full+group[trained]

Backend `hf`, split `dev`, 480 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 480 | 76.0% (72.0% to 79.6%) | 0.755 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 320 | 72.8% (67.7% to 77.4%) | 0.720 | 1.000 | 0.0% | no |
| ORD | 160 | 82.5% (75.9% to 87.6%) | 0.824 | 1.000 | 0.0% | no |
| LAS | 160 | 75.0% (67.8% to 81.1%) | 0.739 | 1.000 | 0.0% | no |
| MEK | 160 | 70.6% (63.2% to 77.1%) | 0.701 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 17.3% | 25.4% | 27.9% | 22.1% | 7.3% |
| Headline (LAS+MEK) | 16.6% | 27.2% | 31.2% | 25.0% | 0.0% |
| ORD | 18.8% | 21.9% | 21.2% | 16.2% | 21.9% |
| LAS | 18.1% | 25.6% | 34.4% | 21.9% | 0.0% |
| MEK | 15.0% | 28.7% | 28.1% | 28.1% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2015-03-28 | 60 | 75.0% | 0.752 | 0.0% |
| 2016-04-09 | 60 | 71.7% | 0.714 | 0.0% |
| 2017-10-21 | 60 | 78.3% | 0.768 | 0.0% |
| 2018-10-21 | 60 | 85.0% | 0.835 | 0.0% |
| 2019-10-20 | 60 | 71.7% | 0.715 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.772 | 0.0% |
| 2021-05-08 | 60 | 71.7% | 0.707 | 0.0% |
| 2022-10-23 | 60 | 78.3% | 0.774 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
