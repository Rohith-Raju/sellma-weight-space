# Run summary: gemma-4b-full+editor[trial_b50_k]

Backend `hf`, split `dev`, 150 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 150 | 74.7% (67.2% to 81.0%) | 0.741 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 100 | 70.0% (60.4% to 78.1%) | 0.695 | 1.000 | 0.0% | no |
| ORD | 50 | 84.0% (71.5% to 91.7%) | 0.833 | 1.000 | 0.0% | no |
| LAS | 50 | 66.0% (52.2% to 77.6%) | 0.663 | 1.000 | 0.0% | no |
| MEK | 50 | 74.0% (60.4% to 84.1%) | 0.727 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 22.0% | 20.7% | 24.0% | 27.3% | 6.0% |
| Headline (LAS+MEK) | 21.0% | 22.0% | 23.0% | 34.0% | 0.0% |
| ORD | 24.0% | 18.0% | 26.0% | 14.0% | 18.0% |
| LAS | 16.0% | 18.0% | 30.0% | 36.0% | 0.0% |
| MEK | 26.0% | 26.0% | 16.0% | 32.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2018-04-14 | 30 | 76.7% | 0.752 | 0.0% |
| 2022-03-12 | 60 | 75.0% | 0.744 | 0.0% |
| 2023-10-22 | 60 | 73.3% | 0.732 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
