# Run summary: gemma-4b-full+editor[trial_b25_mlp]

Backend `hf`, split `dev`, 150 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 150 | 73.3% (65.7% to 79.8%) | 0.731 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 100 | 68.0% (58.3% to 76.3%) | 0.677 | 1.000 | 0.0% | no |
| ORD | 50 | 84.0% (71.5% to 91.7%) | 0.840 | 1.000 | 0.0% | no |
| LAS | 50 | 66.0% (52.2% to 77.6%) | 0.659 | 1.000 | 0.0% | no |
| MEK | 50 | 70.0% (56.2% to 80.9%) | 0.694 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 19.3% | 22.7% | 24.7% | 27.3% | 6.0% |
| Headline (LAS+MEK) | 17.0% | 25.0% | 24.0% | 34.0% | 0.0% |
| ORD | 24.0% | 18.0% | 26.0% | 14.0% | 18.0% |
| LAS | 14.0% | 18.0% | 28.0% | 40.0% | 0.0% |
| MEK | 20.0% | 32.0% | 20.0% | 28.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2018-04-14 | 30 | 70.0% | 0.694 | 0.0% |
| 2022-03-12 | 60 | 73.3% | 0.733 | 0.0% |
| 2023-10-22 | 60 | 75.0% | 0.748 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
