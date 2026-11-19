# Run summary: gemma-4b-full+editor[trial_b10_q_full]

Backend `hf`, split `dev`, 150 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 150 | 71.3% (63.6% to 78.0%) | 0.716 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 100 | 67.0% (57.3% to 75.4%) | 0.675 | 1.000 | 0.0% | no |
| ORD | 50 | 80.0% (67.0% to 88.8%) | 0.800 | 1.000 | 0.0% | no |
| LAS | 50 | 64.0% (50.1% to 75.9%) | 0.649 | 1.000 | 0.0% | no |
| MEK | 50 | 70.0% (56.2% to 80.9%) | 0.700 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.0% | 25.3% | 26.0% | 23.3% | 5.3% |
| Headline (LAS+MEK) | 18.0% | 28.0% | 26.0% | 28.0% | 0.0% |
| ORD | 24.0% | 20.0% | 26.0% | 14.0% | 16.0% |
| LAS | 16.0% | 22.0% | 30.0% | 32.0% | 0.0% |
| MEK | 20.0% | 34.0% | 22.0% | 24.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2018-04-14 | 30 | 73.3% | 0.733 | 0.0% |
| 2022-03-12 | 60 | 68.3% | 0.692 | 0.0% |
| 2023-10-22 | 60 | 73.3% | 0.732 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
