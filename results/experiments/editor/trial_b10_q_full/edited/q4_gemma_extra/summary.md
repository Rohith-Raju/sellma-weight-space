# Run summary: gemma-4b-q4+editor[trial_b10_q_full]

Backend `llamacpp`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 69.2% (60.4% to 76.7%) | 0.689 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 61.3% (50.3% to 71.2%) | 0.608 | 1.000 | 0.0% | no |
| ORD | 40 | 85.0% (70.9% to 92.9%) | 0.850 | 1.000 | 0.0% | no |
| LAS | 40 | 60.0% (44.6% to 73.7%) | 0.597 | 1.000 | 0.0% | no |
| MEK | 40 | 62.5% (47.0% to 75.8%) | 0.619 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.0% | 29.2% | 21.7% | 23.3% | 5.8% |
| Headline (LAS+MEK) | 21.2% | 36.2% | 22.5% | 20.0% | 0.0% |
| ORD | 17.5% | 15.0% | 20.0% | 30.0% | 17.5% |
| LAS | 22.5% | 27.5% | 20.0% | 30.0% | 0.0% |
| MEK | 20.0% | 45.0% | 25.0% | 10.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 71.7% | 0.709 | 0.0% |
| 2025-04-05 | 60 | 66.7% | 0.669 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
