# Run summary: gemma-4b-q4+editor[trial_b10_q_half]

Backend `llamacpp`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 68.3% (59.6% to 76.0%) | 0.692 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 60.0% (49.0% to 70.0%) | 0.613 | 1.000 | 0.0% | no |
| ORD | 40 | 85.0% (70.9% to 92.9%) | 0.850 | 1.000 | 0.0% | no |
| LAS | 40 | 60.0% (44.6% to 73.7%) | 0.607 | 1.000 | 0.0% | no |
| MEK | 40 | 60.0% (44.6% to 73.7%) | 0.620 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 19.2% | 30.0% | 20.0% | 25.0% | 5.8% |
| Headline (LAS+MEK) | 20.0% | 37.5% | 20.0% | 22.5% | 0.0% |
| ORD | 17.5% | 15.0% | 20.0% | 30.0% | 17.5% |
| LAS | 22.5% | 30.0% | 15.0% | 32.5% | 0.0% |
| MEK | 17.5% | 45.0% | 25.0% | 12.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 70.0% | 0.701 | 0.0% |
| 2025-04-05 | 60 | 66.7% | 0.684 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
