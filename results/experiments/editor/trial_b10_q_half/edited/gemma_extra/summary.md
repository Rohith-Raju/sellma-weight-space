# Run summary: gemma-4b-full+editor[trial_b10_q_half]

Backend `hf`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 75.8% (67.4% to 82.6%) | 0.761 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 70.0% (59.2% to 78.9%) | 0.704 | 1.000 | 0.0% | no |
| ORD | 40 | 87.5% (73.9% to 94.5%) | 0.876 | 1.000 | 0.0% | no |
| LAS | 40 | 65.0% (49.5% to 77.9%) | 0.658 | 1.000 | 0.0% | no |
| MEK | 40 | 75.0% (59.8% to 85.8%) | 0.750 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.0% | 26.7% | 20.0% | 27.5% | 5.8% |
| Headline (LAS+MEK) | 22.5% | 32.5% | 20.0% | 25.0% | 0.0% |
| ORD | 15.0% | 15.0% | 20.0% | 32.5% | 17.5% |
| LAS | 20.0% | 32.5% | 15.0% | 32.5% | 0.0% |
| MEK | 25.0% | 32.5% | 25.0% | 17.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 68.3% | 0.689 | 0.0% |
| 2025-04-05 | 60 | 83.3% | 0.833 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
