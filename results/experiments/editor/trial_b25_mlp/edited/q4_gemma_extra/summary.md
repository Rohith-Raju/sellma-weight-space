# Run summary: gemma-4b-q4+editor[trial_b25_mlp]

Backend `llamacpp`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 70.0% (61.3% to 77.5%) | 0.694 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 62.5% (51.5% to 72.3%) | 0.615 | 1.000 | 0.0% | no |
| ORD | 40 | 85.0% (70.9% to 92.9%) | 0.850 | 1.000 | 0.0% | no |
| LAS | 40 | 60.0% (44.6% to 73.7%) | 0.597 | 1.000 | 0.0% | no |
| MEK | 40 | 65.0% (49.5% to 77.9%) | 0.634 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.0% | 30.0% | 19.2% | 25.0% | 5.8% |
| Headline (LAS+MEK) | 21.2% | 37.5% | 18.8% | 22.5% | 0.0% |
| ORD | 17.5% | 15.0% | 20.0% | 30.0% | 17.5% |
| LAS | 22.5% | 30.0% | 15.0% | 32.5% | 0.0% |
| MEK | 20.0% | 45.0% | 22.5% | 12.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 71.7% | 0.704 | 0.0% |
| 2025-04-05 | 60 | 68.3% | 0.683 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
