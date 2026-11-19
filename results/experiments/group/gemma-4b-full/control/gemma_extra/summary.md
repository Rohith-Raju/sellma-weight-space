# Run summary: gemma-4b-full+group[random]

Backend `hf`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 77.5% (69.2% to 84.1%) | 0.777 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 72.5% (61.9% to 81.1%) | 0.726 | 1.000 | 0.0% | no |
| ORD | 40 | 87.5% (73.9% to 94.5%) | 0.879 | 1.000 | 0.0% | no |
| LAS | 40 | 67.5% (52.0% to 79.9%) | 0.675 | 1.000 | 0.0% | no |
| MEK | 40 | 77.5% (62.5% to 87.7%) | 0.777 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 19.2% | 25.8% | 19.2% | 30.0% | 5.8% |
| Headline (LAS+MEK) | 21.2% | 31.2% | 18.8% | 28.7% | 0.0% |
| ORD | 15.0% | 15.0% | 20.0% | 32.5% | 17.5% |
| LAS | 20.0% | 30.0% | 12.5% | 37.5% | 0.0% |
| MEK | 22.5% | 32.5% | 25.0% | 20.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 71.7% | 0.721 | 0.0% |
| 2025-04-05 | 60 | 83.3% | 0.833 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
