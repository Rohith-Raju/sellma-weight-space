# Run summary: gemma-4b-full+group[trained]

Backend `hf`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 77.5% (69.2% to 84.1%) | 0.777 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 72.5% (61.9% to 81.1%) | 0.726 | 1.000 | 0.0% | no |
| ORD | 40 | 87.5% (73.9% to 94.5%) | 0.880 | 1.000 | 0.0% | no |
| LAS | 40 | 67.5% (52.0% to 79.9%) | 0.677 | 1.000 | 0.0% | no |
| MEK | 40 | 77.5% (62.5% to 87.7%) | 0.775 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.8% | 25.8% | 19.2% | 28.3% | 5.8% |
| Headline (LAS+MEK) | 23.8% | 31.2% | 18.8% | 26.2% | 0.0% |
| ORD | 15.0% | 15.0% | 20.0% | 32.5% | 17.5% |
| LAS | 20.0% | 32.5% | 12.5% | 35.0% | 0.0% |
| MEK | 27.5% | 30.0% | 25.0% | 17.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 70.0% | 0.713 | 0.0% |
| 2025-04-05 | 60 | 85.0% | 0.841 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
