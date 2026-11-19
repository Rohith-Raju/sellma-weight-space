# Run summary: gemma-4b-q4+group[trained]

Backend `llamacpp`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 71.7% (63.0% to 79.0%) | 0.710 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 65.0% (54.1% to 74.5%) | 0.638 | 1.000 | 0.0% | no |
| ORD | 40 | 85.0% (70.9% to 92.9%) | 0.854 | 1.000 | 0.0% | no |
| LAS | 40 | 65.0% (49.5% to 77.9%) | 0.631 | 1.000 | 0.0% | no |
| MEK | 40 | 65.0% (49.5% to 77.9%) | 0.645 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.8% | 28.3% | 19.2% | 25.8% | 5.8% |
| Headline (LAS+MEK) | 22.5% | 35.0% | 18.8% | 23.8% | 0.0% |
| ORD | 17.5% | 15.0% | 20.0% | 30.0% | 17.5% |
| LAS | 25.0% | 27.5% | 15.0% | 32.5% | 0.0% |
| MEK | 20.0% | 42.5% | 22.5% | 15.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 70.0% | 0.699 | 0.0% |
| 2025-04-05 | 60 | 73.3% | 0.721 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
