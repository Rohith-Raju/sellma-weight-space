# Run summary: gemma-4b-full+editor[trial_b50_k]

Backend `hf`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 75.0% (66.6% to 81.9%) | 0.757 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 68.8% (57.9% to 77.8%) | 0.698 | 1.000 | 0.0% | no |
| ORD | 40 | 87.5% (73.9% to 94.5%) | 0.876 | 1.000 | 0.0% | no |
| LAS | 40 | 65.0% (49.5% to 77.9%) | 0.669 | 1.000 | 0.0% | no |
| MEK | 40 | 72.5% (57.2% to 83.9%) | 0.726 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 22.5% | 24.2% | 19.2% | 28.3% | 5.8% |
| Headline (LAS+MEK) | 25.0% | 28.7% | 20.0% | 26.2% | 0.0% |
| ORD | 17.5% | 15.0% | 17.5% | 32.5% | 17.5% |
| LAS | 22.5% | 27.5% | 15.0% | 35.0% | 0.0% |
| MEK | 27.5% | 30.0% | 25.0% | 17.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 68.3% | 0.697 | 0.0% |
| 2025-04-05 | 60 | 81.7% | 0.818 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
