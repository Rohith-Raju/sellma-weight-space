# Run summary: gemma-4b-full+editor[trial_b10_q_full]

Backend `hf`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 76.7% (68.3% to 83.3%) | 0.766 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 71.2% (60.5% to 80.0%) | 0.712 | 1.000 | 0.0% | no |
| ORD | 40 | 87.5% (73.9% to 94.5%) | 0.875 | 1.000 | 0.0% | no |
| LAS | 40 | 70.0% (54.6% to 81.9%) | 0.692 | 1.000 | 0.0% | no |
| MEK | 40 | 72.5% (57.2% to 83.9%) | 0.732 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.0% | 28.3% | 21.7% | 24.2% | 5.8% |
| Headline (LAS+MEK) | 22.5% | 35.0% | 22.5% | 20.0% | 0.0% |
| ORD | 15.0% | 15.0% | 20.0% | 32.5% | 17.5% |
| LAS | 20.0% | 35.0% | 17.5% | 27.5% | 0.0% |
| MEK | 25.0% | 35.0% | 27.5% | 12.5% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 70.0% | 0.695 | 0.0% |
| 2025-04-05 | 60 | 83.3% | 0.838 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
