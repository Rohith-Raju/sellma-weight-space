# Run summary: gemma-4b-full+editor[trial_b50_k:random]

Backend `hf`, split `dev`, 150 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 150 | 72.7% (65.0% to 79.2%) | 0.726 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 100 | 68.0% (58.3% to 76.3%) | 0.676 | 1.000 | 0.0% | no |
| ORD | 50 | 82.0% (69.2% to 90.2%) | 0.825 | 1.000 | 0.0% | no |
| LAS | 50 | 66.0% (52.2% to 77.6%) | 0.659 | 1.000 | 0.0% | no |
| MEK | 50 | 70.0% (56.2% to 80.9%) | 0.693 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 19.3% | 22.7% | 24.0% | 28.7% | 5.3% |
| Headline (LAS+MEK) | 18.0% | 25.0% | 22.0% | 35.0% | 0.0% |
| ORD | 22.0% | 18.0% | 28.0% | 16.0% | 16.0% |
| LAS | 16.0% | 18.0% | 28.0% | 38.0% | 0.0% |
| MEK | 20.0% | 32.0% | 16.0% | 32.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2018-04-14 | 30 | 76.7% | 0.757 | 0.0% |
| 2022-03-12 | 60 | 71.7% | 0.719 | 0.0% |
| 2023-10-22 | 60 | 71.7% | 0.717 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
