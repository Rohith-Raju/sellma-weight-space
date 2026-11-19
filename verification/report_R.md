# Report R: independent re-scoring of saved answers

Date 2026-11-19. Model `/root/sellma/models/gemma-3-4b-it-official`, bfloat16, CUDA, loaded with plain
`transformers` (`Gemma3ForConditionalGeneration`). Scorer written from scratch in
`scripts/R_rescore.py` (GPU copy: `/root/sellma/verification/R_rescore.py`). It imports only
`bench.prompts.build_prompt/render_prompt` and the `bench.records.Question` dataclass; no
`bench.backends`, `bench.run` or `bench.editor`. Sampling seed 20261119 (parts 2/3: seed+1, part 4: seed+2).

Result: every re-scored answer matches the saved files. No mismatches.

## Weights

| file | sha256 | expected prefix |
|---|---|---|
| model-00001-of-00002.safetensors | eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f | eb5fd5e97d OK |
| model-00002-of-00002.safetensors | fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13 | fdde0e5aa5 OK |

## Agreement

| # | Saved file | Sample | Compared | Agree | Mismatches |
|---|---|---|---|---|---|
| 1 | runs/20261119T060929Z_gemma-4b-full_test/answers.jsonl | 40 questions (14 ORD, 12 LAS, 14 MEK) | chosen letter | 40/40 | 0 |
| 1 | same | same | all letter probabilities within 1e-3 | 40/40 (largest difference 7.2e-9) | 0 |
| 2 | experiments/analysis/heldout_position.jsonl, unedited, hidden | 30 questions x 4 rotations (14 LAS, 16 MEK) | correct_by_position | 120/120 | 0 |
| 2 | same | same 30 | original_order_correct | 30/30 | 0 |
| 2 | same | same 30 | option_average_correct | 30/30 | 0 |
| 3 | same file, editor3_edit, hidden | same 30 x 4 rotations | correct_by_position | 120/120 | 0 |
| 3 | same | same 30 | original_order_correct | 30/30 | 0 |
| 3 | same | same 30 | option_average_correct | 30/30 | 0 |
| 4 | experiments/analysis/other_exams/unedited_with_choices.json, belebele_en | 20 items x 4 rotations | chosen letter | 80/80 | 0 |
| 4 | same | same | correct flag | 80/80 | 0 |

## Notes

- Letter-token rule reimplemented: every vocabulary id whose decoded text, stripped of whitespace, is the
  letter. 3 token ids per letter for A-E. Probability = softmax over the full vocabulary in float32,
  summed over those ids; chosen letter = highest.
- Only part 1 has saved probabilities. The held-out file stores true/false per answer position and the
  Belebele file stores the chosen letter, so parts 2-4 are letter-level comparisons.
- Editor 3's change (`editor_runs/trial_b50_k/final_edit.safetensors`): one tensor,
  `model.language_model.layers.23.self_attn.k_proj.weight`, form "rows", 256 row indices, values
  256 x 2560 float32. Added in float32 and cast back to bfloat16: 366,231 of 655,360 touched weights
  actually change, the same count as final_edit.json (and claim G5.5). Original weights were restored
  afterwards (bit-equal check passed; a re-score of a part-1 question gave the saved probabilities again).
- The edit comparison is not trivially satisfied: in this sample the edit changes 7 of the 120
  rotation cells (right with the answer at A: 15 -> 19 of 30; C: 22 -> 21), and all 7 match the saved file.
- Belebele English was read straight from `data/eng_Latn.jsonl` at revision 7899cdfa... with
  `hf_hub_download` (not through `datasets`), and the English prompt rebuilt from the text in
  `work/other_exams.py`. In the sample the model was right in 74 of 80 cells.
- Not covered: Qwen, the 4-bit models, dev/gemma_extra runs, the test set of the held-out file,
  Belebele Swedish, MMLU maths, the random control, the prompt-note condition.

Files: `scripts/R_rescore.py`, `scripts/R_results.json` (per-question rows), `scripts/R_rescore.log`.
