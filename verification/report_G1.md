# Report G1: data, split, standard runs (unedited models)

Verifier: independent subagent (Opus), 2026-11-19. Scripts: scripts/G1_data_split.py, G1_answer_keys.py,
G1_render_keys.py, G1_visual_keys.py, G1_booklet_text.py, G1_runs.py. Perplexity taken from damage.json
(needs the model); everything else recomputed from raw files.

| # | Claimed | Recomputed | Verdict |
|---|---|---|---|
| 1 | 28 exams, 57 passes, 55 kept, 1,650 questions; 2018-04-14 p5 set aside | same; the 8 answer-key versions for 2018-04-14 disagree on pass 5, agree on pass 2 | CONFIRMED |
| 2 | 10/10/10 per pass; ORD 5 options, LAS/MEK 4 | exactly, all 55 passes; every question and option text found in its source PDF | CONFIRMED |
| 3 | test 180 (60/60/60), gemma_extra 120, dev 1,350 (900 LAS+MEK) | same; no overlap; hashes in split.json and every run.json match current files | CONFIRMED |
| 4 | Answer keys match the official PDFs | by eye 180/180 (2013-10-26 p1+p4, 2021-05-08 p1+p4, test 2026-10-18 p2+p5); own parser 1,650/1,650 | CONFIRMED |
| 5 | Standard-run scores (10 runs) | identical, recomputed from answers.jsonl; loop rates recounted from generations.jsonl | CONFIRMED |
| 6 | Right weights per run (hashes) | all run.json match; fresh sha256sum on the GPU matches all 8 weight files | CONFIRMED |
| 7 | Mean prob_on_letters >= 0.999 | lowest 0.999409 (qwen-9b-q4 dev); lowest question 0.9853 | CONFIRMED |

Per-run integrity (all 10): question set = split; stored key = questions.jsonl; chosen = highest-probability
letter; is_correct consistent; summary.json = recomputation.

## Other problems
1. Full vs 4-bit perplexity are not comparable (separate texts vs llama-perplexity on joined text, half-window
   scoring, different tokenization): 4-bit is lower for both models. Compare only within one precision.
2. Qwen's loop rate is saturated (88% full, 100% 4-bit; 50 prompts, 2-point steps).
3. run.json hashes may come from a size/mtime cache; fresh hashes match.
4. A second pair of 4-bit files in models/compressed/made_with_f32_intermediate/ is byte-identical; no effect.
