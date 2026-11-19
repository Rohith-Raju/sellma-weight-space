# Report G6: trial edits at full precision and 4-bit

Verifier: independent subagent (Opus), 2026-11-19. Recomputed from answers.jsonl (LAS+MEK) and damage.json;
loop rates recounted from generations.jsonl. Scripts: scripts/G6_verify.py, scripts/G6_q4_diff.py.
Format: all % / A-right % / B,C,D-right % / times A picked.

| # | Claim | Claimed | Recomputed | Verdict |
|---|---|---|---|---|
| 1 | Hidden LAS+MEK size, key=A | 320, 74 | 320, 74 (hidden runs = the 480 hidden ids) | CONFIRMED |
| 2 | Unedited full | 72.5/51.4/78.9/47 | same (232/320, 38/74, 194/246) | CONFIRMED |
| 3 | Unedited 4-bit | 68.8/41.9/76.8/38 | same (220, 31, 189) | CONFIRMED |
| 4 | trial_b50_k full | 73.8/60.8/77.6/58 | same (236, 45, 191) | CONFIRMED |
| 5 | trial_b50_k 4-bit | 70.0/50.0/76.0/47 | same (224, 37, 187) | CONFIRMED |
| 6 | trial_b50_k control full | 72.8/51.4/79.3/47 | same (233, 38, 195) | CONFIRMED |
| 7 | trial_b10_q_full full | 70.3/54.1/75.2/51 | same | CONFIRMED |
| 8 | trial_b10_q_full 4-bit | 65.0/44.6/71.1/42 | same | CONFIRMED |
| 9 | trial_b25_mlp full | 72.5/52.7/78.5/48 | same | CONFIRMED |
| 10 | trial_b25_mlp 4-bit | 65.6/39.2/73.6/39 | same | CONFIRMED |
| 11 | A-right gain b50_k full | +9.4 | +9.46 (45/74 vs 38/74) | CONFIRMED |
| 12 | A-right gain b50_k 4-bit | +8.1 | +8.11 (37/74 vs 31/74) | CONFIRMED |
| 13 | Test A-right full | 47.2 -> 55.6 | 17/36 -> 20/36 | CONFIRMED |
| 14 | b50_k damage within limits | within | perplexity +0.05%, loop 10% -> 8%; 4-bit +0.03%, +0 | CONFIRMED |
| 15 | b10/b25 flagged, loop +4 | +4 | 10 -> 14 both; perplexity passes (-0.03%, -2.51%) | CONFIRMED |
| 16 | 4-bit hidden net b50_k | +4, CI includes 0 | +4 (7 fixed, 3 broken), CI -0.4 to +2.1 pp | CONFIRMED |
| 17 | 4-bit runs used an edited GGUF | edited | b50_k 838d5576…, b10_q_full 2e521bfc…, b25_mlp 8b7230a7…, b10_q_half b2875467…; all differ from fe8598adf4… (re-hashed) | CONFIRMED (from run.json; edited GGUFs deleted) |

## Other problems
- The loop-rate rule is close to noise: 50 prompts, 2 points per prompt; random controls also exceed +2
  (b50_k control +8, b10_q_full control +6, b10_q_half control +4, b25_mlp control +0). "b10/b25 flagged" is not
  edit-specific damage.
- Damage claims cover full precision only; at 4-bit b10_q_full loop -2, b25_mlp +6, b10_q_half +4.
- Full-precision edits are in memory, so file checksums are the unedited ones; evidence of application is the
  change log in run.json and answers differing from unedited (16-37 letters on hidden).
- trial_b50_k results.md header says "budget 1 (used 0)" and "changed 655,360 weights": 655,360 touched,
  366,231 changed.
- results.md uses all 480 hidden questions; claims use the 320 LAS+MEK (b50_k full +5 vs +4; 4-bit +4 both).
- trial_b10_q_half (not in claims): full 70.9/52.7/76.4/50, 4-bit 67.5/44.6/74.4/41.
