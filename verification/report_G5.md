# Report G5: AI editors (gemma_b10, gemma_b25, gemma_b50) and rule audit

Verifier: independent subagent (Fable), 2026-11-19. Scripts: scripts/G5_verify.py, scripts/G5_remote_readonly.py
(read-only on the GPU), scripts/G5_notes_edits.py.

| # | Claim | Claimed | Recomputed | Verdict |
|---|---|---|---|---|
| 1a | Passes used | 5/10, 7/25, 11/50 | same (ledger cost sums, pass folders, final_edit.json) | CONFIRMED |
| 1b | Final edits change 0 weights | 0 | 0, 0, 0 (one zero row/column each; checked against stored weights) | CONFIRMED |
| 2 | No rule breaks | none | none in 74 scripts read line by line | CONFIRMED (caveats below) |
| 3a-d | b10 pass 3/4, b25 pass 3/4 practice | 336->332 (16/20), 336->339 (11/8), 337->339 (9/7), 337->328 (15/24) | same | CONFIRMED |
| 3e-g | b50 pass 5/6/9 | 350->351, 350->351, 350->354 | same (2/1, 3/2, 15/11) | CONFIRMED |
| 3h | b50 whole practice set | 624->628 | same (MEK 179->185, LAS 204->200, ORD 241->243) | CONFIRMED |
| 4a | b10 "A chosen 86 times in 580, key A 145" | 86/580/145 | 93 of 580 (key A 145); 86 is of 512 (key A 130) | WRONG (number in the editor's own quote; conclusion holds) |
| 4b | b25 MEK key=A 47%, LAS 55% | 47, 55 | 37/78 = 47.4%, 37/67 = 55.2% | CONFIRMED |
| 4c | b50 rotation A->D 22/4, B->A 22/2 | 22/4; 22/2 | 22/4; 22/2 (from stored top-k on the GPU) | CONFIRMED |
| 5a-d | Trial edits weights changed | 606,820; 438,051; 3,932,160; 366,231 | same; each byte-identical to the editor's pass edit | CONFIRMED (0.015% = share touched; share changed 0.0085%) |

## Rule audit
- No gradients/autograd/optimisers: all fits closed form (SVD, pseudo-inverse, ridge, mean differences, t-stats).
- File reads: only each editor's own notes/ and scripts/; pass results only via wb.load_pass.
- No model loading; only import is bench.editor.Workbench. No Workbench.create; no ledger or pass-folder writes.
- All 7,512 question ids in answer passes are drill/validation (practice) questions.
- Judgement points (not forbidden): scripts kept in scripts/ subfolder; partial simulations on saved activations
  (logit lens, single-layer pieces) to pre-screen edits; grid searches on those simulations; b25 uses
  eval(sys.argv[1]) for settings (pass folders confirm the edits used).
- Limit: covers saved scripts and logs; a GPU file-time sweep 07:40-08:12 UTC found nothing else relevant.

## Other problems
1. b50's report "21 fixed, 17 broken" is wrong: passes give 18 fixed, 14 broken (net +4 correct).
2. b10 pass 5 "375->374" uses the editor's own reversed-order baseline; vs stored original-order answers 371.
3. b25 calls its submission "an empty edit": a zero delta on one column (same effect).
4. b10 evalpass.py comment says "MEK 111 reversed"; code and pass use 97.
5. Read passes differ from the stored baseline on 2-3 of 580 questions (near-ties); no effect.
6. trial_* runs were registered (budget 1, 0 passes) at 08:47 UTC by the owner's session, after all editors submitted.
7. Not recomputed: mechanism statistics in the explanations (logit lens, head-3 statistics), which need the large
   read-pass files left on the GPU.
