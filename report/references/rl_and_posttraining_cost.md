# Reinforcement learning for non-English ability, RL's capability ceiling, and post-training cost

Scope: (1) RL applied to non-English or multilingual ability, (2) whether RL expands capability or sharpens what the base model already has, (3) reported compute for SFT, DPO and RL at the 4–9B scale. General SFT/DPO multilingual instruction tuning and continued pre-training are covered elsewhere and are left out here.

All sources were accessed on 2026-11-19. Every source below was opened directly: the arXiv API abstract, the full PDF text (extracted locally), the ACL Anthology page, or the official blog. Quotes are verbatim. Numbers marked "(my arithmetic)" are derived by me from the quoted figures and are not stated by the authors.

Peer-review status is given for each source. "Preprint" means arXiv lists no venue and I found none.

---

## Verified references

### A. RL applied to non-English / multilingual ability

**A1. Park, C., Kim, J., Lee, J., Bae, S., Choo, J., & Yoo, K. M. (2025, rev. 2026). "Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models." arXiv:2506.05850v3. PREPRINT (arXiv comment: "Preprint").** https://arxiv.org/abs/2506.05850

- Setup: GRPO on math datasets machine-translated into Korean, Ukrainian, Chinese, Thai and Japanese. Small models (Qwen-2.5-1.5B Instruct, Llama-3.2-3B Instruct, OLMo2-1B).
- Abstract: "under RLVR, CoT in LLMs systematically drifts toward the pre-training dominant language as reasoning performance rises" and "interventions that favor target-language traces--via a language-consistency reward, decoding-time controls, or more balanced backbones--mitigate collapse but reveal a persistent performance-fidelity trade-off."
- Table 3 (Qwen-2.5-1.5B, Korean Global MMLU-Lite): GSM8K-KO only gives 31.5 accuracy with 71.6% Korean words. Adding a language-consistency reward gives 31.0 with 75.2% Korean words. Adding a hard curriculum gives 33.4 with only 23.4% Korean words.
- Table 4 caption: restricting decoding "effectively prevents this language drift, but at the cost of a 5–12 percentage-point drop in accuracy."
- Verdict: **SUPPORTS "RL mostly sharpens existing ability".** RL accuracy gains came from moving reasoning into English, the model's strongest language. Forcing the target language cost accuracy.

**A2. Zhang, X., Liang, Y., Meng, F., Zhang, S., Huang, K., Chen, Y., Xu, J., & Zhou, J. (2026). "Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning." ACL 2026 (Long Papers), pp. 20766–20783. PEER-REVIEWED.** https://aclanthology.org/2026.acl-long.951/ (arXiv:2510.07300)

- Method: GRPO with a strict Language Consistency (LC) reward and a Cross-lingual Thinking Alignment (CTA) reward that uses the model's own English reasoning as reference.
- Abstract: "our M-Thinker-1.5B/4B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath)".
- Table 1 (DeepSeek-R1-Distill-Qwen-7B on MMATH, average over 10 languages):
  - Language consistency: base 25.58% → M-Thinker Iter-2 95.50%.
  - Accuracy: base 58.58% → M-Thinker Iter-2 65.04%.
  - "Naive-RL" (accuracy reward only) reaches 65.43% accuracy but 0.00% language consistency in every non-English, non-Chinese language.
- Verdict: **MIXED, leaning "sharpens".** The big gain is in language consistency. The accuracy gain (+6.5 points) is about the same as plain RL that reasons in English (+6.9 points). The CTA reward transfers the model's English reasoning. It does not add knowledge of the language.

**A3. Hwang, J., Tanmay, K., Lee, S.-J., Agrawal, A., Palangi, H., Ayush, K., Fiete, I., & Liang, P. P. (2025). "Learn Globally, Speak Locally: Bridging the Gaps in Multilingual Reasoning." arXiv:2507.05418v2. PREPRINT.** https://arxiv.org/abs/2507.05418

- Backbone: Qwen-2.5-7B-Instruct, trained on 4 A100 GPUs.
- Table 2 (MGSM, 10 languages): base math 58.7, plain GRPO 58.8 ± 0.4, their M2A method 59.0 ± 0.3.
- Text: "GRPO, in contrast, produces little change, likely due to sparse rewards. For instance, Figure 11 shows that GRPO outputs are identical to the base model".
- Table 3 (GeoFact-X, geography facts in English, Hindi, Japanese, Swahili, Thai, trained on its own train split): answer accuracy base 26.2, GRPO 32.1 ± 0.3, SFT 29.3 ± 0.2, M2A 32.0 ± 0.6.
- Verdict: **MIXED.** On math, GRPO left the 7B model unchanged. On the factual benchmark, GRPO raised accuracy by about 6 points. But the training split is the same domain as the test set, and the paper does not separate learned facts from better recall.

**A4. Dang, J., Ahmadian, A., Marchisio, K., Kreutzer, J., Üstün, A., & Hooker, S. (2024). "RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs." EMNLP 2024, pp. 13134–13156. PEER-REVIEWED.** https://aclanthology.org/2024.emnlp-main.729/ (arXiv:2407.02552)

- Model: Aya 23 8B. Online RL (RLOO) compared with offline DPO across 23 languages.
- Abstract: "Our preference-trained model achieves a 54.4% win-rate against Aya 23 8B ... and a 69.5% win-rate or higher against widely used models like Gemma-1.1-7B-it, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3."
- Paper, finding 3: "RLOO as an online method achieves better overall performance than DPO by a maximum 10.6% difference in their average win-rates (54.4% vs 43.8%)."
- Note: all gains are LLM-judge win-rates on open-ended chat. The paper does not test knowledge or exam benchmarks.
- Verdict: **MIXED.** RL from AI feedback clearly improves chat quality and preference in many languages. The evidence is about response quality and style, not language knowledge.

**A5. Cheng, T. C., Cheng, C. S., Lau, C. M., & Chong, C. H. (2026). "CantoneseLLM v2: Reasoning in a Low-Resource Language." arXiv:2609.06970v1. PREPRINT.** https://arxiv.org/abs/2609.06970

- Pipeline on Qwen3 8B and 30B-A3B: continued pre-training on 784M Cantonese tokens, chat-vector merging, SFT, DPO, then RLVR.
- Abstract: "SFT with limited Cantonese reasoning data substantially shortens or removes reasoning traces and reduces benchmark performance." and "The RLVR training with Cantonese language and Traditional Chinese scripts as multiplicative constraints introduced Cantonese language alignment and restored the lost performance. The 30B-A3B model reaches 73.16 on HKCanto-Eval, within 1.20 points of its merged checkpoint, while retaining the Cantonese reasoning behaviour absent from that checkpoint."
- Verdict: **SUPPORTS "RL mostly sharpens existing ability".** RL fixed the reasoning language and recovered losses from SFT. It ended slightly below the pre-RL merged checkpoint on the Cantonese benchmark. It did not raise the knowledge score.

**A6. Sutawika, L., Swamy, G., Wu, Z. S., & Neubig, G. (2026). "Gained in Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning." arXiv:2601.18722v1. PREPRINT.** https://arxiv.org/abs/2601.18722

- Method (SP3F): SFT on translated English QA, then self-play RL with a pairwise judge that sees the English reference answer.
- Abstract: "SP3F greatly improves base model performance, even outperforming fully post-trained models on multiple math and non-math tasks with less than of the training data across the single-language, multilingual, and generalization to unseen language settings." (The fraction is missing in the arXiv abstract text itself.)
- I read only the abstract. I did not check per-language numbers.
- Verdict: **MIXED.** It claims real gains in lower-resource languages. The mechanism is transfer of English-reference reasoning, and the tasks are mainly math and reasoning.

### B. Does RL expand capability or sharpen the base model?

**B1. Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Yue, Y., Song, S., & Huang, G. (2025). "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" NeurIPS 2025 (Oral). PEER-REVIEWED.** https://arxiv.org/abs/2504.13837

- Abstract: "Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model."
- Abstract: "six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities."
- Paper, Sec. 3: "on the Minerva benchmark with a 32B-sized model, the base model outperforms the RL-trained model by approximately 9% at k = 128".
- Verdict: **SUPPORTS "RL mostly sharpens existing ability".**

**B2. Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., & Dong, Y. (2025). "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models." arXiv:2505.24864v1. PREPRINT (arXiv lists no venue).** https://arxiv.org/abs/2505.24864

- Abstract: "we challenge prevailing assumptions by demonstrating that prolonged RL (ProRL) training can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling." and "RL-trained models consistently outperform base models across a wide range of pass@k evaluations, including scenarios where base models fail entirely regardless of the number of attempts."
- Paper: "compared to DeepSeek-R1-1.5B, we achieve average pass@1 improvements of 14.7% on math benchmarks, 13.9% on coding, 54.8% on logic puzzles, 25.1% on STEM reasoning, and 18.1% on instruction-following tasks".
- Important caveat in Sec. 4.1: "tasks where the base model already performs well (i.e., high pass@128) tend to exhibit minimal or even negative gains in reasoning breadth post-RL." Expansion is largest where the base model is weak, mainly on synthetic logic puzzles. Math and code tasks the base model had seen in pre-training gained little.
- Compute: "We conduct training on 4 8 x NVIDIA-H100-80GB nodes, and the whole training runs for approximately 16k GPUs hours." This is for a 1.5B model.
- Verdict: **MIXED.** It is a counter-claim to B1, but by its own analysis the expansion happens on tasks unfamiliar from pre-training, after very long training.

**B3. Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S. S., Lambert, N., Min, S., Krishna, R., Tsvetkov, Y., Hajishirzi, H., Koh, P. W., & Zettlemoyer, L. (2025). "Spurious Rewards: Rethinking Training Signals in RLVR." arXiv:2506.10947v2. PREPRINT (arXiv lists no venue).** https://arxiv.org/abs/2506.10947

- Abstract: "RLVR training with GRPO improves MATH-500 performance for Qwen2.5-Math-7B by 21.4 percentage points using randomly assigned rewards, nearly matching the 29.1-point gain from ground-truth rewards."
- Abstract: "GRPO exhibits a clipping bias from the clip term, which can amplify high-prior behaviors learned during pretraining even without informative rewards." and "spurious rewards that are effective for Qwen models often fail to produce gains for other model families, such as Llama3 or OLMo2."
- Verdict: **SUPPORTS "RL mostly sharpens existing ability".** Much of the measured RL gain can be amplification of behaviour already present after pre-training. How much depends heavily on the base model.

### C. Cost of post-training at 4–9B (figures only)

**C1. Lambert, N., Morrison, J., Pyatkin, V., et al. (2024, rev. 2025). "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv:2411.15124v5. PREPRINT (arXiv lists no venue).** https://arxiv.org/abs/2411.15124

Base model: Llama 3.1 8B. All figures are for the final runs and exclude sweeps and ablations.
- SFT: "The final 8B model is trained on 32 GPUs for 6 hours" → 192 H100-hours (my arithmetic).
- DPO: "The 8B DPO model is trained for 10 hours on 8 Nvidia H100 GPUs" → 80 H100-hours (my arithmetic).
- Reward model and RL: "The final 8B reward model is trained in 9 hours on 8 H100 gpus, while the final 8B RL run takes 65 hours on 8 GPUs" → 72 + 520 H100-hours (my arithmetic).
- Total for the final 8B pipeline: about 860 H100-hours (my arithmetic).
- PPO vs DPO: "The PPO runtime is roughly 28 hours using two nodes, whereas the DPO runtime is about 4 hours using a single node."

**C2. Team Olmo (Ettinger, A., Bertsch, A., ... Hajishirzi, H.) (2025). "Olmo 3." arXiv:2512.13961v2. PREPRINT (arXiv lists no venue).** https://arxiv.org/abs/2512.13961

- 7B reasoning RL: "For our Olmo 3 Think 7B training run, we used an initial version of our infrastructure without pipelineRL or truncation importance sampling, which took approximately 15 days. We later replicated the same run with our newer infrastructure, achieving similar performance in just 6 days of training."
- Table 49, 7B Think RL: 16 learner GPUs plus 56 actor GPUs (72 H100s). If that run used this setup, 6 days is about 10,400 H100-hours and 15 days is about 25,900 H100-hours (my arithmetic, assumes 24 h/day and the Table 49 GPU count).
- Table 47, 7B Thinking SFT: 45.4B tokens on 64 GPUs. Table 48, 7B Thinking DPO: 150K pairs on 32 GPUs. The GPU-hour rows did not come through in the PDF text, so I report only GPU counts.
- Where the compute goes in RL: "Inference dominated our computational costs ... our learner spends 75% of the time waiting for data, so in terms of GPU utilization, we use approximately 5x as much for inference as for training."
- 32B overall: "approximately 56 days elapsed ... on a cluster with 1024 H100 GPUs", of which "Pretraining: ∼47 days" and "Post-training: ∼9 days (SFT, DPO, and RL)". SFT: "we swept over four candidate learning rates, on 256 GPUs each, in parallel for 36 hours." Post-training is repeated: "we run each stage multiple times, sweeping over learning rates and other hyperparameters."

**C3. Team OLMo (Walsh, P., Soldaini, L., Groeneveld, D., ... Hajishirzi, H.) (2025). "2 OLMo 2 Furious." COLM 2025 (shorter version). PEER-REVIEWED.** https://arxiv.org/abs/2501.00656

- Pretraining energy, Table 19: OLMo 2 7B 131 MWh total GPU power. Llama 3.1 8B 1,022 MWh. Useful as a pre-training reference at 7–8B. The report gives post-training recipe details (Tülu 3 based, with RLVR) but this table covers pre-training only.

**C4. Hugging Face (2025). "SmolLM3: smol, multilingual, long-context reasoner." Hugging Face blog. NOT PEER-REVIEWED.** https://huggingface.co/blog/smollm3

- "The model was trained on 384 H100 GPUs for 24 days." This is 3B pre-training on 11.2T tokens. The blog gives token counts for SFT ("1.8B tokens", 4 epochs) but no GPU figures for post-training.

**Cost comparison against continued pre-training (context figures supplied by the requester, not verified here):** Danish 7B continued pre-training took 8,928 GPU-hours. AI Sweden Llama-3-8B continued pre-training used 92 A100s for 30 days, about 66,000 A100-hours (my arithmetic).
- A Tülu-3-style 8B SFT + DPO + RL final pipeline (~860 H100-hours) costs about one-tenth of the Danish continued pre-training run in GPU-hours. Sweeps and evaluation multiply this several times (Olmo 3 text above).
- A long reasoning RL run at 7B (Olmo 3 Think 7B, about 10,000–26,000 H100-hours) costs about the same as, or more than, the Danish continued pre-training run.
- H100 and A100 hours are not equal. An H100-hour does roughly two to three times the work of an A100-hour, so the comparisons are order-of-magnitude only.

---

## What this means for SE.LLMA

- **RL in a non-English language has mostly changed behaviour, not knowledge.** In M-Thinker (A2, ACL 2026), language consistency rose from 25.58% to 95.50%, but accuracy rose by about the same amount as plain RL that reasoned in English (65.04% vs 65.43%). In CantoneseLLM v2 (A5), RLVR restored the reasoning language but the Cantonese benchmark ended 1.20 points below the pre-RL checkpoint. This matches the note's own finding that weight edits shifted answer selection without adding exam knowledge.
- **The capability ceiling is set before RL.** Yue et al. (B1, NeurIPS 2025) find the base model solves more problems than the RL model at large k. ProRL (B2, preprint) disputes this, but its own Sec. 4.1 shows little or negative breadth gain on tasks the base model already knew. Spurious Rewards (B3, preprint) shows large RL gains with random rewards on Qwen but not on Llama or OLMo 2. If a Swedish model lacks facts needed for the exam, RL alone is unlikely to supply them.
- **RL can push a Swedish model to reason in English.** Cross-lingual Collapse (A1, preprint) shows GRPO drifting toward the dominant pre-training language as accuracy rises. A language-consistency reward slowed the drift but cost accuracy (5–12 points with decoding limits on Ukrainian). Any Swedish RL run should track the language of reasoning traces, not only exam accuracy.
- **RL is a reasonable last step for chat quality and instruction following in Swedish.** Dang et al. (A4, EMNLP 2024) report online RLOO beating DPO by up to 10.6 win-rate points across 23 languages on an 8B model. These are judge-based preference gains, so they support RL for helpfulness and style, not for knowledge.
- **Cost: standard post-training is cheap next to continued pre-training, but reasoning RL is not.** A Tülu-3-style 8B SFT + DPO + RL final run is about 860 H100-hours (C1). A long reasoning RL run at 7B took 6–15 days on 72 H100s (C2, roughly 10,000–26,000 H100-hours by my arithmetic), similar to or larger than the 8,928 GPU-hour Danish continued pre-training. Olmo 3 says inference takes about 5x the GPU time of training in RL.
- **Suggested wording for the note:** recommend continued pre-training or data work for Swedish knowledge, then SFT/DPO, then RL with a Swedish language-consistency reward as a final step for reasoning format and instruction following. Do not claim RL will add Swedish knowledge. No source here shows that.

---

## Could not verify

- **o7 (Danish GRPO reasoning model), Hugging Face card huggingface.co/syvai/o7.** A search summary said it was trained with GRPO for 8 days on one A100. I did not open the card.
- **AMD ROCm blog on Finnish SFT + GRPO reasoning (rocm.blogs.amd.com, July 2026).** Seen only in a search snippet.
- **"VEPO: Variable Entropy Policy Optimization" for low-resource languages (arXiv:2603.19152).** Seen only in a search snippet.
- **"Verifying the Subjective: Structured Multilingual Rewards for Low-Resource Alignment" (ACL Findings 2026).** Seen only in a search snippet.
- **arXiv:2608.13698 reporting "severe, language-dependent regressions" from non-English GRPO.** Seen only in a search snippet.
- **Lee et al. (2025), customized GRPO for Korean reasoning.** Cited in M-Thinker. I did not open it.
- **Olmo 3 7B pretraining "~234k H100 hours".** Appears only in an unofficial summary site. I could not find it in the Olmo 3 PDF text.
- **Olmo 3 Tables 47–49 GPU-hour rows.** The captions say "GPU hours assume NVIDIA H100 accelerator", but the rows did not survive PDF text extraction. Only the GPU counts are reported above.
- **Danish 7B (8,928 GPU-hours) and AI Sweden Llama-3-8B (92 A100s, 30 days) figures.** Supplied by the requester. Not re-checked in this task.
- **Peer-review status of ProRL, Spurious Rewards, Tülu 3 and Olmo 3.** arXiv lists no venue. They may have been accepted somewhere, but I did not confirm it, so they are marked as preprints.
