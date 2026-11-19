# Continued pre-training, language adaptation pipelines and forgetting: verified references

Prepared 2026-11-19 for the SE.LLMA research note. Every source below was opened on 2026-11-19 (arXiv abstract or HTML page, ACL Anthology page or PDF, model card, or vendor blog). Text in quotation marks is copied verbatim from the opened page. Numbers come from the opened page's tables or text. Verdict labels:

- **SUPPORTS**: continued pre-training (CPT) adds target-language ability.
- **QUALIFIES**: CPT helps but with forgetting, small gains, or conditions.
- **CONTRADICTS**: CPT did not add target-language ability on the measured task.

Peer-review status is flagged for each source. "Preprint" means arXiv only, with no venue I could confirm.

---

## Verified references

### A. Swedish evaluation context

**1. Kurfalı, M., Zahra, S., Gogoulou, E., Dürlich, L., Carlsson, F., and Nivre, J. (2025). SweSAT-1.0: The Swedish University Entrance Exam as a Benchmark for Large Language Models.** In *Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025)*, Tallinn, pp. 331–339. University of Tartu Library. URL: https://aclanthology.org/2025.nodalida-1.36/ (PDF: https://aclanthology.org/2025.nodalida-1.36.pdf). Accessed 2026-11-19. **Peer-reviewed.**

- Introduction (PDF p. 331): "The results show that while multilingual LLMs outperform their Swedish-oriented counterparts, even the commercial models fail at solving the reasoning tasks in Swedish".
- Results (p. 334): "Swedish-oriented models –– all models in the GPT-SW3 family in addition to a fine-tuned Llama 3 version –– consistently show lower accuracy across all question types."
- On the AI Sweden Llama 3 model (p. 334): it "exhibits better performance than all evaluated GPT-SW3 models. Yet, it achieves slightly lower average accuracy than the original Llama 3 on five of the eight exams. This raises questions on whether continued pre-training on a mix of Scandinavian languages is useful for this task".
- Table 2 averages (accuracy, %): AI-Sweden/Llama-3-8B **40.40** vs Llama-3-8B **43.18** vs Llama-3.1-8B 46.46. Gemma-2-9b 58.04, Gemma-2-27b 63.19, GPT-4o-mini 63.99, GPT-4o 71.79. GPT-SW3-1.3b 19.03, GPT-SW3-6.7b-v2 19.15, GPT-SW3-20b 22.71. Appendix lists the instruct checkpoints used (for example `AI-Sweden-Models/Llama-3-8B-instruct` and `meta-llama/Meta-Llama-3-8B-Instruct`).
- Headline confirmed: Swedish-oriented models score below the large multilingual models. The only CPT model tested scored 2.78 points below its own base model.
- **Verdict: CONTRADICTS** (on Högskoleprovet specifically). This is the closest published evidence to the SE.LLMA task.
- **Correction (checked against Table 2 on 2026-11-19):** the 40.40 vs 43.18 averages include the three maths sections (XYZ, KVA, NOG). On the three verbal sections only: ORD 71.25 vs 68.44, LAS 56.25 vs 65.00, MEK 59.69 vs 55.62, average 62.40 vs 63.02 (our arithmetic). Gemma-2-9b verbal average 84.79. So on the Swedish language sections the Nordic CPT model is about equal to its base, not worse; the verdict on verbal sections is NO CLEAR EFFECT, and base-model choice dominates. One model pair, no confidence intervals, and the 2020–2024 papers may be in training data.

**2. AI Sweden (2024). AI-Sweden-Models/Llama-3-8B (model card).** Hugging Face. URL: https://huggingface.co/AI-Sweden-Models/Llama-3-8B. Accessed 2026-11-19. **Model card, not peer-reviewed.**

- "It was trained on a subset from The Nordic Pile containing Swedish, Norwegian and Danish."
- "The training dataset consists of 227 105 079 296 tokens."
- "The training used 23 nodes of a duration of 30 days, where one node contained 4X Nvidia A100 GPUs, yielding 92 GPUs." (About 66,000 A100 GPU-hours by my arithmetic: 92 × 30 × 24.)
- The card gives no benchmark scores.
- **Verdict: QUALIFIES.** Large Nordic CPT (227B tokens) without reported gains. Read with source 1, where this model scored below base Llama 3 8B on SweSAT.

**3. Kunz, J. (2026). A Diagnostic Benchmark for Sweden-Related Factual Knowledge.** In *Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)*, Palma de Mallorca, pp. 5328–5334. ELRA. URL: https://aclanthology.org/2026.lrec-1.416/ (arXiv: https://arxiv.org/abs/2510.21360). Accessed 2026-11-19. **Peer-reviewed.**

- "smaller models with stronger Swedish coverage perform comparably to a multilingual model three times larger"
- "continued pre-training on Swedish generally improves factual knowledge" but "leads to partial forgetting of previously known information."
- **Verdict: QUALIFIES.** Swedish CPT adds Sweden-specific facts but also causes forgetting.

**4. Borggren, L., Kunz, J., and Kuhlmann, M. (2026). Reading the News: Adapting Large Language Models to Swedish Journalism Through Continued Pre-Training.** Accepted at EMNLP 2026 Industry Track (pp. 4305–4327 per the arXiv comment). arXiv:2608.30609. URL: https://arxiv.org/abs/2608.30609. Accessed 2026-11-19. **Peer-reviewed (accepted, per arXiv comment).**

- CPT gives "benefits in the target domain, but only when paired with experience replay to mitigate forgetting".
- "We observe consistent enhancements in the models' generation quality and factual knowledge".
- "an existing Swedish benchmark largely fails to capture the models' in-domain performance gains".
- The abstract gives no numbers.
- **Verdict: QUALIFIES.** Swedish CPT works only with replay, and standard Swedish benchmarks may miss the gains.

### B. Nordic and other mid/low-resource European languages

**5. Samuel, D., Mikhailov, V., Velldal, E., Øvrelid, L., Charpentier, L. G. G., Kutuzov, A., and Oepen, S. (2025). Small Languages, Big Models: A Study of Continual Training on Languages of Norway.** NoDaLiDa 2025. arXiv:2412.06484. URL: https://arxiv.org/abs/2412.06484 (HTML v2 read for numbers). Accessed 2026-11-19. **Peer-reviewed.**

- The three-stage method "substantially improves the downstream performance". NorMistral-11B is "a continually trained Mistral-Nemo-Base-2407 language model".
- Recipe: a new 51,200-token tokenizer (replacing 131,072), embeddings realigned for 1,000 steps with the rest frozen, then full training. 250B tokens. 256 MI250X GPUs, about 55,000 GPU hours, 1.7 × 10²² FLOPs. Swedish, Danish, Icelandic, Faroese, English and code were mixed in. Low-resource data was repeated at most 16 times.
- Table 2, base → NorMistral-11B: NorCommonsenseQA (Bokmål) 46.9 → 61.0. NRK-Quiz-QA (Bokmål) 47.4 → 63.7, (Nynorsk) 47.2 → 71.9. Two benchmarks got worse: Belebele 62.8 → 56.7 and NorOpenBookQA (Bokmål) 86.9 → 77.9.
- **Verdict: SUPPORTS**, with some regressions on knowledge and reading benchmarks.

**6. Zosa, E., Luoma, J., Hakala, K., Virtanen, A., Koistinen, M., and Burdge, J. (2025, 18 June). Continued Pretraining: A Practical Playbook for Language-Specific LLM Adaptation (Poro 2).** AMD ROCm Blogs. URL: https://rocm.blogs.amd.com/artificial-intelligence/multilingual-continued-pretraining/README.html. Accessed 2026-11-19. **Vendor technical blog, not peer-reviewed.** The companion "Poro 2: Continued Pretraining for Language Acquisition" (LumiOpen, 2025) is cited on the model cards, but I found no arXiv version.

- Final mix: "30% Finnish, 30% English, 30% code, and 10% math", "resulting in 165B tokens overall with this data mix". LR 3e-4 (8B) and 1.5e-4 (70B), cosine schedule. Base model is Llama 3.1.
- Finnish average: 8B 39.18 → 49.97 ("~10pp average improvement"). 70B 60.29 → 64.23 ("~4pp").
- English: 8B keeps "most of the original English capabilities with a 1pp average decrease". 70B: "English capabilities remain largely the same with a ~0.4pp average improvement."
- Forgetting: 8B English GSM8K fell 78.01 → 54.81. HumanEval fell 35.97 → 31.09 (8B) and 57.3 → 48.78 (70B). An earlier 70%-Finnish mix caused a larger maths decline.
- Pipeline: "one round of supervised fine-tuning (SFT) followed by preference tuning". SFT used LR 5e-6 for 2 epochs. DPO used LR 5e-7, beta 0.01, and HelpSteer3.
- **Verdict: SUPPORTS.** Gains are larger at 8B than at 70B, and maths and code are partly lost.

**7. Dorkin, A., Purason, T., Kalbaliyev, E., Kuulmets, H.-A., Ojastu, M., Fišel, M., Alumäe, T., Aedmaa, E., Kruusmaa, K., and Sirts, K. (2026). EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training.** arXiv:2603.02041 (v2, 25 Aug 2026). URL: https://arxiv.org/abs/2603.02041 (HTML read for numbers). Accessed 2026-11-19. **Preprint.**

- "we apply CPT with Estonian-enriched multilingual replay". "the more English-centric Llama model achieves substantially larger gains after adaptation". "some English capabilities regress relative to the original instruction-tuned models". "chat vector merging substantially restores English instruction-following and reasoning performance".
- Mix (Table 1, 35.7B tokens): Estonian 8.6B, English 6.9B, code 3.3B, maths 9.5B, instruction data 7.4B. So Estonian is only about a quarter of the mix.
- Estonian discriminative average: Llama 3.1 8B 54.0 → 69.5. Apertus 8B 63.6 → 68.0.
- English after CPT: 76.9 → 75.4. Instruct model vs Llama-3.1-8B-Instruct: MMLU-Redux 69.6 → 63.9, IFEval 81.1 → 75.3, TruthfulQA 52.4 → 36.6. After chat-vector merge (α = 0.5): English IFEval 75.3 → 81.7, Estonian IFEval 51.7 → 61.4. TruthfulQA did not recover (36.6 → 36.4).
- **Verdict: SUPPORTS** (full CPT → SFT → preference → merge pipeline). Gains are small for an already multilingual base model, and some English is lost.

**8. Gogoulou, E., Lesort, T., Boman, M., and Nivre, J. (2024). Continual Learning Under Language Shift.** TSD 2024. arXiv:2311.01200. URL: https://arxiv.org/abs/2311.01200. Accessed 2026-11-19. **Peer-reviewed.**

- Setup: start from an English model and add Danish, Icelandic and Norwegian in sequence.
- "forward transfer is largely positive and independent of language order". Backward transfer "can be positive or negative depending on the order and characteristics of new languages". "a combination of language contamination and syntactic similarity best fits our results."
- **Verdict: SUPPORTS** for forward transfer to Nordic languages. **QUALIFIES** for retention of earlier languages.

### C. Non-European reference recipes

**9. Fujii, K., Nakamura, T., Loem, M., Iida, H., Ohi, M., Hattori, K., Shota, H., Mizuki, S., Yokota, R., and Okazaki, N. (2024). Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities (Swallow).** arXiv:2404.17790. URL: https://arxiv.org/abs/2404.17790 (HTML v1 read for numbers). Accessed 2026-11-19. **Preprint on the arXiv page. I could not confirm a venue (see Could not verify).**

- "approximately 100B tokens were sampled for the training data of continual pre-training". "we adopted the 9:1 JA:EN ratio".
- Japanese average (Table 4), Llama 2 → Swallow: 7B 32.0 → 39.4. 13B 39.6 → 46.3. 70B 48.3 → 55.3.
- English: "the English scores are 2--5 points lower". Averages fell 49.0 → 44.0 (7B), 54.0 → 49.2 (13B) and 62.7 → 60.4 (70B).
- The abstract says Japanese performance kept rising with data up to 100B tokens. Vocabulary expansion did not hurt accuracy except on summarisation, and it cut Japanese token count by 56.2%.
- **Verdict: SUPPORTS** (about +7 points), at a cost of 2–5 English points with only 10% English replay.

**10. Cui, Y., Yang, Z., and Yao, X. (2023). Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca.** arXiv:2304.08177. URL: https://arxiv.org/abs/2304.08177. Accessed 2026-11-19. **Preprint (no venue listed on arXiv).**

- Adds "an additional 20,000 Chinese tokens" and uses "secondary pre-training using Chinese data". Reports that "the newly proposed model markedly enhances the original LLaMA's proficiency in understanding and generating Chinese content."
- **Verdict: SUPPORTS.** This is an early recipe: vocabulary extension, then CPT, then instruction tuning.

**11. Zhao, J., Zhang, Z., Gao, L., Zhang, Q., Gui, T., and Huang, X. (2024). LLaMA Beyond English: An Empirical Study on Language Capability Transfer.** arXiv:2401.01055. URL: https://arxiv.org/abs/2401.01055. Accessed 2026-11-19. **Preprint.**

- Uses 1,440 GPU hours. "comparable performance to state-of-the-art transfer models can be achieved with less than 1% of the pretraining data". "the experimental outcomes across the thirteen low-resource languages also exhibit similar trends".
- **Verdict: SUPPORTS.** It also suggests that knowledge benchmarks saturate with modest CPT, so more data does not keep paying off.

### D. Recipes for avoiding forgetting

**12. Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., and Rish, I. (2024). Simple and Scalable Strategies to Continually Pre-train Large Language Models.** *Transactions on Machine Learning Research* (TMLR), 2024. arXiv:2403.08763. URL: https://arxiv.org/abs/2403.08763. OpenReview: https://openreview.net/forum?id=DimPeeCxKO. Accessed 2026-11-19. **Peer-reviewed (TMLR, confirmed via mlanthology.org).**

- Core method: "a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data". It matches full retraining. The English→German shift was tested at 405M parameters.
- Replay: "even the lowest tested replay of 1% significantly reduces forgetting" (weak shift, where 5% was used). For a language shift, "choosing 25% replay to handle the starker shift significantly reduces the amount of forgetting". 50% replay ("an extreme amount") slows adaptation.
- "rewarming the learning rate appears to be a significant cause of the increase in loss". The paper proposes infinite LR schedules as an alternative.
- **Verdict: QUALIFIES.** Forgetting is real, but replay plus a re-warm/re-decay schedule controls it.

**13. Elhady, A., Agirre, E., and Artetxe, M. (2025). Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation.** ACL 2025 (main). arXiv:2506.00288. URL: https://arxiv.org/abs/2506.00288. Accessed 2026-11-19. **Peer-reviewed.**

- Including English "does not impact validation perplexity" but is "critical for the emergence of downstream capabilities in the target language". The authors find "catastrophic forgetting early on CPT when English is not included". They propose "curriculum learning and exponential moving average (EMA) of weights as effective alternatives."
- **Verdict: QUALIFIES.** CPT works only if English replay (or EMA/curriculum) prevents early forgetting. Perplexity alone hides this failure.

**14. Alexandrov, A., Raychev, V., Müller, M. N., Zhang, C., Vechev, M., and Toutanova, K. (2024). Mitigating Catastrophic Forgetting in Language Transfer via Model Merging.** arXiv:2407.08699. URL: https://arxiv.org/abs/2407.08699. Accessed 2026-11-19. **Preprint on arXiv (no venue listed on the page).**

- "such language adaptation is often accompanied by catastrophic forgetting of the base model's capabilities". The Branch-and-Merge (BaM) method "can significantly reduce forgetting while matching or even improving target domain performance" (tested on Bulgarian and German). The abstract has no numbers.
- **Verdict: QUALIFIES.** Merging is a mitigation for forgetting.

**15. Huang, S.-C., Li, P.-Z., Hsu, Y.-C., Chen, K.-M., Lin, Y. T., Hsiao, S.-K., Tsai, R. T.-H., and Lee, H.-y. (2024). Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages.** ACL 2024. arXiv:2310.04799. URL: https://arxiv.org/abs/2310.04799. Accessed 2026-11-19. **Peer-reviewed.**

- "The chat vector is derived by subtracting the weights of a pre-trained base model" from its chat model. Adding it to a CPT model gives "chat capabilities in new languages without the need for further training."
- **Verdict: SUPPORTS** as a cheap post-CPT step. EstLLM (source 7) applies it.

**16. Yamaguchi, A., Villavicencio, A., and Aletras, N. (2024–2025). How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?** Accepted to *Computational Linguistics*. arXiv:2406.11477. URL: https://arxiv.org/abs/2406.11477. Accessed 2026-11-19. **Peer-reviewed (accepted, per arXiv comment).**

- "Vocabulary expansion with target language tokens is a widely used cross-lingual vocabulary adaptation approach". The authors find workable strategies with "only 30K sentences". The aim is "faster inference, while striving to maintain competitive downstream performance to baselines".
- **Verdict: QUALIFIES.** Vocabulary expansion mainly buys efficiency, not accuracy.

---

## Recipe that the evidence supports for SE.LLMA

- **Do not expect CPT on its own to raise Högskoleprovet scores.** The one published test is SweSAT-1.0 (source 1). There, AI Sweden's Llama-3-8B, continued on 227B Nordic tokens (source 2), scored 40.40 against 43.18 for base Llama 3 8B. Large multilingual models (Gemma-2, GPT-4o) scored far higher. The exam's quantitative sections stayed below 50% for every model. Base-model reasoning and scale matter more than extra Swedish text.
- **Use CPT where the evidence shows gains: Swedish fluency, Sweden-specific facts and domain knowledge.** Swedish CPT improves factual knowledge (source 3) and journalism-domain generation (source 4). For Finnish, Norwegian, Estonian and Japanese, CPT gave about +4 to +16 points on benchmark averages (Poro 2 +10.8 and +3.9, EstLLM +15.5 and +4.4, Swallow about +7), and up to +24.7 points on single Norwegian tasks (sources 5, 6, 7, 9). Gains are largest for English-centric or smaller base models (6, 7). Expect smaller gains on a strongly multilingual base such as Gemma 3, by analogy with Apertus at +4.4 points (source 7) and Poro 2 70B at +4 points (source 6).
- **Always mix in replay data.** Successful recipes kept the target language at roughly 25–90% of tokens: Poro 2 used 30% Finnish with 30% English, 30% code and 10% maths (source 6). EstLLM used about 24% Estonian (source 7). Swallow used 90% Japanese and still lost 2–5 English points (source 9). For a language shift, Ibrahim et al. found 25% replay works and 50% slows learning (source 12). Elhady et al. show that leaving out English causes early collapse of in-context learning that perplexity does not reveal (source 13). Include maths and code, because Poro 2 lost 23 GSM8K points (source 6).
- **Re-warm and re-decay the learning rate.** Use a re-warm then re-decay schedule (source 12). Poro 2 used 3e-4 for 8B and 1.5e-4 for 70B with cosine decay (source 6). Re-warming itself causes forgetting, so pair it with replay (source 12).
- **Plan token budgets in the tens to low hundreds of billions.** Successful recipes used 35.7B (EstLLM), 100B (Swallow), 165B (Poro 2) and 250B tokens (NorMistral, about 55,000 MI250X GPU-hours). That is roughly 10³–10⁵ GPU-hours. More Nordic tokens did not help on SweSAT (227B tokens in source 2). Some evidence suggests far less data is enough for knowledge benchmarks (source 11, a preprint).
- **Follow CPT with SFT and preference tuning, then merge.** Poro 2 (source 6) and EstLLM (source 7) both run CPT, then SFT (EstLLM's SFT data is mostly English), then DPO-style preference tuning. Adding a chat vector or merging with the original instruct model recovers lost instruction-following (sources 7, 15). Branch-and-Merge reduces forgetting during CPT (source 14).
- **Vocabulary extension is optional.** It saves inference cost (Swallow cut Japanese tokens by 56.2%; NorMistral's sequences are 30% shorter) but does not by itself add accuracy (sources 9, 16). If used, realign the embeddings first with the rest of the model frozen (source 5). Gemma's large vocabulary probably makes this less needed for Swedish (an inference, not tested here).
- **Evaluate beyond one benchmark.** Track Swedish (SweSAT, Kunz's Sweden-facts set), English, maths and code before and after each stage. Standard Swedish benchmarks can miss real domain gains (source 4), and CPT can lower some benchmarks while raising others (source 5).

---

## Could not verify

- **Swallow venue.** I could not confirm on any opened page that arXiv:2404.17790 appeared at COLM 2024. It is treated as a preprint above.
- **Poro 2 paper.** "Poro 2: Continued Pretraining for Language Acquisition" (Zosa et al., LumiOpen, 2025) is cited on the model cards, but I found no arXiv or peer-reviewed version. The numbers above come from the AMD ROCm blog only.
- **Viking 7B/13B/33B (LumiOpen/Silo AI).** Only model cards and blogs were found. No paper, and no CPT gains to report (Viking was trained from scratch).
- **EuroLLM-9B (arXiv:2506.04079), Salamandra, Occiglot, LLaMAntino.** These were not opened in this pass, so they are not cited. EuroLLM and Salamandra are trained from scratch, so they are not CPT evidence anyway.
- **EEVE ("Efficient and Effective Vocabulary Expansion", Korean, arXiv:2402.14714).** Not opened.
- **SnakModel (Danish).** Already known to the author. Not re-opened here.
- **Numerical details of BaM (source 14) and of Borggren et al. (source 4).** The abstracts give no numbers. The EuroEval figures for BonLM 8B and Ministral 3 8B (e.g. ScaLA 59.85 vs 51.70) appeared only in a search-result summary, and I did not open the table.
- **Any 2025–2026 Swedish model release with reported CPT gains.** No dated Swedish model release with published benchmark gains was found. BonLM 8B appears in source 4, but I did not open a model card for it.
