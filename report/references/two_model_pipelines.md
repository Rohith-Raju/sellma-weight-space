# Two-component systems: a strong model reasons, a separate component handles the target language

Scope: published comparisons of a two-component system against one model that answers directly in the target language. Same-model methods are left out (self-translate, XLT prompting, Shi et al. MGSM translate-test). Liu et al. NAACL 2025 ("Is Translation All You Need?", arXiv 2403.10258) is also left out because an earlier search covered it.

All sources were opened on 2026-11-19. The quotes below are copied from the pages as the fetch tool returned them. Averages marked "computed here" were worked out from the paper's per-language numbers. They are not printed in the paper.

Labels:
- **SUPPORTS**: the two-component system beat the single model.
- **AGAINST**: the single model was as good or better.
- **MIXED**: the result depends on the model, the language or the metric.

---

## Verified references

### 1. Bu & Feng (2026), XBridge: encoder, then a frozen LLM, then the NLLB decoder writes the answer. MIXED, leaning SUPPORTS

- **Citation:** Mengyu Bu, Yang Feng. 2026. "Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality." ACL 2026 Main Conference, according to the arXiv comments field. arXiv:2603.17512 (v4). https://arxiv.org/abs/2603.17512 and https://arxiv.org/html/2603.17512. Accessed 2026-11-19.
- **Components:** A frozen LLM does the reasoning. The paper tested MetaMath-7B, LLaMA3-8B, Aya-23-8B and Qwen2.5-7B-Instruct. The second component is NLLB-200-1.3B. Its encoder reads the query. Its decoder writes the answer in the target language. Learned mapping layers join the two. The paper calls it "a compositional encoder-LLM-decoder architecture".
- **Output design:** The paper separates two outputs. "XBridge-LLM" is the LLM's English reasoning. "XBridge-Dec" is the NLLB decoder's output in the target language. The paper reports that "decoder-generated multilingual reasoning (XBridge_Dec) achieves accuracy comparable to English LLM outputs".
- **MGSM accuracy (Table 6), Bengali / Swahili / Thai and overall average:**
  - LLaMA3-8B. SFT baseline: 38.80 / 40.80 / 44.00, overall 45.64. Translate-Test: 40.40 / 46.80 / 35.20, overall 50.68. XBridge_Dec: 46.00 / 49.60 / 51.20, overall 54.48. XBridge_LLM: 47.20 / 49.60 / 54.40, overall 56.72.
  - Qwen2.5-7B-Instruct. SFT: 49.60 / 48.40 / 62.80, overall 59.12. Translate-Test: 43.20 / 58.40 / 44.00, overall 59.60. XBridge_Dec: 60.80 / 66.80 / 64.00, overall 69.08.
  - Aya-23-8B. SFT overall 44.12. Translate-Test overall 45.60. XBridge_Dec overall 46.60.
  - Low-resource average (computed here), LLaMA3-8B: SFT 41.2, Translate-Test 40.8, XBridge_Dec 48.9.
- **XL-Sum ROUGE-L (Table 7), where the decoder writes the summary in the target language:** LLaMA3-8B SFT 26.44 overall. XBridge 27.68 overall. Swahili goes from 28.73 to 30.75. Bengali drops from 20.87 to 20.65. For Qwen2.5-7B the overall gain is small (25.97 to 26.04). Translate-Test is not reported for XL-Sum.
- **Limits stated in the paper:** "the overall model still exhibits some imbalance in multilingual capabilities". The authors also note that "translation-cascaded systems are limited by the NMT model". The paper defines Translate-Test as a system that "translates inputs to English, queries the English-SFT LLM, and translates the output back to the target language". On FLORES translation (Table 1), XBridge stays below plain NLLB-200-1.3B on every low-resource direction. One example is En-Bn, where MetaMath+XBridge gets 29.23 BLEU and NLLB gets 32.83.
- **What improved:** Task score (reasoning accuracy, and to a small degree summary ROUGE). The paper has no human fluency or naturalness evaluation, so the effect on language quality is not measured.

### 2. Lorandi & Belz (2024), LLM writes English, then an MT engine translates, compared with direct generation in Irish, Welsh, Maltese and Breton. MIXED (SUPPORTS when the MT engine is strong)

- **Citation:** Michela Lorandi, Anya Belz. 2024. "High-quality Data-to-Text Generation for Severely Under-Resourced Languages with Out-of-the-box Large Language Models." arXiv:2402.12267 (v1). **Preprint**: the arXiv page names no venue. https://arxiv.org/html/2402.12267. Accessed 2026-11-19.
- **Components:** GPT-3.5 (text-davinci-003), BLOOM, LLaMa2-chat and Falcon-chat generate English text from WebNLG data. Google Translate, Alibaba Translate or NLLB then translates it. The baseline is the same LLM writing directly in the target language.
- **BLEU (Table 1), direct generation compared with English generation plus MT:**
  - GPT-3.5 Irish: direct 15.35. With Google Translate 20.40, Alibaba 18.34, NLLB 17.14.
  - GPT-3.5 Welsh: direct 18.95. With Google Translate 25.12.
  - GPT-3.5 Maltese: direct 15.43. With NLLB 17.13.
  - GPT-3.5 Breton: direct 18.59. With Alibaba 10.24. **This is worse.** Google Translate does not support Breton.
  - LLaMa2-chat Irish: direct 10.41. With Google Translate 19.34.
- **Quotes:** English plus Google Translate "has better scores than direct generation into the under-resourced language by substantial margins". For Breton, "generation into English plus Alibaba is a lot worse than direct generation in case of GPT-3.5."
- **Language quality:** The authors cite WebNLG 2023 human scores for the GPT-3.5 plus Google Translate system. They describe it as "on-a-par-with-human performance". That human evaluation covers only this one pipeline system. It is not a side-by-side human comparison against direct generation.
- **What improved:** Output text quality as measured by BLEU/chrF++. The gain depends on how good the MT engine is for the language. NLLB gave smaller gains than Google Translate.

### 3. MindMerger (Huang et al., NeurIPS 2024): mT5 encoder plus a frozen LLM, compared with the LLM alone and with an NLLB translate-test pipeline. SUPPORTS (input side only)

- **Citation:** Zixian Huang, Wenhao Zhu, Gong Cheng, Lei Li, Fei Yuan. 2024. "MindMerger: Efficiently Boosting LLM Reasoning in non-English Languages." NeurIPS 2024. https://papers.neurips.cc/paper_files/paper/2024/file/3bf80b34f731313b8292f4578e820c90-Paper-Conference.pdf and arXiv:2405.17386. Accessed 2026-11-19.
- **Components:** The paper "used the encoder part of mT5-xl ... as the multilingual model in our methods, used the NLLB200-3.3B as the translation model for baselines, and used the Llama 2-7B as the LLM for all methods". The LLM still answers in English. The math answers are numbers.
- **MGSM (Table 1), Bengali / Thai / Swahili, low-resource average, overall average:**
  - MonoReason (the single LLM, fine-tuned on English): 6.8 / 7.2 / 6.8. Low-resource 6.9. Overall 38.3.
  - Translate-En (NLLB-200-3.3B translates the question into English): 48.4 / 37.6 / 37.6. Low-resource 41.2. Overall 50.6.
  - LangBridge: low-resource 45.5. Overall 50.2.
  - MindMerger-Soft: 50.4 / 52.8 / 57.2. Low-resource 53.5. Overall 57.3.
- **Quote:** "Without updating the parameters of LLMs, the average accuracy improved by 6.7% and 8.0% across all languages and low-resource languages on the MGSM dataset". The paper also notes that the merged design "has a higher average accuracy than the replacement-based Translate-En of 4.4% and 11.8% across high-resource languages in the MGSM and MSVAMP datasets".
- **What improved:** Reasoning accuracy for questions written in non-English languages. Target-language output quality is not tested.

### 4. LangBridge (Yoon et al., ACL 2024): mT5 encoder plus an English reasoning LM. SUPPORTS for low-resource languages, AGAINST for English and high-resource languages

- **Citation:** Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, Minjoon Seo. 2024. "LangBridge: Multilingual Reasoning Without Multilingual Supervision." ACL 2024 (long paper 405). https://arxiv.org/html/2401.10695v2 (the ACL Anthology PDF, https://aclanthology.org/2024.acl-long.405.pdf, timed out). Accessed 2026-11-19.
- **Components:** An mT5-XL or mT5-XXL encoder is connected to a 13B English-specialised LM (MetaMath, Orca 2, Code Llama, Llemma). It is trained on English data only.
- **MGSM (Table 1), MetaMath-13B, then LangBridge-15B:** average 40.5 to 53.5. Bengali 6.8 to 50.0. Swahili 11.6 to 41.6. Thai 4.8 to 52.8. Telugu 0.8 to 36.4.
- **Regression:** English goes from 70.4 to 67.6 (LB-15B) and 66.4 (LB-20B). The paper says the method "may degrade performance on high-resource languages". It also says the reasoning steps "would predominantly be in English".
- **What improved:** Reasoning on low-resource input. The answer is not written in the target language, so output fluency is not tested.

### 5. IrokoBench (Adelani et al.): NLLB-200 translate-test compared with in-language prompting for African languages. MIXED (helps open models, hurts GPT-4o)

- **Citation:** David Ifeoluwa Adelani, Jessica Ojo, Israel Abebe Azime, Jian Yun Zhuang, et al. 2025 (v2). "IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models." arXiv:2406.03368v2. https://arxiv.org/html/2406.03368v2. Accessed 2026-11-19. The arXiv page names no venue. The paper may have a peer-reviewed version, but it was not checked here.
- **Components:** The paper says: "For MT, we use NLLB-200 (3.3B)". NLLB translates the African-language question into English, and the LLM answers it. This is input-side only.
- **Average scores across African languages, in-language compared with translate-test:**
  - Gemma 2 27B: AfriMMLU 39.9 to 48.8. AfriMGSM 28.5 to 46.1. AfriXNLI 42.8 to 49.0.
  - LLaMa 3.1 70B: AfriMMLU 39.4 to 51.3. AfriMGSM 24.6 to 45.6. AfriXNLI 38.0 to 42.8.
  - GPT-4o: AfriMMLU 60.0 to 54.1. AfriMGSM 52.6 to 42.6. AfriXNLI 64.3 to 52.1. **All three get worse.**
- **Quote:** On AfriMGSM, "Gemma 2 27B and LLaMa 3.1 70B improved by +17.6 and +21.0 respectively".
- **What improved:** Knowledge and reasoning scores for open models that are weak in the target language. The strongest model lost points when its input was translated.

### 6. MEGA (Ahuja et al., EMNLP 2023): Bing Translator translate-test compared with monolingual prompting. MIXED, mostly SUPPORTS for low-resource languages

- **Citation:** Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Maxamed Axmed, Kalika Bali, Sunayana Sitaram. 2023. "MEGA: Multilingual Evaluation of Generative AI." EMNLP 2023. arXiv:2303.12528. https://ar5iv.labs.arxiv.org/html/2303.12528. Accessed 2026-11-19.
- **Components:** "We use Bing Translator to translate the test examples into English." The LLMs include GPT-3.5-Turbo and text-davinci-003. This is input-side only. It uses a commercial MT engine, as Liu et al. did.
- **Numbers:** For GPT-3.5-Turbo on Burmese, Tamil and Telugu, "the relative improvement can be >30%". On Urdu XNLI, translate-test reaches "54% accuracy compared to 49.1% for monolingual". English reaches 76.2%, so most of the gap remains.
- **What improved:** Classification and reasoning scores on low-resource languages. MEGA reports that high-resource languages perform about the same either way.

### 7. TALL (Ofer, Zamler & Azaria, 2025): frozen LLM plus two bilingual Marian MT models, compared with direct Hebrew and with a naive translate pipeline. SUPPORTS, but the scale is tiny

- **Citation:** Moshe Ofer, Orel Zamler, Amos Azaria. 2025. "TALL -- A Trainable Architecture for Enhancing LLM Performance in Low-Resource Languages." arXiv:2506.05057 (v1). **Preprint.** https://arxiv.org/abs/2506.05057 and https://arxiv.org/html/2506.05057v1. Accessed 2026-11-19.
- **Components:** The LLM is Bloomz-560m or Qwen2.5-0.5B. The other components are "pre-trained Marian MT models" for Hebrew to English and English to Hebrew, joined by trainable adapters. The output is Hebrew.
- **Task:** Predict the missing last word of a Hebrew sentence.
- **Accuracy % (Table 1), WIKI-SLVM, Bloomz-560m:** Direct Hebrew 0.63. Fine-tuned 0.16. Naive translate-and-back pipeline 2.11. TALL 5.59. With Qwen2.5-0.5B: Direct 3.77, Naive 2.85, TALL 5.15. Note that the naive pipeline was **worse** than direct Hebrew for Qwen.
- **Quote:** "TALL consistently outperforms all baseline approaches, achieving up to 5.59% accuracy with bloomz-560m".
- **Caveat:** These are very small models and a narrow word-prediction task. All accuracies are below 6%. This is weak evidence for a "strong model" setting.

### 8. Robinson et al. (WMT 2023): a dedicated MT system writes low-resource languages better than ChatGPT. SUPPORTS using a dedicated writer component

- **Citation:** Nathaniel Robinson, Perez Ogayo, David R. Mortensen, Graham Neubig. 2023. "ChatGPT MT: Competitive for High- (but Not Low-) Resource Languages." Proceedings of the Eighth Conference on Machine Translation (WMT 2023), pp. 392–418. https://aclanthology.org/2023.wmt-1.40. Accessed 2026-11-19.
- **Finding:** ChatGPT models "consistently lag for low-resource languages (LRLs), under-performing traditional MT for 84.1% of languages we covered." The study uses FLORES-200 and 204 languages.
- **Relevance:** This is not a pipeline study. It shows that in most low-resource languages, the "writer" step is done better by a dedicated MT model than by a general LLM. This is the basis for the two-component idea.

### 9. Bafna et al. (IJCNLP-AACL 2025), the Translation Barrier Hypothesis. SUPPORTS the reasoning behind the design (mechanistic, not a system comparison)

- **Citation:** Niyati Bafna, Tianjian Li, Kenton Murray, David R. Mortensen, David Yarowsky, Hale Sirin, Daniel Khashabi. 2025. "The Translation Barrier Hypothesis: Multilingual Generation with Large Language Models Suffers from Implicit Translation Failure." IJCNLP-AACL 2025, pp. 1541–1568. https://aclanthology.org/2025.ijcnlp-long.83 and https://arxiv.org/abs/2506.22724v2. Accessed 2026-11-19.
- **Finding (v2 abstract):** "the translation barrier explains a dominant portion of error for a majority of language pairs". The barrier "is especially severe for low-resource target languages". The study covers word translation over 108 language pairs.
- **Relevance:** Models often solve the task but fail at the last step of putting the answer into the target language. This is the failure a separate writer component would target. The evidence comes from a word-level task only.

### 10. Guo et al. (ACL 2025), naturalness of LLM output in non-English languages. Context for measuring translationese (no pipeline comparison)

- **Citation:** Yanzhu Guo, Simone Conia, Zelin Zhou, Min Li, Saloni Potdar, Henry Xiao. 2025. "Do Large Language Models Have an English 'Accent'? Evaluating and Improving the Naturalness of Multilingual LLMs." ACL 2025, according to the arXiv comments field. arXiv:2410.15956v3. https://arxiv.org/abs/2410.15956v3. Accessed 2026-11-19.
- **Finding:** LLMs "often generate unnatural outputs in non-English languages". In French and Chinese they found "a tendency towards English-influenced patterns". The paper proposes corpus-level lexical and syntactic naturalness metrics.
- **Relevance:** Direct generation by an English-centric model already has a translationese cost. A pipeline would add MT translationese on top. No study found here measures both on the same outputs. These metrics could be used to measure them for Swedish.

### 11. Valentini et al. (COLM 2026), translationese in multilingual LLM generations. Context (no numbers in the abstract)

- **Citation:** Maria Valentini, Téa Wright, Julisa Granados, Eliana Colunga, Katharina von der Wense. 2026. "An Investigation of Translationese in the Generations of Multilingual Large Language Models." Accepted to COLM 2026. arXiv:2608.17399. https://arxiv.org/abs/2608.17399. Accessed 2026-11-19.
- **Research questions quoted from the abstract:** "Does text generated by MLLMs resemble translationese?" and "How does translationese produced by MLLMs differ from translationese produced through direct translation?"
- **Caveat:** Only the abstract was read. It gives no numbers. It does not say whether "direct translation" means machine or human translation. Treat this as a pointer only.

---

## Closest match to "strong model thinks, small model writes the target language"

- **The single closest study is XBridge (Bu & Feng, ACL 2026, ref. 1).** A frozen 7–8B LLM does the reasoning in English. A separate 1.3B NLLB decoder writes the answer in the target language. It is compared with the same LLM fine-tuned to answer directly (SFT) and with an NLLB translate-test cascade. On MGSM, the decoder-written answers beat SFT for LLaMA3-8B (overall 54.48 vs 45.64) and Qwen2.5-7B (69.08 vs 59.12). They also beat the text cascade.
- **Why it is only close, not exact.** The "strong model" is only 7–8B. The gains are task scores (MGSM accuracy, XL-Sum ROUGE-L). There is no human rating of fluency. XL-Sum gains are small, from +0.07 to +1.24 ROUGE-L overall, and Bengali got worse for LLaMA3. Swedish is not tested.
- **The closest text-level pipeline is Lorandi & Belz (2024, ref. 2).** GPT-3.5 writes English and an MT engine translates it. This beat direct generation in Irish, Welsh and Maltese, by up to about 6 BLEU with Google Translate. It lost badly in Breton (10.24 vs 18.59), where only a weaker MT engine was available. It is the clearest evidence that the result depends on how good the writer component is for the language.
- **No study found** pairs a frontier-scale reasoner with a small *target-language LLM* (rather than an MT model) that rewrites or localises the draft, and evaluates it against direct generation. Proxy-tuning (Liu et al., COLM 2024, arXiv 2401.08565) and large–small collaborative decoding exist. The searches found no language-specific evaluation of them.

## What this means for SE.LLMA

- **Expect the benefit to shrink as the main model gets stronger in the target language.** IrokoBench (ref. 5) shows that translate-test helped Gemma 2 27B and LLaMa 3.1 70B a lot but hurt GPT-4o on all three tasks. Liu et al. (excluded here) and MEGA (ref. 6) report similar patterns for high-resource languages. Swedish is a mid-to-high-resource language for frontier models. So a pipeline may help a small open model but hurt a strong one.
- **The writer component sets an upper limit.** Lorandi & Belz (ref. 2) found gains with Google Translate but losses with a weaker engine (Breton). XBridge (ref. 1) says cascades are "limited by the NMT model". It also stays below plain NLLB on translation. A Swedish writer model must be clearly better at Swedish than the reasoning model already is. Otherwise it adds errors.
- **Reasoning gains and language-quality gains are different claims.** MindMerger, LangBridge, IrokoBench and MEGA (refs. 3–6) improve the input side only (understanding the question), and their answers stay in English. XBridge and Lorandi & Belz are the only verified studies where a separate component writes the target-language text. Neither of them reports a side-by-side human fluency rating against direct generation.
- **Measure naturalness directly.** Direct generation already shows English-influenced patterns (Guo et al., ref. 10). An MT step could add its own translationese (Valentini et al., ref. 11). An SE.LLMA test should score both systems on the same prompts with native Swedish raters or corpus-level naturalness metrics, not only on task accuracy.
- **Watch for regressions on what already works.** LangBridge lost about 3–4 points in English (ref. 4). GPT-4o lost 6–12 points under translate-test (ref. 5). Any two-component design should be compared against the strong model answering directly in Swedish, not only against a weak baseline.

## Could not verify

- **AfroBench** (Ojo et al., 2025): not opened. Whether it reports NLLB translate-test numbers is unknown.
- **"Lost in Translation: Large Language Models in Non-English Content Analysis"** (Nicholas & Bhatia, CDT, 2023, arXiv 2306.07377): found by search but not opened. It is a policy report and not a system comparison.
- **LayAlign** (NAACL 2025 Findings): only seen as a baseline inside XBridge. The original paper was not opened.
- **Proxy-tuning** (Liu et al., COLM 2024, arXiv 2401.08565): found by search. No multilingual or low-resource evaluation was found, so it is not cited as evidence.
- **Ryu et al. 2026** ("Understanding LLM Behavior in Multi-Target Cross-Lingual Summarization", arXiv 2606.01252): opened, but it is excluded. Its pipelines use the *same* LLM for translation, so it is a same-model study. It found that pipelines help 2B models, while end-to-end wins for larger or more multilingual models.
- **Wang et al. 2023** ("Zero-Shot Cross-Lingual Summarization via Large Language Models", NewSum 2023): opened, but it is excluded. Its "pipeline" prompts run inside one LLM.
- **LangBridge ACL Anthology PDF**: timed out. The numbers above come from arXiv v2.
- **Bafna et al. v1 percentages** (">50% for 65% and 78% of pairs"): these appeared in a search snippet. They are not in the v2 abstract that was opened, so they are not cited.
