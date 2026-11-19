# Methods, models, benchmarks, statistics: verified references (collected 2026-11-19)

I verified 33 references by opening each source; none is cited from memory. Most bibliographic details come from arXiv's metadata service (abstract-page data), ACL Anthology BibTeX, Crossref, the GitHub API and the official model cards. Three gaps first:

- **No paper shows that model edits (ROME/MEMIT-style) are wiped out by quantization.** I searched arXiv and the web several ways and found none. The closest evidence is [Zhang2025] on unlearning and [Egashira2024].
- **The Qwen3.5 blog page (qwen.ai/blog?id=qwen3.5) wouldn't load**; it renders only with JavaScript. I used Alibaba's announcement, the Alibaba Cloud copy of the blog and the QwenLM GitHub README instead.
- **OpenReview pages were blocked by a browser check.** I confirmed ICLR venues through the iclr.cc poster pages, the paper PDF header or arXiv comments instead.

All URLs accessed 2026-11-19.

## 1. Models

**[Gemma3]** Gemma Team, Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., et al. (216 authors) (2025). *Gemma 3 Technical Report.* arXiv:2503.19786 (submitted 25 Mar 2025). https://arxiv.org/abs/2503.19786
- Use: the base model. The official model card's BibTeX entry (`@article{gemma_2025, author={Gemma Team}, publisher={Google DeepMind}`) points to this arXiv id.

**[Gemma3Card]** Google (2025). *Gemma 3 model card.* https://gemma.ai/models/gemma-3/ (ai.google.dev/gemma/docs/core/model_card_3 now redirects here). Hugging Face repo: https://huggingface.co/google/gemma-3-4b-it (access requires accepting Google's license).
- Supporting facts, quoted:
  - "The knowledge cutoff date for the training data was August 2024."
  - The 4B model was trained on 4 trillion tokens.
  - Input context is 128K tokens (4B/12B/27B); output is 8192 tokens.
  - "over 140 languages."
- Caveat: the Hugging Face card for gemma-3-4b-it does **not** state the cutoff. Cite the gemma.ai card for that fact.

**[Qwen3.5]** Qwen Team (2026). *Qwen3.5: Towards Native Multimodal Agents.* Blog post, February 2026. https://qwen.ai/blog?id=qwen3.5
- Release dates (QwenLM README, https://github.com/QwenLM/Qwen3.5):
  - "2026-02-16: We release Qwen3.5. The first release includes a 397B-A17B MoE model."
  - "2026-03-02: Qwen3.5-9B, Qwen3.5-4B, Qwen3.5-2B, and Qwen3.5-0.8B are now available."
- Corroborating sources:
  - Alibaba Group announcement dated 16 Feb 2026: https://www.alibabagroup.com/document-1960233590314762240. It says "201 languages and dialects, up from 119".
  - Alibaba Cloud copy of the blog dated 17 Feb 2026: https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894. It describes "Gated DeltaNet + Gated Attention hybrid attention".

**[Qwen3.5-9B]** Qwen (2026). *Qwen/Qwen3.5-9B model card.* Apache-2.0. https://huggingface.co/Qwen/Qwen3.5-9B
- Layout: 32 layers in the pattern "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))". That is 24 linear-attention layers and 8 full-attention layers.
- Other specs: hidden size 4096, FFN size 12288, native context 262,144 tokens.
- The model's config.json confirms this: `full_attention_interval: 4`, the `layer_types` field alternates linear and full attention, and there are no `num_experts` fields.
- So the **9B is dense**, even though the family-level text describes the architecture as "Gated Delta Networks combined with sparse Mixture-of-Experts". Don't call the 9B MoE.

**[GatedDeltaNet]** Yang, S., Kautz, J., Hatamizadeh, A. (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025 (camera ready). arXiv:2412.06464. https://arxiv.org/abs/2412.06464
- Use: primary citation for the linear-attention layer type inside Qwen3.5. You didn't ask for this; I added it.

## 2. Editing, steering and interpretability methods

**[LoRA]** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685 (June 2021). https://arxiv.org/abs/2106.09685 ; poster page: https://iclr.cc/virtual/2022/poster/6319
- Use: the training baseline. LoRA freezes the pretrained weights and trains low-rank update matrices.

**[ROME]** Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022 (Advances in Neural Information Processing Systems 35). arXiv:2202.05262. https://arxiv.org/abs/2202.05262 ; https://proceedings.nips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf
- Use: Rank-One Model Editing. It changes middle-layer MLP weights to rewrite one factual association. It is the closest analogue to our single-question rank-1 edit.

**[MEMIT]** Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., Bau, D. (2023). *Mass-Editing Memory in a Transformer.* ICLR 2023 (top-25% poster). arXiv:2210.07229. https://arxiv.org/abs/2210.07229 ; https://iclr.cc/virtual/2023/poster/11880 (OpenReview id MkbcAHIYgyS)
- Use: extends ROME to thousands of edits spread across several MLP layers.

**[ActAdd]** Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., MacDiarmid, M. (2023). *Steering Language Models With Activation Engineering.* arXiv:2308.10248 (v5, 10 Oct 2024). https://arxiv.org/abs/2308.10248
- v1 (20 Aug 2023) was titled *"Activation Addition: Steering Language Models Without Optimization"*. Cite the version you mean.
- Use: activation steering. The abstract says the method "contrasts the intermediate activations on prompt pairs" to build a steering vector that is added during the forward pass.
- arXiv lists no venue.

**[RepE]** Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al. (21 authors) (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405. https://arxiv.org/abs/2310.01405
- Use: representation engineering. arXiv lists no venue.

**[LogitLens]** nostalgebraist (2020). *interpreting GPT: the logit lens.* LessWrong, 31 Aug 2020. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Use: applying the final unembedding to intermediate-layer activations, which is what the AI editors' analysis did.

**[TunedLens]** Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., Steinhardt, J. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112. https://arxiv.org/abs/2303.08112
- Use: a learned per-layer refinement of the logit lens. arXiv lists no venue.

**[Elhage2021]** Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., Olah, C. (2021). *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread (Anthropic), 22 Dec 2021. https://transformer-circuits.pub/2021/framework/index.html
- Use: the view of an attention head as two parts.
  - The QK circuit "computes the attention pattern".
  - The OV circuit "computes how each token affects the output if attended to".
  - Keys, queries and values are "intermediate results in the computation of the low-rank matrices W_Q^TW_K and W_OW_V".
- This is why editing a head's key projection (editor 3) changes only the attention pattern (the QK side) and not what the head writes (the OV side).

## 3. Quantization

**[llama.cpp]** ggml-org (2023–2026). *llama.cpp* (software). https://github.com/ggml-org/llama.cpp
- Release b8680 was published 2026-04-06T19:38:24Z, commit 15f786e6581598638840276948a7e6183fc96a83: https://github.com/ggml-org/llama.cpp/releases/tag/b8680

**[kquants]** Kawrakow, I. (ikawrakow) (2023). *k-quants.* llama.cpp pull request #1684, merged 2023-06-05. https://github.com/ggml-org/llama.cpp/pull/1684
- Q4_K format: "'type-1' 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw." Here "type-1" means w = d·q + m, where d is the block scale and m is the block minimum.
- Q4_K_M mix: "uses GGML_TYPE_Q6_K for half of the attention.wv and feed_forward.w2 tensors, else GGML_TYPE_Q4_K."
- Caveat: this describes the mix as defined in 2023. The per-tensor rules in b8680 may have changed since; I did not check.

**[GGUF]** ggml-org. *GGUF* specification, docs/gguf.md in the ggml repository. https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Use: describes GGUF as a binary format for GGML-based inference and the successor to GGML/GGMF/GGJT. Single file, key-value metadata, mmap-compatible.

**[LLM.int8]** Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339. https://arxiv.org/abs/2208.07339
- Note: the NeurIPS listing (as shown by mlanthology) uses the title "GPT3.int8(): 8-Bit Matrix Multiplication for Transformers at Scale".

**[Zhang2025]** Zhang, Z., Wang, F., Li, X., Wu, Z., Tang, X., Liu, H., He, Q., Yin, W., Wang, S. (2025). *Catastrophic Failure of LLM Unlearning via Quantization.* ICLR 2025. arXiv:2410.16454. https://arxiv.org/abs/2410.16454
- Use: the closest published evidence that small weight edits can be erased by quantization.
- Abstract: unlearned models keep 21% of the "forgotten" knowledge at full precision, rising to 83% after 4-bit quantization.
- Mechanism, from the paper: "minimal changes to model weights during unlearning", so that "quantization is likely to map the weights of the target LLM and the unlearned LLM to the same values."
- Caveat: this is about unlearning, not ROME-style editing. Cite it as an analogue.

**[Egashira2024]** Egashira, K., Vero, M., Staab, R., He, J., Vechev, M. (2024). *Exploiting LLM Quantization.* NeurIPS 2024. arXiv:2405.18137. https://papers.nips.cc/paper_files/paper/2024/hash/496720b3c860111b95ac8634349dcc88-Abstract-Conference.html
- Use: the reverse effect. It shows that many full-precision weight settings map to the same quantized model, so a model can behave differently before and after quantization.

## 4. Benchmarks, data and Swedish context

**[Belebele]** Bandarkar, L., Liang, D., Muller, B., Artetxe, M., Shukla, S. N., Husa, D., Goyal, N., Krishnan, A., Zettlemoyer, L., Khabsa, M. (2024). *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* Proceedings of ACL 2024 (Vol. 1: Long Papers), pp. 749–775, Bangkok. doi:10.18653/v1/2024.acl-long.44. https://aclanthology.org/2024.acl-long.44/
- The Hugging Face dataset facebook/belebele has a `swe_Latn` config (data/swe_Latn.jsonl).

**[MMLU]** Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., Steinhardt, J. (2021). *Measuring Massive Multitask Language Understanding.* ICLR 2021. arXiv:2009.03300. https://arxiv.org/abs/2009.03300
- The venue is stated in the arXiv comment and the PDF header. The benchmark has 57 tasks.

**[SweSAT-1.0]** Kurfalı, M., Zahra, S., Gogoulou, E., Dürlich, L., Carlsson, F., Nivre, J. (2025). *SweSAT-1.0: The Swedish University Entrance Exam as a Benchmark for Large Language Models.* Proceedings of NoDaLiDa/Baltic-HLT 2025, pp. 331–339, Tallinn. University of Tartu Library. https://aclanthology.org/2025.nodalida-1.36/
- Use: the peer-reviewed precedent for using Högskoleprovet as an LLM benchmark. "867 questions across six different tasks."

**[skolprov]** Ekgren, A. (2025). *Ekgren/swedish_skolprov* (Hugging Face dataset). Created 2025-03-17, last modified 2025-04-01. https://huggingface.co/datasets/Ekgren/swedish_skolprov
- Contents: config `högskoleprovet` is the verbal section of the 2024-10-20 test (file hp_2024-10-20_VERB.csv). The dataset also has the medical (läkare), dentistry (tandläkare), audiology (audionom), pharmacy (apotekare) and math/physics tests, plus a small leaderboard.
- There is no paper; cite it as a dataset.

**[SweSAT-syn]** Adesam, Y., Borin, L. (2024). *SweSAT Synonyms v1.1 (Högskoleprovet Ordförståelse).* Språkbanken Text, University of Gothenburg. doi:10.23695/6NM7-C102. https://researchdata.se/en/catalogue/dataset/doi-10-23695-6nm7-c102
- Contents: 822 items (83 train / 739 test) taken from Högskoleprovet 2006–2021. Each item has five answer choices. "Part of the SuperLim collection."

**[Superlim]** Berdicevskis, A., Bouma, G., Kurtz, R., Morger, F., Öhman, J., Adesam, Y., Borin, L., Dannélls, D., Forsberg, M., Isbister, T., Lindahl, A., Malmsten, M., Rekathati, F., Sahlgren, M., Volodina, E., Börjeson, L., Hengchen, S., Tahmasebi, N. (2023). *Superlim: A Swedish Language Understanding Evaluation Benchmark.* Proceedings of EMNLP 2023, pp. 8137–8153, Singapore. doi:10.18653/v1/2023.emnlp-main.506. https://aclanthology.org/2023.emnlp-main.506/

**[ScandEval]** Nielsen, D. (2023). *ScandEval: A Benchmark for Scandinavian Natural Language Processing.* Proceedings of NoDaLiDa 2023, pp. 185–201, Tórshavn. University of Tartu Library. https://aclanthology.org/2023.nodalida-1.20/ ; arXiv:2304.00906
- Name note: ACL Anthology and arXiv list the author as "Dan Nielsen" / "Dan Saattrup Nielsen". The EuroEval CITATION.cff file lists "Dan Saattrup Smart".

**[EuroEval]** Smart, D. S., Enevoldsen, K., Schneider-Kamp, P. (2025). *Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks.* Proceedings of NoDaLiDa/Baltic-HLT 2025, pp. 561–572. arXiv:2406.13469.
- This is the paper the EuroEval README asks you to cite, together with [ScandEval]. Software: https://github.com/EuroEval/EuroEval (formerly ScandEval).
- Caveat: I took this entry from the README's BibTeX. I did not open the Anthology page for it.

**[GPT-SW3]** Ekgren, A., Cuba Gyllensten, A., Stollenwerk, F., Öhman, J., Isbister, T., Gogoulou, E., Carlsson, F., Casademont, J., Sahlgren, M. (2024). *GPT-SW3: An Autoregressive Language Model for the Scandinavian Languages.* Proceedings of LREC-COLING 2024, pp. 7886–7900, Torino. ELRA and ICCL. https://aclanthology.org/2024.lrec-main.695/
- The arXiv preprint 2305.12987 is titled "...for the Nordic Languages".

## 5. Statistics

**[McNemar1947]** McNemar, Q. (1947). Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. *Psychometrika* 12(2), 153–157. doi:10.1007/BF02295996
- Use: the paired test on discordant pairs. Its exact form is the binomial sign test on those pairs.

**[Dietterich1998]** Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation* 10(7), 1895–1923. doi:10.1162/089976698300017197
- Use: standard machine-learning reference for applying McNemar's test to compare two classifiers on the same items.

**[EfronTibshirani1993]** Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Monographs on Statistics and Applied Probability 57. New York: Chapman & Hall. ISBN 0412042312.
- Date note: library catalogs give 1993. The Crossref record for the Chapman & Hall/CRC edition (doi:10.1201/9780429246593) shows 1994-05-15.

**[Koehn2004]** Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. Proceedings of EMNLP 2004, pp. 388–395, Barcelona. ACL. https://aclanthology.org/W04-3250/
- Use: the paired bootstrap as used in NLP.