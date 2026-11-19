# Position / selection bias in MCQ: verified references (collected 2026-11-19)

All references below were opened today (arXiv abstract via API plus arXiv full-text HTML unless noted); quotes are copied from the fetched text. Two caveats on verification: the NAACL/ACL Anthology records for [Pezeshkpour24] and [Wang24] were confirmed from search-result listings only, with their quotes taken from the arXiv versions; [Yang25] was read on the Anthology abstract page only.

## 1. Selection / position bias in MCQ

**[Zheng24]** Zheng, C., Zhou, H., Meng, F., Zhou, J., Huang, M. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. ICLR 2024 (Spotlight). https://arxiv.org/abs/2309.03882 (accessed 2026-11-19)
- **Definition and cause:** "they prefer to select specific option IDs as answers (like 'Option A')"; "this behavioral bias primarily stems from LLMs' token bias". 20 LLMs, on MMLU, ARC and CSQA.
- **Direction is model-specific:** on 1,000 balanced MMLU items, "llama-30B selects A/B/C/D 34.6% / 27.3% / 22.3% / 15.8% of the time, while gpt-3.5-turbo for 22.5% / 25.6% / 32.3% / 19.6%".
- **An A-averse model exists:** in the answer-moving table (0-shot MMLU), falcon-inst-40B goes from 51.5 to 38.3 (-13.3) with gold at A, 38.9 (-12.7) at B, 55.7 (+4.1) at C, 69.1 (+17.6) at D.
- **Across domains:** "Selection bias within the same model displays a moderate similarity across different domains", but "the preference ranking may not strictly persist across tasks or domains".
- **Position vs token:** "the model's position bias is somewhat present but quite irregular, largely depending on models and tasks".
- **Relation:** supports the existence and size of the effect and that an instruction-tuned model can under-pick A. Contrasts with our maths reversal, since they report mostly stable within-model preferences. Languages are not studied.

**[Pezeshkpour24]** Pezeshkpour, P., Hruschka, E. (2024). Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions. Findings of NAACL 2024, pp. 2006–2017. https://aclanthology.org/2024.findings-naacl.130 ; preprint https://arxiv.org/abs/2308.11483 (accessed 2026-11-19)
- **Size:** "a considerable performance gap of approximately 13% to 75% in LLMs on different benchmarks, when answer options are reordered".
- **Uncertainty link:** "this sensitivity arises when LLMs are uncertain about the prediction between the top-2/3 choices, and specific options placements may favor certain prediction between those top choices depending on the question caused by positional bias".
- **Calibration:** two approaches give "up to 8 percentage points improvement".
- **Relation:** supports, and is direct prior art for, "bias grows with uncertainty". The arXiv v1 covers GPT-4 and InstructGPT only.

**[Robinson23]** Robinson, J., Rytting, C. M., Wingate, D. (2023). Leveraging Large Language Models for Multiple Choice Question Answering. ICLR 2023. https://arxiv.org/abs/2210.12353 (accessed 2026-11-19)
- **Finding:** "The LLM needs what we term multiple choice symbol binding (MCSB) ability. This ability varies greatly by model."
- **Metric:** Proportion of Plurality Agreement over all n! orderings; "Simply changing the order of the candidate answers changes the model's answer."
- **Relation:** foundational background; no direction-of-bias analysis.

**[Wang24]** Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., Sui, Z. (2024). Large Language Models are not Fair Evaluators. ACL 2024, pp. 9440–9450. https://aclanthology.org/2024.acl-long.511 ; https://arxiv.org/abs/2305.17926 (accessed 2026-11-19)
- **Finding:** "Vicuna-13B could beat ChatGPT on 66 over 80 tested queries with ChatGPT as an evaluator" by changing order alone.
- **Fix:** "Balanced Position Calibration, which aggregates results across various orders".
- **Relation:** position bias in pairwise judging, and prior art for order-averaging.

**[Alzahrani24]** Alzahrani, N., et al. (2024). When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards. ACL 2024. https://arxiv.org/abs/2402.01781 (accessed 2026-11-19)
- **Finding:** "changing the order of choices or the method of answer selection, result in changes in rankings up to 8 positions"; "selection bias is apparent in all LLMs we test both in 0 and 5-shot setups".
- **On Zheng's shuffling test:** "we find it inconclusive as variance in ΔRStd is large".
- **Relation:** supports; shows the token-vs-position split is unsettled.

**[Gupta24]** Gupta, V., Pantoja, D., Ross, C., Williams, A., Ung, M. (2024). Changing Answer Order Can Decrease MMLU Accuracy. arXiv:2406.19470. https://arxiv.org/abs/2406.19470 (accessed 2026-11-19)
- **Gemma number:** Gemma-7B-it scores 51.7 on MMLU and 38.0 on their retest metric, a 26.5% drop.
- **Maths:** "the models struggled the most with problem-solving subdatasets, such as high school mathematics. For Gemma-7B and Falcon-40B models, the drop in accuracy on these categories were as high as 40%."
- **Relation:** supports that Gemma-family models are order-sensitive and that maths is the worst category. It does not report direction.

**[Wei24]** Wei, S.-L., Wu, C.-K., Huang, H.-H., Chen, H.-H. (2024). Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models. Findings of ACL 2024. https://arxiv.org/abs/2406.03009 (accessed 2026-11-19)
- **Finding:** PaLM 2, Gemini Pro and GPT-3.5 "are more sensitive to option order than to symbols/tokens in 17 out of 18 cases".
- **Difficulty:** "more challenging tasks, characterized by lower accuracy, tend to exhibit greater sensitivity".
- **Relation:** supports the uncertainty/difficulty link; contrasts with Zheng on token vs order.

**[Labruna25]** Labruna, T., Gallo, S., Da San Martino, G. (2025). Positional Bias in Binary Question Answering: How Uncertainty Shapes Model Preferences. arXiv:2506.23743 (search results indicate a CLiC-it 2025 version, not opened). https://arxiv.org/abs/2506.23743 (accessed 2026-11-19)
- **Finding:** "positional bias is nearly absent under low-uncertainty conditions, but grows exponentially when it becomes doubtful to decide which option is correct."
- **Gemma-3-12B (4-bit Q4_K_M), Italian SQuAD-it-2, accuracy correct-first vs wrong-first:**

| Condition | Correct first | Correct second |
|---|---|---|
| Low uncertainty | 0.918 | 0.907 |
| Medium uncertainty | 0.695 | 0.662 |
| High uncertainty | 0.411 | 0.590 |
| WebGPT | 0.736 | 0.563 |
| Winning Arguments | 0.302 | 0.823 |

- **Task dependence:** "in the Winning Argument dataset ... all models systematically perform better when the correct answer is presented second."
- **Quantization:** a 100-instance check gave "almost identical accuracy across both versions, suggesting that quantization does not substantially affect model preference or correctness".
- **Relation:** closest behavioural prior work. Same model family (Gemma 3), non-English, bias grows with uncertainty, and direction flips with task, including first-option under-picking. It is binary choice, not 4-option, and has no mechanism.

**[Tamba26]** Tamba, H. (2026). Position Bias is Hidden Behind Ceiling Effects: A Permutation Diagnostic for LLM Benchmarks. arXiv:2607.20864 (single-author preprint, not peer-reviewed). https://arxiv.org/abs/2607.20864 (accessed 2026-11-19)
- **Finding:** "Position bias turns out to be statistically detectable only within a roughly 60-95% base-accuracy Goldilocks zone"; two signatures: "monotone A-to-D decrease ... and non-monotone D-drop".
- **Relation:** supports accuracy/uncertainty dependence and subject-dependent shape. Its low-tier pattern favours A, the opposite of our Swedish result.

**[XWang24]** Wang, X., Hu, C., Ma, B., Röttger, P., Plank, B. (2024). Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think. COLM 2024. https://arxiv.org/abs/2404.08382 (accessed 2026-11-19)
- **Finding:** Gemma-7b-Inst scores 30.2 on MMLU by first-token probability vs 50.8 by text answer; "the text answer is more robust to option order changes than the debiased first token probabilities using state-of-the-art debiasing methods such as PriDe" when mismatch exceeds 50%.
- **Relation:** methodological caveat. If our A-penalty is measured on first-token logits, part of it may be a read-out artefact for an instruction-tuned Gemma.

**[Sadanandan26]** Sadanandan, B., Behzadan, V. (2026). When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models. arXiv:2603.25960. https://arxiv.org/abs/2603.25960 (accessed 2026-11-19)
- **Finding (MedGemma-4B, bfloat16, MedMCQA):** "Shuffling answer options causes the model to change predictions 59.1% of the time"; few-shot raises position bias "from 0.14 to 0.47"; "Permutation voting recovers 4 percentage points" (49.0% vs 45.1% mean per-permutation, K=4).
- **Relation:** supports large order sensitivity in a 4B Gemma-family model and that permutation voting helps. I did not verify from the paper that MedGemma-4B is built on Gemma 3; check before stating it.

## Mechanistic work

**[Lieberum23]** Lieberum, T., Rahtz, M., Kramár, J., Nanda, N., Irving, G., Shah, R., Mikulik, V. (2023). Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla. arXiv:2307.09458 (no venue listed). https://arxiv.org/abs/2307.09458 (accessed 2026-11-19)
- **Finding:** "'Correct Letter' heads, which attend from the final position to the correct label"; their query/key subspaces "represent an 'Nth item in an enumeration' feature to at least some extent", but this is "only a partial explanation".
- **Other head classes:** "'Single letter' heads, which mostly attend to a single fixed letter".
- **Relation:** mechanistic basis for why a late-layer attention edit can change letter read-out. It does not study bias.

**[Wiegreffe25]** Wiegreffe, S., Tafjord, O., Belinkov, Y., Hajishirzi, H., Sabharwal, A. (2025). Answer, Assemble, Ace: Understanding How LMs Answer Multiple Choice Questions. ICLR 2025 (Spotlight). https://arxiv.org/abs/2407.15018 (accessed 2026-11-19)
- **Finding:** "the prediction of a specific answer symbol is causally attributed to a few middle layers, and specifically their multi-head self-attention mechanisms"; later promotion is "associated with a sparse set of attention heads with unique roles".
- **Scope:** Llama 3.1, Olmo 0724 and Qwen 2.5 on MMLU, HellaSwag and a Colors task. For Olmo 7B Instruct, "layer 24 also plays the key causal role".
- **Relation:** supports locating the effect in a specific attention layer. It studies correct behaviour, not bias; no Gemma, no non-English.

**[Tulchinskii24]** Tulchinskii, E., et al. (2024). Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models. arXiv:2410.02343 (arXiv comment: "Accepted for NeurIPS 2026"). https://arxiv.org/abs/2410.02343 (accessed 2026-11-19)
- **Finding:** "select-and-copy heads in middle layers"; a single head's QK-score "exceeds the model's own zero-shot accuracy by up to +27.4 pp on HellaSwag"; 24 models including Gemma; "causal zero-ablation collapses MCQA accuracy to near-random".
- **Relation:** supports that the model internally knows the answer better than its letter output shows, consistent with a read-out bias that can be fixed downstream.

## 2. Debiasing methods

**Permutation averaging (prior art for our option averaging).**
- [Zheng24] defines Cyclic Permutation: it "reduces the computational cost ... from ×n! to ×n and ensures one pairing between each option ID d_i and option content o_j". PriDe "achieves superior debiasing effectiveness and performance improvement to Full/Cyclic Perm, especially in the low-cost scenario". Our 4-cyclic averaging is exactly their baseline, and PriDe is the cheaper alternative to cite.
- [Wang24] Balanced Position Calibration, [Sadanandan26] permutation voting, and [Choi25]'s "voting accuracy" are the same idea.

**[Zhao21]** Zhao, T. Z., Wallace, E., Feng, S., Klein, D., Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML 2021. https://arxiv.org/abs/2102.09690 (accessed 2026-11-19)
- **Finding:** instability "arises from the bias of language models towards predicting certain answers, e.g., those that are placed near the end of the prompt or are common in the pre-training data"; contextual calibration with a content-free input improves accuracy "up to 30.0% absolute".
- **Relation:** prior art for output-prior calibration, the cheapest alternative to our edit.

**[Reif24]** Reif, Y., Schwartz, R. (2024). Beyond Performance: Quantifying and Mitigating Label Bias in LLMs. NAACL 2024. https://arxiv.org/abs/2405.02743 (accessed 2026-11-19)
- **Finding:** "substantial label bias in models both before and after debiasing attempts", across 279 classification tasks and ten LLMs.
- **Relation:** calibration does not fully remove bias.

**[Li25]** Li, R., Gao, Y. (2025). Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions. Findings of ACL 2025. https://arxiv.org/abs/2405.03205 (accessed 2026-11-19)
- **Finding:** GPT-2 models "consistently favour the first choice 'A'"; "certain value vectors in the MLP ... and specific attention heads pay more weight on the 'A' position".
- **Fix:** they "update these critical value vectors in the MLP" and recalibrate "attention patterns by swapping the attention weight between the anchored position and the correct answer choice position"; the dominant heads "are also distributed closer to the final layer".
- **Relation:** closest prior work to our edit. The differences are the model (GPT-2), the bias direction (toward A, not away), and no multilingual or uncertainty analysis.

**[Choi25]** Choi, H. K., Xu, W., Xue, C., Eckman, S., Reddy, C. K. (2025). Mitigating Selection Bias with Node Pruning and Auxiliary Options. ACL 2025 (Main). https://arxiv.org/abs/2409.18857 (accessed 2026-11-19)
- **Method:** Bias Node Pruning "removes just 0.002% of the model's weights" in the final output projection.
- **Result:** combined with an "I don't know" option, Llama-3 on ARC-Challenge goes "from 52.3% to 65.3%".
- **Observations:** "Llama-3 prefers option 'D' while Bloomz prefers 'A'. This imbalance in preference is more pronounced in the questions in which the models give incorrect answers"; "The bias is strongest in the final layers".
- **Relation:** closest prior work on permanent weight modification, and supports "bias larger when wrong/uncertain". Their target is the unembedding, ours is mid/late attention.

**[Zhou24]** Zhou, H., Feng, Z., Zhu, Z., Qian, J., Mao, K. (2024). UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation. NeurIPS 2024. https://arxiv.org/abs/2405.20612 (accessed 2026-11-19)
- **Finding (Llama-2 7B, ARC):** "we identify a biased attention head that demonstrates a position bias consistently favoring the first option regardless of the ground truth labels of the test samples ... or changes in the sequence of options".
- **Method:** "identifies and eliminates biased FFN vectors and attention heads" at inference.
- **Relation:** close prior work, using head masking for position bias. Again toward the first option.

**[Yang25]** Yang, Z., Jian, P., Li, C. (2025). Option Symbol Matters: Investigating and Mitigating Multiple-Choice Option Symbol Bias of Large Language Models. NAACL 2025, pp. 1902–1917. https://aclanthology.org/2025.naacl-long.95/ (accessed 2026-11-19)
- **Finding:** per the abstract page, a small set of attention heads causes symbol bias, with "a tuning-free, causal effect driven debiasing method" intervening on their activations.
- **Relation:** close prior work (attention-head intervention), but for symbol bias, not position.

**[Adila24]** Adila, D., Zhang, S., Han, B., Wang, Y. (2024). Discovering Bias in Latent Space: An Unsupervised Debiasing Approach (SteerFair). ICML 2024. https://arxiv.org/abs/2406.03631 (accessed 2026-11-19)
- **Method:** "finds the bias direction in the model's representation space and steers activation values away from it during inference", exploiting "the spurious association between the first option and correctness likelihood".
- **Result:** beats a 100-label supervised baseline "by an average of 10.86% accuracy points".
- **Relation:** the activation-steering counterpart of our edit, on attention-head activations. Tested mostly on vision-language models.

**[Guda25]** Guda, B., Francis, L., Ashungafac, G. Z., Joe-Wong, C., Busogi, M. (2025). Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA Fine-Tuning and Efficient Majority Voting Approach. arXiv:2511.21709 (arXiv comment: accepted to IJCNLP-AACL). https://arxiv.org/abs/2511.21709 (accessed 2026-11-19)
- **Finding:** "majority voting, though effective, is computationally prohibitive"; proposes a label-free permutation metric and LoRA fine-tuning.
- **Relation:** prior art for weight-level debiasing by fine-tuning, with permutation voting as the reference standard.

**Position bias generally (not MCQ letters).**
- **[Liu23]** Liu, N. F., et al. (2023). Lost in the Middle. TACL. https://arxiv.org/abs/2307.03172: "performance is often highest when relevant information occurs at the beginning or end".
- **[Hsieh24]** Hsieh, C.-Y., et al. (2024). Found in the Middle. Findings of ACL 2024. https://arxiv.org/abs/2406.16008: "U-shaped attention bias"; attention calibration gains "up to 15 percentage points".
- **[ZWang24]** Wang, Z., et al. (2024). Eliminating Position Bias of Language Models: A Mechanistic Approach (PINE). arXiv:2407.01100. https://arxiv.org/abs/2407.01100: attributes position bias to "causal attention and relative positional encodings".
- All accessed 2026-11-19. They give a generic architectural reason, but mostly predict primacy/recency advantages, not a first-position penalty.

## 3. Non-English / multilingual

**[Bandarkar24]** Bandarkar, L., et al. (2024). The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. ACL 2024, pp. 749–775. https://arxiv.org/abs/2308.16884 (accessed 2026-11-19)
- **Protocol:** "We present the passage, question, and four possible answers, and instruct the model to provide the letter of the answer"; in-context: "Within the answers {A, B, C, D}, we determine the prediction to be the one with the highest probability".
- **Option order:** the full text contains no occurrence of "shuffl", "permut", "position bias", "selection bias", "option order" or "answer order". The paper does not control for or discuss option order.
- **Relation:** our Belebele rotation result fills a gap in the benchmark's own protocol.

**[Menschikov25]** Menschikov, M., et al. (2025). Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs. arXiv:2505.16134. https://arxiv.org/abs/2505.16134 (accessed 2026-11-19)
- **Finding:** "Position bias is primarily model-driven but shows language-specific nuances. Notably, Qwen2.5-7B-Instruct, DeepSeek 7B Chat and Mistral 7B consistently favor late positions challenging the common assumption of universal early-token preference."
- **Scope:** five languages (English, Russian, German, Hindi, Vietnamese); includes Gemma-7B-it.
- **Relation:** supports "direction depends on model, magnitude on language". It concerns context-passage position in QA, not answer options.

**[Azime24]** Azime, I. A., et al. (2024). ProverbEval: Exploring LLM Evaluation Challenges for Low-resource Language Understanding. arXiv:2411.05049. https://arxiv.org/abs/2411.05049 (accessed 2026-11-19; abstract only, HTML unavailable)
- **Finding:** "performance variances of up to 50%, depending on the order in which answer choices were presented in multiple-choice tasks."
- **Relation:** supports that order effects can be larger in low-resource languages.

[Labruna25] above is Italian. I found nothing on option-order bias in Swedish or on Swedish exams, and no paper comparing option-position bias on parallel multilingual MCQ such as Belebele.

## 4. Quantization

I found no study that measures whether quantization changes MCQ option-position or selection bias. The nearest evidence:

**[Dutta24]** Dutta, A., Krishnan, S., Kwatra, N., Ramjee, R. (2024). Accuracy is Not All You Need. NeurIPS 2024. https://arxiv.org/abs/2407.09141 (accessed 2026-11-19)
- **Finding:** quantization schemes "have negligible difference in accuracy (0–2%)" yet "exhibit large number of flips (up to 13.6%)"; counting incorrect-to-incorrect transitions would raise flips further, e.g. "43% in MMLU".
- **Relation:** quantization changes which MCQ answers are chosen even when accuracy is flat, so a shift in option preference is plausible but unmeasured.

**[Marchisio24]** Marchisio, K., et al. (2024). How Does Quantization Affect Multilingual LLMs? Findings of EMNLP 2024. https://arxiv.org/abs/2407.03211 (accessed 2026-11-19)
- **Finding:** "languages are disparately affected by quantization, with non-Latin script languages impacted worst; and ... challenging tasks like mathematical reasoning degrade fastest"; "a 1.7% average drop in Japanese across automatic tasks corresponds to a 16.0% drop reported by human evaluators".

**[Dettmers22]** Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 2022. https://arxiv.org/abs/2208.07339 (accessed 2026-11-19)
- **Finding:** "highly systematic emergent features in transformer language models that dominate attention and transformer predictive performance"; outlier dimensions are kept in 16-bit "while still more than 99.9% of values are multiplied in 8-bit".
- **Relation:** general mechanism by which quantizing a few attention-relevant dimensions can shift behaviour.

**[Hong24]** Hong, J., et al. (2024). Decoding Compressed Trust. ICML 2024. https://arxiv.org/abs/2403.15447 (accessed 2026-11-19)
- **Finding:** "a 4-bit quantized model retains the trustworthiness of its original counterpart"; "extreme quantization to very low bit levels (3 bits) tends to reduce trustworthiness significantly."

**[Gonçalves23]** Gonçalves, G., Strubell, E. (2023). Understanding the Effect of Model Compression on Social Bias in Large Language Models. EMNLP 2023. https://arxiv.org/abs/2312.05662 (accessed 2026-11-19)
- **Finding:** "quantization showed a regularizer effect". This concerns social bias, not position.

**[Ferrara26]** Ferrara, E. (2026). QuantiBias: Benchmarking Quantization-Induced Bias in LLMs. arXiv:2607.21063 (preprint). https://arxiv.org/abs/2607.21063 (accessed 2026-11-19)
- **Finding:** a quantized model "still selects the unbiased multiple-choice answer" yet "volunteers stereotypes in all eight languages"; backbones are Qwen and Gemma. This is social bias; MCQ is used only as a control.

Two incidental data points: [Labruna25]'s 100-item check found 4-bit and full-precision Gemma-3-12B "almost identical"; and a search-result summary of Gupta, A., et al. (2026), arXiv:2609.32867 (https://arxiv.org/abs/2609.32867) says a position contrast for Qwen3-4B was -7.8 pp at 4-bit vs -11.1 pp at full precision. I confirmed that paper's abstract but not that passage. Neither suggests quantization amplifies position effects.

## Is our finding known, partly known, or new?

Partly known.

- **Known:** selection bias is large, model-specific in direction, and need not favour A ([Zheng24] falcon-inst-40B, [Choi25] Llama-3 prefers D, [Menschikov25]). It grows when the model is uncertain or wrong ([Pezeshkpour24], [Choi25], [Wei24], [Labruna25], [Tamba26]). Cyclic-permutation averaging removes it at n× cost ([Zheng24]). Editing or masking late-layer attention/MLP components reduces it ([Li25], [Zhou24], [Choi25], [Yang25], [Adila24]).
- **Closest behavioural precedent:** [Labruna25] shows a quantized Gemma-3-12B in Italian with negligible position bias at low uncertainty and a first-option penalty at high uncertainty, with direction changing by task. That is binary choice only.
- **Apparently new:** (a) a first-letter (A) penalty in 4-option MCQ for Gemma 3 4B, quantified on Swedish material and on parallel Swedish/English Belebele, which the Belebele paper never controls for; (b) a sign reversal of the preference within one model between reading/exam items and MMLU maths, where [Zheng24] reports only "moderate" cross-domain similarity and [Gupta24] reports that maths is most order-sensitive without direction; (c) an attention weight edit in a Gemma model that repairs an anti-A bias, compared head-to-head with cyclic averaging. Prior mechanistic edits all target a pro-first/pro-A bias in GPT-2, Llama-2 or Llama-3.
- **How to frame the edit:** since option averaging matched or beat it, and the literature already treats permutation averaging as the reference debiaser, the edit is best presented as mechanistic evidence, not as a better debiasing method.
- **Quantization:** there is no literature to cite either way; any claim there would be ours alone.