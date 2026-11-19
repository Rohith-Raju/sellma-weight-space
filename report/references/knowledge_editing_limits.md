# Knowledge editing: what it can and cannot do, verified references (collected 2026-11-19)

I opened and checked all 20 sources below. Arxiv quotes come from the full PDF text or the arXiv API abstract. The blog and Hugging Face card quotes come from the raw HTML and the README. Accessed 2026-11-19.

## Verified references

**Primary sources (the blog and its replication)**

**[Takehi2026]** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL) (2026, Nov 2). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* Blog post, not peer-reviewed. https://rikiyat.github.io/blog/weight-space-programming/
- Exact sentence, verbatim: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked."
- Context: "Fixing loops, however, is not the same as making a model better at math: it lets the model finish answers it could already work out."
- Gemma 3 270M "rarely loops: only 8.5% of its answers reach its token limit, against 93% for Qwen. Here, Astra could not improve it." Astra found "ordinary reasoning errors, such as wrong arithmetic and invalid algebraic steps, with no single component behind them to switch off". "none moved MATH-500 by more than about a point, a gain Astra itself reported as not established."
- Gemma 3 270M MATH-500 (%): base 6.8; edited after 5, 10, 25 and 50 forward passes: 7.0, 7.6, 7.7, 7.0.
- **Supports** our conclusion directly. It is the closest analogue: small targeted edits, a Gemma 3 model, and a null result on capability.

**[Coletti2026]** JonathanColetti (2026). *Qwen3.5-0.8B-Astra-WeightEdit*, Hugging Face model card. https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit
- Verbatim: "This removes a failure mode; it doesn't add capability. Almost all of the gain comes from answers that no longer loop to the token limit."
- The edit touches 64 output columns of `mlp.down_proj` in layers 15 and 19, about 130k weights (0.016% of the model).
- With thinking on, base vs. edited, score (share of answers hitting the token limit): MATH-500 7.4 (93%) → 50.2 (32%); GSM8K 13.0 → 67.6; IFEval 19.2 → 62.1.
- With thinking off, the edit costs about 1.5–2 points.
- The uploader is an unaffiliated third party who applied the authors' released diff. Cite it as an independent replication, not as the authors' own claim.
- **Supports** our conclusion.

**The counterpoint: small edits can insert facts**

**[ROME]** Meng, Bau, Andonian, Belinkov (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. https://arxiv.org/abs/2202.05262
- Abstract: "we modify feed-forward weights to update specific factual associations using Rank-One Model Editing (ROME)". On counterfactuals it "simultaneously maintains both specificity and generalization".
- Table 4 (CounterFact), GPT-J: ROME efficacy 99.9, paraphrase 99.1, neighborhood 78.9 (unedited model: 83.0). GPT-2 XL: efficacy 100.0, paraphrase 96.4, neighborhood 75.4.
- §3.7 Limitations: "it only edits a single fact at a time, and it is not intended as a practical method for large-scale model training. Associations edited by ROME are directional". Also: "we have not investigated other kinds of learned beliefs such as logical, spatial, or numerical knowledge." And: "Even when a model's stored factual association is changed successfully, the model will guess plausible new facts that have no basis in evidence and that are likely to be false."
- **Contradicts** any unscoped claim that small edits cannot add knowledge. Its limitations section **qualifies** that.

**[MEMIT]** Meng, Sen Sharma, Andonian, Belinkov, Bau (2023). *Mass-Editing Memory in a Transformer.* ICLR 2023 (stated in the PDF header). https://arxiv.org/abs/2210.07229
- Abstract: "it can scale up to thousands of associations for GPT-J (6B) and GPT-NeoX (20B)".
- Table 1, 10,000 zsRE edits on GPT-J: efficacy 96.7, paraphrase 89.7, specificity 26.6 (unedited: 27.0). The paper says "zsRE tests MEMIT's ability to add correct information."
- Limitations: "limited in scope to working with directional (s, r, o) relations: it does not cover spatial or temporal reasoning, mathematical knowledge, linguistic knowledge, procedural knowledge, or even symmetric relations."
- **Contradicts** the unscoped claim (it inserts thousands of real facts). Its scope statement **qualifies** it: exam reasoning and linguistic knowledge are explicitly outside what it covers.

**Inserted facts fail to propagate or support reasoning**

**[Cohen2024]** Cohen, Biran, Yoran, Globerson, Geva (2024). *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. https://arxiv.org/abs/2307.12976
- Abstract: "current methods fail to introduce consistent changes in the model's knowledge".
- Table 3 (RECENT subset), GPT-J with ROME: logical generalization 15.2, compositionality I 29.5, subject aliasing 90.3. GPT-J with MEMIT: 18.0, 35.0, 88.4.
- An in-context editing baseline on LLaMA averages 82.8, against 60.8 for ROME.
- **Supports**: edits change the specific queried association, not the connected knowledge around it.

**[Yao2023]** Yao, Wang, Tian, Cheng, Li, Deng, Chen, Zhang (2023). *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. https://arxiv.org/abs/2305.13172
- Table 1, CounterFact on GPT-J: ROME reliability 99.80, generalization 86.63. Table 3, portability on GPT-J (subject-replace / reverse-relation / one-hop): ROME 37.42 / 46.42 / 50.91; MEMIT 27.73 / 47.67 / 52.74.
- Quote: "the performance of current model editing methods regarding portability is somewhat suboptimal."
- **Supports.**

**[MQuAKE]** Zhong, Wu, Manning, Potts, Chen (2023). *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP 2023. https://arxiv.org/abs/2305.14795
- Quote: "current knowledge-editing approaches can recall edited facts accurately, they fail catastrophically on the constructed multi-hop questions."
- "Even with the strongest baseline approach, MEMIT, multi-hop performance changes from 40.5% →7.0% with GPT-J and 30.2% →4.9% with Vicuna-7B." Edit-wise accuracy for MEMIT on GPT-J is 96.2.
- Also: "these models struggle to utilize the edited knowledge internally."
- **Supports**: an inserted fact is not usable for reasoning, which is what exams require.

**Degradation from many edits**

**[Gupta2024]** Gupta, Rao, Anumanchipalli (2024). *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024. https://arxiv.org/abs/2401.07453
- Quote: "as the model is edited sequentially with multiple facts, it continually forgets previously edited facts and the ability to perform downstream tasks."
- For ROME, the decline in efficacy "can come as early as 100 edits, or as late as 1000 edits". For MEMIT, "The catastrophic forgetting phase begins after approximately 1400 edits".
- Also: "edits made by ROME and MEMIT are not as localized as previously believed to be."
- **Qualifies** the counterpoint: closing many knowledge gaps one edit per fact does not scale.

**[Gu2024]** Gu, Xu, Ma, Lu, Ling, Chang, Peng (2024). *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP 2024. https://arxiv.org/abs/2401.04700
- Abstract: "it is challenging for current editing methods to simultaneously improve factuality of LLMs and maintain their general abilities." The authors attribute this to "model editing altering the original model weights excessively, leading to overfitting to the edited facts."
- Conclusion: current methods "unintentionally hurt the general abilities of LLMs no matter in instance- or batch-editing, and single- or sequential-editing."
- **Qualifies.**

**Editing may override or mimic outputs rather than rewrite knowledge**

**[Hase2023]** Hase, Bansal, Kim, Ghandeharioun (2023). *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023 (Spotlight). https://arxiv.org/abs/2301.04213
- Table 1: edit layer explains R² = 0.947 of the variance in ROME rewrite score; tracing effect explains 0.016; both together 0.948. "the tracing effect is able to explain only 0.1% of the variance in edit success".
- The authors suggest "it is possible to 'override' the information in layer ℓ with an edit to another layer k". Abstract: "better mechanistic understanding … may not always translate to insights about how to best change their behavior."
- **Supports, indirectly.** Edits succeed without touching where the fact is stored, which is consistent with overriding the output rather than changing stored knowledge. The paper does not itself claim "editing changes behaviour, not knowledge", so do not attribute that wording to it.

**[Xie2025]** Xie, Cao, Chen, Liu, Zhao (2025). *Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing.* ACL 2025. https://arxiv.org/abs/2505.12636
- Abstract: despite "near-perfect performance on conventional metrics, the models edited by them are still prone to generating original knowledge."
- On LLaMA3-8B under the Wiki attack, PMET and AlphaEdit reach high efficacy "yet simultaneously maintain high OM metrics of 70.33% and 72.53%". OM is the rate of reverting to the original knowledge; lower is better.
- **Supports.**

**[Gu2026]** Gu, Huang, Hong, Xie, Lou, Zhang (2026). *The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models.* Findings of ACL 2026. https://arxiv.org/abs/2604.05995
- Quote: "editors achieve high benchmark scores by merely mimicking target outputs without structurally overwriting internal beliefs". In their multiple-choice self-assessment test, the edited model "reverts to the parametric answer".
- **Supports.** It is especially relevant because our exams are multiple choice.

**[Yang2025]** Yang, Sun, Tan, Ma, Cao, Yin, Shen, Cheng (2025). *The Mirage of Model Editing: Revisiting Evaluation in the Wild.* ACL 2025. https://arxiv.org/abs/2502.11177
- Quote: "current editing methods perform substantially worse than previously reported (38.5% vs. 96.8%)". Approaches "fail drastically with only 1000 edits." They trace the inflation to teacher forcing during testing.
- **Qualifies** the ROME/MEMIT headline numbers.

**[Liu2025]** Liu, Xu, Liu, Deng, Wang, Wang, Li, Teh, Lee (2025). *Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation.* arXiv preprint, self-described as "a work in progress". https://arxiv.org/abs/2510.00625
- Quotes: "state-of-the-art approaches collapse even under the simplest negation queries" and "editing is likely to be based on shortcuts rather than full semantics."
- **Supports.** Flag it as a preprint.

**[Zhang2025]** Zhang, Ye, Liu, Ren, Wu, Chen (2025). *Uncovering Overfitting in Large Language Model Editing.* ICLR 2025. https://arxiv.org/abs/2410.07819
- Abstract: "Editing Overfit, where edited models assign disproportionately high probabilities to the edit target, hindering the generalization of new knowledge in complex scenarios".
- **Supports.**

**Steering and direction edits: behaviour changes, measured knowledge does not**

I found no paper that flatly says "steering cannot add capability". The closest evidence is that steering or a rank-one direction edit moves behaviour a lot while knowledge benchmarks stay flat.

**[Arditi2024]** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. https://arxiv.org/abs/2406.11717
- Removing one direction from the weights disables refusal. Yet "for MMLU, ARC, and GSM8K, orthogonalized models perform similarly to baseline models."
- Gemma 7B, edited / base: MMLU 51.8 / 51.7, ARC 51.7 / 51.5, GSM8K 31.3 / 32.0.
- **Supports**: a tiny weight edit changes behaviour a lot and leaves knowledge scores flat, and the model is from the Gemma family.

**[Panickssery2024]** Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner (2024). *Steering Llama 2 via Contrastive Activation Addition.* arXiv 2312.06681. The PDF I opened carries no venue line; it is commonly cited as ACL 2024, but I did not verify that. https://arxiv.org/abs/2312.06681
- Quote: CAA "significantly alters model behavior … and minimally reduces capabilities."
- Table 5, MMLU on Llama 2 13B Chat: base 0.63; steered with ±1 multipliers, 0.57–0.65.
- **Supports**: behaviour shifts, MMLU does not improve.

**[Tan2024]** Tan, Chanin, Lynch, Kanoulas, Paige, Garriga-Alonso, Kirk (2024). *Analyzing the Generalization and Reliability of Steering Vectors.* NeurIPS 2024. https://arxiv.org/abs/2407.12404
- Quote: "many behaviours turn out to be unsteerable". Also: "steering vectors have substantial limitations both in- and out-of-distribution."
- **Qualifies**: steering is unreliable even for behaviour.

**[Li2023-ITI]** Li, Patel, Viégas, Pfister, Wattenberg (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS 2023 (spotlight). https://arxiv.org/abs/2306.03341
- On Alpaca, ITI "improves its truthfulness from 32.5% to 65.1%." LLaMA 7B shows "a full 40% difference between probe accuracy and generation accuracy". "LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface."
- **Qualifies, and partly supports.** An intervention can raise benchmark accuracy a lot. The authors frame that gain as closing the gap between knowledge the model already has and what it outputs, which is a change in selection, not new knowledge.

## How to phrase the conclusion defensibly

Do not claim that small weight edits cannot add knowledge. ROME (CounterFact efficacy 99.9 on GPT-J) and MEMIT (10,000 real zsRE facts, 96.7% efficacy) show they can insert individual factual associations.

The defensible claim has three parts. Scope it to our edits, make it about the exams, and use the limits literature to say why one-fact-per-edit insertion would not close the gaps:

> "The targeted edits we applied changed how Gemma 3 4B selects among answer options on Swedish exams, not what it knows. The questions it still gets wrong reflect missing knowledge that these edits did not supply. This mirrors Takehi et al. (2026): editing 'could remove an obstacle that hid an ability, but it could not add an ability the model lacked'. Fact-insertion editors such as ROME and MEMIT can write individual subject–relation–object facts into weights. But each edit covers one directional fact, and inserted facts rarely spread to related questions (logical generalization 15–18% on RippleEdits; multi-hop accuracy 7.0% after MEMIT on MQuAKE). Edited models often fall back to the original answer in multiple-choice or adversarial probes. Thousands of sequential edits degrade the model. So we do not expect editing to close knowledge gaps spread across the exams, and we did not test per-fact insertion."

The last clause matters. If our edits were not ROME/MEMIT-style fact insertion, say so, so the claim is "these edits did not close the gaps" rather than "editing cannot".