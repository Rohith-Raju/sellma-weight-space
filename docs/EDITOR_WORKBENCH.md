# Editor workbench: manual for the editor

You are the **editor**. Your job is to change the weights of Gemma 3 4B it by hand so that it
answers more Swedish Högskoleprovet questions correctly, and to hand in that change.

You get three things:

| | what | cost |
|---|---|---|
| Weights | every parameter of the model as tensors, the tokenizer, the config | free |
| Practice data | 870 practice questions with the right answers and the unedited model's answers | free |
| Forward passes | run the model (optionally with a candidate edit) on questions or on your own texts | **1 pass per call, from a fixed budget** |

Everything is one Python object, `Workbench`. You write scripts that import it and run them on the
GPU machine:

```bash
cd /root/sellma && uv run python my_script.py
```

## The rules

1. **No gradients.** You may not compute a gradient of the model by any route: no autograd, no
   backward pass, no finite differences through extra model runs outside the workbench, no
   training or fine-tuning code. The workbench never gives you the model object; do not load the
   model yourself (not with transformers, not with bench.backends, not with anything else).
2. **Forward passes only through the workbench.** `wb.answer` and `wb.read` are the only ways you
   may run the model. Each call costs one pass. When the budget is spent, they refuse.
3. **Practice data only.** Use only what `wb.practice_questions()` gives you. Do not open
   `exams/`, `experiments/`, `runs/` or any other file with exam questions or model answers. The
   hidden papers your edit will be judged on are not yours to see.
4. **Your run folder only.** Your state lives in `editor_runs/<your run name>/`. Do not edit or
   delete files there (the ledger, the pass folders), and do not create a second run to get a
   fresh budget. Put your own scripts and notes somewhere else, or in a subfolder `notes/`.
5. **Hand in once.** `wb.submit(edit, explanation)` works one time and closes the run.

What is free and unlimited: any numpy/torch arithmetic on the weights (CPU, or GPU with autograd
off): projections, SVDs, dot products with embeddings, products of weight matrices, and so on.
Only running the model costs passes. The GPU is shared with other jobs; keep your own GPU
arithmetic small.

Everything you do through the workbench is logged in `ledger.jsonl`.

## Starting

```python
from bench.editor import Workbench

wb = Workbench.create("my_run", budget=25)   # once, at the very start
# in every later script or session:
wb = Workbench.open("my_run")

print(wb.budget, wb.passes_used, wb.passes_left)
```

`create` refuses if the run exists. The budget cannot be changed afterwards, and it is counted
from the ledger on disk, so a new Python process continues where the last one stopped.

## The model in brief

Gemma 3 4B it, text part: 34 layers, model width 2560, MLP width 10240, vocabulary 262,208,
weights in bfloat16. The checkpoint also holds a vision tower (`model.vision_tower...`), which
plays no part in text-only prompts.

Parameter names per layer `L` (0–33), with shapes `[out, in]`:

```
model.language_model.embed_tokens.weight                       [262208, 2560]
model.language_model.layers.L.input_layernorm.weight           [2560]
model.language_model.layers.L.self_attn.{q,k,v}_proj.weight
model.language_model.layers.L.self_attn.o_proj.weight
model.language_model.layers.L.self_attn.{q,k}_norm.weight
model.language_model.layers.L.post_attention_layernorm.weight  [2560]
model.language_model.layers.L.pre_feedforward_layernorm.weight [2560]
model.language_model.layers.L.mlp.gate_proj.weight             [10240, 2560]
model.language_model.layers.L.mlp.up_proj.weight               [10240, 2560]
model.language_model.layers.L.mlp.down_proj.weight             [2560, 10240]
model.language_model.layers.L.post_feedforward_layernorm.weight [2560]
model.language_model.norm.weight                               [2560]
```

One layer computes, for the residual stream `x`:

```
x = x + post_attention_layernorm(attention(input_layernorm(x)))
h = pre_feedforward_layernorm(x)
a = gelu(gate_proj h) * (up_proj h)          # the 10240 "MLP neuron activations"
x = x + post_feedforward_layernorm(down_proj a)
```

So **column `j` of `down_proj.weight` is what neuron `j` writes** (before the layer's output
norm), and row `j` of `gate_proj`/`up_proj` is what it reads. After the last layer:
`logits = unembedding · norm(x)`.

**Input and output embeddings are tied.** There is one matrix,
`model.language_model.embed_tokens.weight`: row `t` is both the input embedding of token `t`
(multiplied by √2560 when it enters the model) and the output direction whose dot product with
the final state is the logit of token `t`. An edit to it changes both roles at once.

## 1. Weights (free)

```python
w = wb.weights()
w.names()                       # all 883 parameter names
name = wb.layer_param(23)       # 'model.language_model.layers.23.mlp.down_proj.weight'
name = wb.layer_param(23, "mlp.gate_proj.weight")
w.shape(name), w.dtype(name)    # (2560, 10240), 'bfloat16'
W = w.float32(name)             # float32 copy on CPU
W_raw = w[name]                 # as stored (bfloat16)

wb.config                       # the model's config.json as a dict
wb.tokenizer                    # the Hugging Face tokenizer
wb.token_ids(" A")              # token ids of a text, no BOS added
wb.embedding_rows(" Stockholm") # [n tokens, 2560] float32, one row per token of the text
wb.unembedding_rows([562])      # same matrix for Gemma (tied); a list of token ids also works
wb.embedding_name, wb.unembedding_name, wb.embeddings_tied
```

The weights are read from the `.safetensors` files; no model is loaded. Every access reads the
tensor again and gives you your own copy, so keep a reference to tensors you reuse (the embedding
matrix is 1.3 GB in bfloat16, 2.7 GB in float32). Nothing you do to a tensor changes the model:
the only way to change the model is an *edit* passed to a pass or to `submit`.

## 2. Practice data (free)

```python
practice = wb.practice_questions()      # 870 items: 720 from 'drill' papers, 150 from 'validation' papers
item = practice[0]
item.id, item.paper                     # '2013-10-26/p1/ORD/1', 'drill'
q = item.question                       # .section ('ORD' | 'LAS' | 'MEK'), .question, .options {'A': ...},
                                        # .answer (correct letter), .passage / .passage_title (LAS only)
b = item.baseline                       # the unedited model on this question:
b.chosen_letter, b.is_correct           #   what it answered
b.letter_probs                          #   {'A': p, ...} next-token probability of each letter
b.prob_on_letters, b.prob_correct       #   their sum; share of it on the correct letter

wrong = [item for item in practice if not item.baseline.is_correct]     # 246 of 870

text = wb.render_prompt(item)                    # the exact prompt the bench gives the model
text = wb.render_prompt(item, "CABDE")           # same question, options shown in another order
```

The three sections: ORD (pick the synonym of a word), LAS (questions on a reading passage, long
prompts), MEK (fill the gaps in a sentence). The 'drill' and 'validation' papers are different
exam dates; the names only mean that you can, if you like, develop on one and check on the
other. Both are yours to use freely.

**How an answer is scored.** The model sees the rendered prompt, which ends with the instruction
to answer with the letter only and the opened model turn. The probability of a letter is the
next-token probability summed over every vocabulary token that is that letter once whitespace is
stripped ("A", " A", "A\n", ...). The chosen letter is the one with the highest probability; the
question is correct if that is the key's letter. Nothing is generated or sampled.

**Option order.** `options_order="CABDE"` lists the *original* letters in the order they are
shown: the original option C is shown as A, the original A as B, and so on. The correct letter
moves with its option. Use this to tell "the model knows the answer" from "the model likes a
letter".

## 3. Forward passes (budgeted)

Each call below costs **exactly one pass**, however many items it holds, so fill your passes. A
call that is refused or fails (bad input, budget spent, crash) costs nothing and returns nothing.
A pass loads the model in a fresh process: about 10 s of overhead, then the items one by one.

### `wb.answer(questions, edit=None, options_orders=None)` — up to 512 practice questions

```python
subset = wrong[:200] + [item for item in practice if item.baseline.is_correct][:312]
answers = wb.answer(subset)                                   # the unedited model
answers = wb.answer(subset, edit=my_edit)                     # the model with my_edit added
answers = wb.answer([item, item], options_orders=[None, "EDCBA"])   # a question may appear several times

a = answers[0]
a.question_id, a.section, a.paper, a.options_order
a.correct_letter, a.chosen_letter, a.is_correct      # letters as shown in that prompt
a.chosen_original                                    # the chosen option's letter in the stored question
a.letter_probs, a.prob_on_letters, a.prob_correct
accuracy = sum(a.is_correct for a in answers) / len(answers)
```

`questions` may be the practice items, their `Question` records or their ids. Anything that is
not a practice question is refused. With `edit=None` and no reordering you get exactly the
baseline you already have for free, so do not spend a pass on that.

### `wb.read(texts, edit=None, layers=None, top_k=20)` — up to 1,024 texts you write

```python
texts = ["Huvudstaden i Sverige heter", wb.render_prompt(item)]
r = wb.read(texts, layers=[10, 23], top_k=20)

r.top_tokens[0]      # ['Stockholm', ...] the 20 most likely next tokens after texts[0], decoded
r.top_ids, r.top_probs   # [n texts, 20] their ids and probabilities
r.n_tokens           # token count of each text
r.layers             # [10, 23]
r.mlp_acts           # [n texts, 2 layers, 10240]  MLP neuron activations = the input of down_proj,
                     #                             at the last position of each text
r.residual           # [n texts, 2 layers, 2560]   the residual stream after that layer, last position
```

All tensors are float32 on the CPU. `mlp_acts[i, k]` is the vector `a` in the layer formula
above for layer `layers[k]`, so `down_proj.weight @ a` is that MLP's raw output and you can work
out for free what a change to `down_proj` does to it. `residual[i, k]` is `x` after the whole
layer `layers[k]`. Only the last position is returned: to look inside a text, cut it where you
want to look and pass the prefix as its own text. A text may be at most 4,096 tokens. BOS is
added in front of a text unless the text already starts with `<bos>` (rendered prompts do).
Without `layers` you get the next-token distribution only. Asking for many layers on many texts
makes large results (1,024 texts × 34 layers is about 1.7 GB), which is allowed.

### Results are kept

Every finished pass is stored in `editor_runs/<run>/passes/NNN_<kind>/`. If your script dies after
a pass, the result is not lost:

```python
answers = wb.load_pass(3)     # the result of your third pass again, free
wb.ledger()                   # every pass so far: time, kind, n_items, edit summary, passes_left
```

## 4. Edits

An edit is a dict from parameter name to a change that is **added** to that parameter:

```python
import torch

name = wb.layer_param(23)                        # [2560, 10240]

# a) dense: a delta with the parameter's full shape
edit = {name: torch.zeros(2560, 10240)}

# b) some columns: values[:, j] is added to column cols[j]      (values: [n rows, len(cols)])
edit = {name: {"cols": [17, 4021, 9000], "values": torch.zeros(2560, 3)}}

# c) some rows: values[i] is added to row rows[i]               (values: [len(rows), n cols])
edit = {name: {"rows": [5, 6], "values": torch.zeros(2, 10240)}}
```

An edit may hold several parameters, each in any of the three forms. `"rows"` also works on a
vector such as a norm weight (`values: [len(rows)]`). Values may be any float tensor or array;
indices must be distinct.

```python
edit = wb.make_edit(edit)     # checks names, shapes and indices; returns float32 CPU tensors
size = wb.edit_size(edit)
size["weights_touched"]       # non-zero entries of your deltas
size["weights_changed"]       # entries that really differ after rounding to bfloat16 (see below)
size["share_of_model"]        # weights_touched / all 4.3 billion weights
size["tensors"][name]         # per tensor: form, delta_norm, tensor_norm, relative_norm (Frobenius)
```

`answer`, `read` and `submit` check an edit the same way, so calling `make_edit` yourself is
optional. During a pass the edit is added in memory for that pass only; edits never accumulate.
To test A and B together, pass one edit that holds both.

**Rounding.** The model's weights are bfloat16, about 3 significant decimal digits. Your delta is
added in float32 and the sum is rounded back to bfloat16, in a pass and in the final evaluation
alike. A change smaller than roughly 0.4% of the weight it is added to is rounded away
completely. `weights_changed` tells you how much of your edit survives; a spread-out edit with
tiny entries may do nothing at all, a concentrated one keeps its effect.

### Example: remove a direction from what some neurons write

```python
import torch
from bench.editor import Workbench

wb = Workbench.open("my_run")
layer = 23
name = wb.layer_param(layer)
W = wb.weights().float32(name)                      # [2560, 10240]

# A direction in the residual stream, built from token embeddings (free).
d = wb.unembedding_rows(" A")[0] - wb.unembedding_rows(" B")[0]
d = d / d.norm()

# The 64 neurons whose output column points most along it (free).
along = d @ W                                       # [10240] how much each neuron writes along d
cols = along.abs().topk(64).indices

# Subtract the direction from those columns: new column = column - (column · d) d
values = -torch.outer(d, along[cols])               # [2560, 64]
edit = {name: {"cols": cols, "values": values}}
print(wb.edit_size(edit))

# One pass: does it help on the questions the model got wrong, and what does it break?
practice = wb.practice_questions()
answers = wb.answer(practice[:512], edit=edit)
before = sum(item.baseline.is_correct for item in practice[:512])
print(before, "->", sum(a.is_correct for a in answers), "passes left:", wb.passes_left)
```

(The direction here is only an illustration. Finding out what the model actually gets wrong, and
which direction and neurons matter, is the work.)

## 5. Handing in

```python
record = wb.submit(edit, explanation="""
What I found, what the edit changes and why, what I measured on the practice questions.
""")
```

This writes `final_edit.safetensors` and `final_edit.json` (size, explanation, passes used) into
your run folder. It works **once**: a second call is refused, and so is every later pass. Submit
only the edit you tested; you do not have to use up the budget first. Your edit is then added to
the model with the same arithmetic as in your passes and scored on exam papers you have not seen,
so an edit that only fits the practice questions will not hold up.

## Errors

All are subclasses of `bench.editor.WorkbenchError`:

| error | meaning |
|---|---|
| `BudgetExhausted` | no passes left; you can still do free work and `submit` |
| `NotPracticeQuestion` | a question outside the practice set was given |
| `AlreadySubmitted` | the run is closed |
| `WorkbenchError` | bad input (the message says what), or a pass that failed and was not charged; the worker's output is in the pass folder's `worker.log` |

## Quick reference

```
Workbench.create(run_name, budget)      Workbench.open(run_name)
wb.budget  wb.passes_used  wb.passes_left  wb.submitted  wb.ledger()  wb.load_pass(n)

free      wb.weights() -> .names() .shape(n) .dtype(n) [n] .float32(n) .total_weights()
          wb.config  wb.tokenizer  wb.token_ids(text)  wb.layer_param(layer, part)
          wb.embedding_rows(text | ids)  wb.unembedding_rows(text | ids)
          wb.embedding_name  wb.unembedding_name  wb.embeddings_tied
          wb.practice_questions()  wb.render_prompt(question, options_order=None)
          wb.make_edit(edit)  wb.edit_size(edit)

1 pass    wb.answer(questions[<=512], edit=None, options_orders=None) -> [PracticeAnswer]
          wb.read(texts[<=1024], edit=None, layers=None, top_k=20)    -> ReadResult

once      wb.submit(edit, explanation)
```
