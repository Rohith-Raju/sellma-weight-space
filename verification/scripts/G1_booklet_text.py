"""Sanity check for claims 1-2: every question stem and option text in questions.jsonl occurs in the text of
its source booklet PDF (whitespace/hyphen-insensitive)."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, re, collections, pymupdf
B = _REPO + ''
norm = lambda s: re.sub(r'[\s\-­_]+', '', s).lower()
cache = {}
miss = collections.Counter(); ex = []
qs = [json.loads(l) for l in open(f'{B}/exams/questions.jsonl')]
for q in qs:
    b = q['source']['booklet']
    if b not in cache: cache[b] = norm(''.join(p.get_text() for p in pymupdf.open(f'{B}/{b}')))
    t = cache[b]
    parts = {'stem': q['question']} | {f'opt{k}': v for k, v in q['options'].items()}
    for k, v in parts.items():
        v2 = norm(v)
        if v2 and v2[:60] not in t:
            miss[k[:3]] += 1
            if len(ex) < 8: ex.append((q['id'], k, v[:60]))
print('questions', len(qs), 'missing pieces', dict(miss)); print(ex)
