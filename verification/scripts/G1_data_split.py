"""G1 claims 1-3: recompute from exams/questions.jsonl, exams/split.json, exams/raw/."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, os, re, collections, glob, hashlib
R = _REPO + '/exams'
qs = [json.loads(l) for l in open(f'{R}/questions.jsonl')]
print('questions:', len(qs), 'unique ids:', len({q['id'] for q in qs}))
print('sha256 questions.jsonl:', hashlib.sha256(open(f'{R}/questions.jsonl','rb').read()).hexdigest())
print('sha256 split.json:', hashlib.sha256(open(f'{R}/split.json','rb').read()).hexdigest())
exams_dirs = sorted(d for d in os.listdir(f'{R}/raw') if os.path.isdir(f'{R}/raw/{d}'))
print('exam dirs:', len(exams_dirs))
booklets = []
for d in exams_dirs:
    for f in sorted(os.listdir(f'{R}/raw/{d}')):
        if 'facit' in f or f.startswith('hp-'):
            continue
        booklets.append((d, f))
print('booklet PDFs (verbal passes found):', len(booklets))
for d, f in booklets:
    pass
kept = collections.defaultdict(list)
for q in qs:
    kept[(q['exam_date'], q['exam_pass'])].append(q)
print('kept passes:', len(kept), 'exams with kept questions:', len({k[0] for k in kept}))
kept_booklets = {q['source']['booklet'] for q in qs}
not_kept = [f'exams/raw/{d}/{f}' for d, f in booklets if f'exams/raw/{d}/{f}' not in kept_booklets]
print('booklets with no kept questions:', not_kept)
# claim 2
bad = []
for k, lst in kept.items():
    c = collections.Counter(q['section'] for q in lst)
    if c != {'ORD': 10, 'LAS': 10, 'MEK': 10}: bad.append((k, c))
    for q in lst:
        n = len(q['options']); exp = 5 if q['section'] == 'ORD' else 4
        letters = sorted(q['options'])
        if n != exp or letters != list('ABCDE'[:exp]) or q['answer'] not in q['options']:
            bad.append((q['id'], n, q['answer']))
        if any(not str(v).strip() for v in q['options'].values()): bad.append((q['id'], 'empty option'))
        if not str(q['question']).strip(): bad.append((q['id'], 'empty stem'))
        if q['section'] == 'LAS' and not q.get('passage'): bad.append((q['id'], 'LAS without passage'))
    nums = sorted(q['number'] for q in lst)
    if nums != list(range(1, 31)): bad.append((k, 'numbers', nums))
print('claim2 problems:', bad[:20], len(bad))
# answer letter distribution
for s in ['ORD','LAS','MEK']:
    print(s, collections.Counter(q['answer'] for q in qs if q['section']==s))
# claim 3
sp = json.load(open(f'{R}/split.json'))
ids_by_exam = collections.defaultdict(list)
for q in qs: ids_by_exam[q['exam_date']].append(q['id'])
allids = {q['id'] for q in qs}
seen = set()
for name, s in sp['splits'].items():
    ids = s['question_ids']
    ex = collections.Counter(i.split('/')[0] for i in ids)
    secs = collections.Counter(i.split('/')[2] for i in ids)
    print(name, 'n=', len(ids), 'unique', len(set(ids)), 'exams', len(ex), dict(ex) if len(ex) < 6 else '', 'listed exams', len(s.get('exams', [])), dict(secs))
    assert set(ids) <= allids
    print('  overlap with earlier splits:', len(seen & set(ids)))
    seen |= set(ids)
    if name == 'dev': print('  dev exams:', sorted(ex)); print('  LAS+MEK:', secs['LAS']+secs['MEK'])
print('union covers all questions:', seen == allids)
