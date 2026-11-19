"""G1 claims 5-7: recompute standard-run metrics from gpu_results/runs/*/answers.jsonl, generations.jsonl,
damage.json, run.json; cross-check against questions.jsonl/split.json and then against summary.json."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, glob, os, re, hashlib, collections
B = _REPO + ''
qs = {json.loads(l)['id']: json.loads(l) for l in open(f'{B}/exams/questions.jsonl')}
split = json.load(open(f'{B}/exams/split.json'))['splits']
qsha = hashlib.sha256(open(f'{B}/exams/questions.jsonl', 'rb').read()).hexdigest()
ssha = hashlib.sha256(open(f'{B}/exams/split.json', 'rb').read()).hexdigest()

def loops(reply):
    words = [re.sub(r'[^\w]', '', w) for w in reply.lower().split()]
    words = [w for w in words if w]
    c = collections.Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return any(v >= 3 for v in c.values())

pct = lambda a, b: 100 * a / b if b else float('nan')
for d in sorted(glob.glob(f'{_RES}/runs/2*')):
    name = os.path.basename(d)
    if 'dry-' in name: continue
    run = json.load(open(f'{d}/run.json'))
    ans = [json.loads(l) for l in open(f'{d}/answers.jsonl')]
    probs = []
    ids = [a['question_id'] for a in ans]
    sp = run['split']
    issues = []
    if sorted(ids) != sorted(split[sp]['question_ids']): issues.append('question set != split')
    if len(set(ids)) != len(ids): issues.append('duplicate ids')
    if run.get('questions_sha256') != qsha: issues.append('questions sha differs')
    if run.get('split_sha256') != ssha: issues.append('split sha differs')
    acc = collections.defaultdict(lambda: [0, 0])
    for a in ans:
        q = qs[a['question_id']]
        if a['correct_letter'] != q['answer']: issues.append(f"key mismatch {a['question_id']}")
        if a['section'] != q['section']: issues.append('section mismatch')
        lp = a['letter_probs']
        if sorted(lp) != sorted(q['options']): issues.append(f"letters {a['question_id']}")
        arg = max(lp, key=lp.get)
        if arg != a['chosen_letter']: issues.append(f"chosen!=argmax {a['question_id']}")
        isc = a['chosen_letter'] == q['answer']
        if isc != a['is_correct']: issues.append('is_correct wrong')
        if abs(sum(lp.values()) - a['prob_on_letters']) > 1e-6: issues.append('pol != sum')
        probs.append(sum(lp.values()))
        for k in (a['section'], 'HEAD' if a['section'] in ('LAS', 'MEK') else None, 'ALL'):
            if k: acc[k][0] += isc; acc[k][1] += 1
    dm = json.load(open(f'{d}/damage.json'))
    gens = [json.loads(l) for l in open(f'{d}/generations.jsonl')]
    lr = pct(sum(loops(g['reply']) for g in gens), len(gens))
    f = lambda k: pct(*acc[k])
    print(f"{run['model']:15s} {sp:11s} n={len(ans)} head {f('HEAD'):.1f} | {f('ORD'):.1f}/{f('LAS'):.1f}/{f('MEK'):.1f}"
          f" | ppl {dm.get('perplexity'):.4g} (docs {dm.get('n_perplexity_docs')}) | loop mine {lr:.0f}% ({len(gens)} gens) damage.json {dm.get('loop_rate_percent')}"
          f" | mean pol {sum(probs)/len(probs):.6f} min {min(probs):.4f}")
    # compare with summary.json
    sm = json.load(open(f'{d}/summary.json'))
    sh = sm['headline']['accuracy'] * 100; ss = {k: v['accuracy'] * 100 for k, v in sm['sections'].items()}
    if abs(sh - f('HEAD')) > 1e-9 or any(abs(ss[k] - f(k)) > 1e-9 for k in ss): issues.append('summary.json differs')
    # weights
    desc = run['describe']
    w = {k: v for k, v in desc.get('model_files', {}).items() if k.endswith('.safetensors')}
    if desc.get('sha256'): w = {os.path.basename(desc['path']): desc['sha256']}
    print('    weights:', {k: v[:10] for k, v in w.items()}, '| path', desc['path'], '| dtype', desc.get('dtype'), '| skip_damage', run.get('skip_damage'), '| limit', run.get('limit'))
    print('    issues:', issues[:10], len(issues))
