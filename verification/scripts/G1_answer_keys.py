"""G1 claim 4 (automatic part): independent positional parse of every answer-key PDF and comparison
with exams/questions.jsonl. Assigns each (number, letter) pair to the nearest 'Provpass N' header
above-left of it. Also checks that all key files for one exam that carry a pass agree."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, os, re, collections, sys
import pymupdf
R = _REPO + '/exams'

def parse_key(path):
    cols = collections.defaultdict(dict)   # pass -> {num: letter}
    kinds = {}
    for pg in pymupdf.open(path):
        ws = [list(w[:5]) for w in pg.get_text('words')]
        # some keys split two-digit numbers into separate one-digit words ("1","1" for 11): merge them
        ws.sort(key=lambda w: (round(w[1]), w[0]))
        merged = []
        for w in ws:
            m = merged[-1] if merged else None
            if (m and re.fullmatch(r'\d', m[4]) and re.fullmatch(r'\d', w[4]) and abs(m[1] - w[1]) < 2
                    and 0 <= w[0] - m[2] < 2.5):
                m[2] = w[2]; m[4] += w[4]
            else:
                merged.append(w)
        ws = [tuple(w) for w in merged]
        heads = []
        for i, w in enumerate(ws):
            if w[4].lower() == 'provpass':
                # pass number = next numeric word
                for j in range(i + 1, min(i + 4, len(ws))):
                    if re.fullmatch(r'\d', ws[j][4]):
                        heads.append((w[0], w[1], int(ws[j][4]))); break
        # part label ("Verbal"/"Kvantitativ") nearest below each header
        for hx, hy, p in heads:
            lab = [w for w in ws if w[4] in ('Verbal', 'Kvantitativ') and abs(w[0] - hx) < 25 and 0 < w[1] - hy < 60]
            if lab: kinds[p] = min(lab, key=lambda w: w[1] - hy)[4]
        nums = [w for w in ws if re.fullmatch(r'\d{1,2}', w[4])]
        lets = [w for w in ws if re.fullmatch(r'[A-E]', w[4])]
        for L in lets:
            cy = (L[1] + L[3]) / 2
            # number just left on the same row
            cand = [n for n in nums if n[2] <= L[0] + 1 and L[0] - n[2] < 40 and n[1] - 3 < cy < n[3] + 3]
            if not cand:
                # number above (vertical split layouts): same x region, within 30px above
                cand = [n for n in nums if abs(n[0] - (L[0] - 15)) < 25 and 0 < L[1] - n[1] < 30]
            if not cand: continue
            n = max(cand, key=lambda n: (n[2], n[1]))
            x, y = n[0], n[1]
            hs = [h for h in heads if h[0] <= x + 15 and h[1] < y]
            if not hs: continue
            mx = max(round(h[0] / 30) for h in hs)
            h = max((h for h in hs if round(h[0] / 30) == mx), key=lambda h: h[1])
            num = int(n[4])
            if num in cols[h[2]] and cols[h[2]][num] != L[4]:
                cols[h[2]][num] = '?'
            else:
                cols[h[2]][num] = L[4]
    return cols, kinds

qs = [json.loads(l) for l in open(f'{R}/questions.jsonl')]
by_exam = collections.defaultdict(list)
for q in qs: by_exam[q['exam_date']].append(q)
total = agree = 0; mism = []; disagree_keys = []; incomplete = []
parsed_cache = {}
for exam in sorted(by_exam):
    keyfiles = sorted(f for f in os.listdir(f'{R}/raw/{exam}') if 'facit' in f or f.startswith('hp-'))
    parsed = {f: parse_key(f'{R}/raw/{exam}/{f}') for f in keyfiles}
    for p in sorted({q['exam_pass'] for q in by_exam[exam]}):
        versions = {f: c[p] for f, (c, k) in parsed.items() if p in c}
        kinds = {f: k.get(p) for f, (c, k) in parsed.items() if p in c}
        for f, col in versions.items():
            if any(col.get(n) in (None, '?') for n in range(1, 31)):
                incomplete.append((exam, p, f, [n for n in range(1, 31) if col.get(n) in (None, '?')]))
        vals = list(versions.values())
        if any({n: v.get(n) for n in range(1, 31)} != {n: vals[0].get(n) for n in range(1, 31)} for v in vals):
            disagree_keys.append((exam, p, list(versions)))
        if set(kinds.values()) != {'Verbal'}: print('pass kind?', exam, p, kinds)
        ref = vals[0] if vals else {}
        for q in by_exam[exam]:
            if q['exam_pass'] != p: continue
            total += 1
            mine = ref.get(q['number'])
            if mine == q['answer']: agree += 1
            else: mism.append((q['id'], q['answer'], mine))
print('compared', total, 'agree', agree)
print('mismatches', mism[:40])
print('key files that disagree for a kept pass:', disagree_keys)
print('incomplete parses:', incomplete)
