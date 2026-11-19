"""G1 claim 4 (manual part): answers transcribed by eye from rendered official answer-key PDFs
(G1_render_keys.py output), compared with exams/questions.jsonl."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json
VIS = {  # (exam, pass): answers for questions 1..30, read off the PNG of the facit PDF
    ('2013-10-26', 1): 'EABEDDBACA BDBDCACDBD BADDCBCDAB',
    ('2013-10-26', 4): 'EDCDACADEC AADCACCBDC DCCACDABAA',
    ('2021-05-08', 1): 'EEDAECBABD CBCCBBDACC CDBADCADBA',
    ('2021-05-08', 4): 'DCCCAAEEBB BCBACADCBC DADBADCDBC',
    ('2026-10-18', 2): 'CADCDEECBE BDAACDBCDC ACDAABBCBB',   # test exam
    ('2026-10-18', 5): 'DCBBBADEED CDADBDCABC ADDABDCABA',   # test exam
}
qs = {json.loads(l)['id']: json.loads(l) for l in open(_REPO + '/exams/questions.jsonl')}
n = ok = 0
for (exam, p), s in VIS.items():
    s = s.replace(' ', ''); assert len(s) == 30
    for i, L in enumerate(s, 1):
        sec = 'ORD' if i <= 10 else 'LAS' if i <= 20 else 'MEK'
        q = qs[f'{exam}/p{p}/{sec}/{i}']
        n += 1; ok += q['answer'] == L
        if q['answer'] != L: print('MISMATCH', q['id'], 'pdf', L, 'jsonl', q['answer'])
print(f'visual check: {ok}/{n} match over {len({e for e, _ in VIS})} exams')
