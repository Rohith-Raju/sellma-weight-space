"""Render answer-key PDFs to PNG for visual (human) reading. Usage: G1_render_keys.py OUTDIR pdf..."""
import sys, os, pymupdf
out = sys.argv[1]
for p in sys.argv[2:]:
    for i, pg in enumerate(pymupdf.open(p)):
        name = p.split('/')[-2] + '_' + os.path.basename(p)[:-4] + f'_{i}.png'
        pg.get_pixmap(dpi=110).save(os.path.join(out, name)); print(name)
