#!/usr/bin/env bash
# Build the full report PDF from report/report_full.md:  bash report/latex/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
pandoc -d latex/defaults.yaml report_full.md -o latex/report_full.tex
cd latex
rm -rf figures && cp -r ../figures figures
for i in 1 2; do xelatex -interaction=nonstopmode -halt-on-error report_full.tex > build.log; done
grep -q "Missing character" report_full.log && echo "WARNING: missing characters, see report_full.log" || true
cp report_full.pdf ../report.pdf
echo "wrote report/report.pdf"
