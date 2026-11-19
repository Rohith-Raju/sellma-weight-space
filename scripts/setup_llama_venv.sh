#!/usr/bin/env bash
# Run on the GPU machine, once: makes the separate Python environment that llama.cpp's
# convert_hf_to_gguf.py needs (bench.compress calls it). llama.cpp pins older library versions
# than the bench uses, so the two must not share an environment.
#
#   scripts/setup_llama_venv.sh
set -euo pipefail

LLAMA_CPP_DIR=/root/llama.cpp
VENV=/root/llama-venv

[ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
# unsafe-best-match: the requirements file mixes PyPI with the PyTorch CPU index.
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
  -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
echo "converter environment ready at $VENV"
