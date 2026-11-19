"""Compresses a Hugging Face model folder to a 4-bit GGUF file with llama.cpp.

    compress_model(hf_dir, out_gguf)            # from Python
    uv run python -m bench.compress HF_DIR OUT.gguf [--quant Q4_K_M]

Two steps, always with the same settings, so an edited model and its unedited baseline are
compressed in exactly the same way:

1. llama.cpp's convert_hf_to_gguf.py writes the weights unchanged to a temporary GGUF file in
   bfloat16, the format both models are published in and edited models are saved in. (Checked on
   both baselines: a float32 temporary file gives a byte-identical result, and needs twice the
   disk. A model saved in float32 would need INTERMEDIATE_TYPE = "f32", or small edits would be
   rounded away.) Only the language model is converted; the image encoder of these checkpoints
   is left out, as the bench is text only.
2. llama-quantize turns that file into the requested type (default Q4_K_M: about 4.8 bits
   per weight, no importance matrix, no calibration data, so nothing depends on sample text).

The temporary file is deleted afterwards. The output appears under its final name only when
it is complete, so a file that exists is a finished file. A JSON file next to it
(<name>.gguf.json) records the settings, the llama.cpp version, the converter's library
versions and the checksums of the input and the output.

The converter runs in its own Python environment (/root/llama-venv, made by
scripts/setup_llama_venv.sh), because llama.cpp pins older library versions than the bench.

Symlinks: an edited model folder holds symlinks to the shards it did not change. The
converter opens files by path, so links are followed and nothing special is needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.backends import cached_sha256, folder_checksums

LLAMA_CPP_DIR = Path("/root/llama.cpp")                       # the GPU machine's checkout (tag b8680)
CONVERT_PYTHON = Path("/root/llama-venv/bin/python")          # separate environment for the converter
INTERMEDIATE_TYPE = "bf16"       # lossless here: the weights on disk are bfloat16 (see above)
QUANTIZE_THREADS = 32            # only affects speed
CONVERTER_PACKAGES = ("torch", "transformers", "numpy", "sentencepiece", "gguf")
_LOG_TAIL_LINES = 40


def compress_model(hf_dir: Path, out_gguf: Path, quant: str = "Q4_K_M") -> Path:
    """Convert the model in `hf_dir` to `out_gguf` at quantization type `quant`. Returns `out_gguf`.

    A finished output (file and its .json record) is kept and returned as is, so a restarted
    script does not repeat the work. Delete both files to force a new compression.
    """
    hf_dir, out_gguf = Path(hf_dir), Path(out_gguf)
    record_path = _record_path(out_gguf)
    if out_gguf.exists() and record_path.exists():
        _require_same_request(record_path, hf_dir, quant)
        print(f"compress: {out_gguf} already exists, keeping it", flush=True)
        return out_gguf
    if not (hf_dir / "config.json").exists():
        raise FileNotFoundError(f"{hf_dir} is not a Hugging Face model folder (no config.json)")

    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    # Both temporary files sit next to the output: same disk, and the names cannot clash with another model's.
    intermediate = out_gguf.with_name(f".{out_gguf.name}.{INTERMEDIATE_TYPE}.tmp")
    unfinished = out_gguf.with_name(f".{out_gguf.name}.unfinished")
    convert_command = _convert_command(hf_dir, intermediate)
    quantize_command = _quantize_command(intermediate, unfinished, quant)
    try:
        convert_seconds = _run(convert_command)
        quantize_seconds = _run(quantize_command)
        unfinished.rename(out_gguf)
    finally:
        intermediate.unlink(missing_ok=True)
        unfinished.unlink(missing_ok=True)

    record = {
        "source": str(hf_dir),
        "source_files": folder_checksums(hf_dir),
        "output": str(out_gguf),
        "output_sha256": cached_sha256(out_gguf),
        "output_bytes": out_gguf.stat().st_size,
        "quant": quant,
        "intermediate_type": INTERMEDIATE_TYPE,
        "importance_matrix": None,
        "convert_command": convert_command,
        "quantize_command": quantize_command,
        "llama_cpp": llama_cpp_version(),
        "converter_packages": converter_package_versions(),
        "convert_seconds": round(convert_seconds, 1),
        "quantize_seconds": round(quantize_seconds, 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_gguf


def llama_cpp_version() -> dict[str, str | None]:
    """Tag and commit of the llama.cpp checkout that did the work."""
    return {
        "dir": str(LLAMA_CPP_DIR),
        "tag": _output_or_none(["git", "-C", str(LLAMA_CPP_DIR), "describe", "--tags", "--always"]),
        "commit": _output_or_none(["git", "-C", str(LLAMA_CPP_DIR), "rev-parse", "HEAD"]),
    }


def converter_package_versions() -> dict[str, str] | None:
    """Versions of the libraries in the converter's own environment."""
    code = (
        "import json, importlib.metadata as m\n"
        f"names = {list(CONVERTER_PACKAGES)!r}\n"
        "found = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        found[name] = m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        found[name] = 'not installed'\n"
        "print(json.dumps(found))\n"
    )
    output = _output_or_none([str(CONVERT_PYTHON), "-c", code])
    return json.loads(output) if output else None


def _convert_command(hf_dir: Path, intermediate: Path) -> list[str]:
    return [
        str(CONVERT_PYTHON), str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"), str(hf_dir),
        "--outfile", str(intermediate), "--outtype", INTERMEDIATE_TYPE,
    ]


def _quantize_command(intermediate: Path, output: Path, quant: str) -> list[str]:
    return [
        str(LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"),
        str(intermediate), str(output), quant, str(QUANTIZE_THREADS),
    ]


def _record_path(out_gguf: Path) -> Path:
    return out_gguf.with_name(out_gguf.name + ".json")


def _require_same_request(record_path: Path, hf_dir: Path, quant: str) -> None:
    """An existing output is only reused for the request that made it; anything else is a mistake."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if Path(record["source"]).resolve() != hf_dir.resolve() or record["quant"] != quant:
        raise FileExistsError(
            f"{record_path.with_suffix('')} was made from {record['source']} at {record['quant']}, "
            f"not from {hf_dir} at {quant}; choose another output name or delete it"
        )


def _run(command: list[str]) -> float:
    """Run one step, return its duration in seconds; on failure raise with the end of its output."""
    print(f"compress: {' '.join(command)}", flush=True)
    started = time.monotonic()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-_LOG_TAIL_LINES:])
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}\n{tail}")
    return time.monotonic() - started


def _output_or_none(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a Hugging Face model folder to a GGUF file.")
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("out_gguf", type=Path)
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()
    print(compress_model(args.hf_dir, args.out_gguf, args.quant))


if __name__ == "__main__":
    main()
