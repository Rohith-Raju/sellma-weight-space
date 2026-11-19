"""Backend factory and the checksum cache the GPU backends share.

make_backend(name) reads config/models.toml and returns the backend for that entry.
The GPU backends are imported only when asked for, so dry runs never touch torch,
transformers or the llama.cpp binaries.

Checksums: hashing 19 GB takes minutes, and every run would repeat it. cached_sha256 keeps
the digest in a hidden sidecar file next to the model and reuses it while the file's size
and modification time are unchanged. Any edit or re-download changes one of those, so the
digest is recomputed.

An edited model folder holds symlinks to the shards it did not change. The cache follows the
link and keeps the digest next to the real file, so an unchanged 5 GB shard is hashed once,
not once per edited model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bench.backends.base import Backend
from bench.backends.dry import BrokenBackend, OracleBackend, RandomBackend
from bench.models import ModelSpec, get_model

SIDECAR_NAME = ".bench_sha256.json"
DEFAULT_HF_DTYPE = "bfloat16"   # the precision the models are published in; `dtype` in models.toml overrides it
_CHUNK_BYTES = 8 * 1024 * 1024


def make_backend(name: str) -> Backend:
    spec = get_model(name)
    if spec.backend == "random":
        return RandomBackend(seed=int(spec.setting("seed", 0)))
    if spec.backend == "oracle":
        return OracleBackend()
    if spec.backend == "broken":
        return BrokenBackend()
    if spec.backend == "hf":
        from bench.backends.hf import HFBackend

        return HFBackend(model_path=_required_path(spec), tokenizer_path=spec.tokenizer or spec.path,
                         dtype=str(spec.setting("dtype", DEFAULT_HF_DTYPE)))
    if spec.backend == "llamacpp":
        from bench.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(gguf_path=_required_path(spec), llama_bin_dir=_llama_bin_dir(spec),
                               tokenizer_path=spec.tokenizer)
    raise ValueError(f"model {name!r} has unknown backend {spec.backend!r}")


def _required_path(spec: ModelSpec) -> Path:
    if spec.path is None:
        raise ValueError(f"model {spec.name!r} needs a path in config/models.toml")
    return spec.path


def _llama_bin_dir(spec: ModelSpec) -> Path | None:
    # Optional override; the backend falls back to its default build location when None.
    value = spec.setting("llama_cpp_bin")
    return Path(value).expanduser() if value else None


def cached_sha256(file: Path) -> str:
    """sha256 of one file, reusing the sidecar value while size and mtime are unchanged."""
    file = file.resolve()   # a symlinked shard shares the cache entry of the file it points to
    sidecar = file.parent / SIDECAR_NAME
    cache = _read_sidecar(sidecar)
    stat = file.stat()
    entry = cache.get(file.name)
    if entry and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
        return entry["sha256"]

    digest = _hash_file(file)
    cache[file.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    _write_sidecar(sidecar, cache)
    return digest


def folder_checksums(folder: Path) -> dict[str, str]:
    """sha256 of every regular file in a folder, skipping hidden files such as the sidecar."""
    return {
        file.name: cached_sha256(file)
        for file in sorted(folder.iterdir())
        if file.is_file() and not file.name.startswith(".")
    }


def _hash_file(file: Path) -> str:
    digest = hashlib.sha256()
    with open(file, "rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sidecar(sidecar: Path) -> dict:
    # A missing or damaged sidecar only costs a rehash, so it is never an error.
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_sidecar(sidecar: Path, cache: dict) -> None:
    try:
        sidecar.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # a read-only model folder just means every run rehashes
