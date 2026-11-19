"""The parts of the GPU backends that can be checked without a GPU, torch or llama.cpp."""

import socket
from pathlib import Path

from bench.backends import SIDECAR_NAME, cached_sha256, folder_checksums


def test_gpu_backend_modules_import_without_gpu_libraries():
    import bench.backends.hf  # noqa: F401
    import bench.backends.llamacpp  # noqa: F401
    import bench.compress  # noqa: F401


def test_free_port_can_be_bound():
    from bench.backends.llamacpp import _free_port

    port = _free_port()
    with socket.socket() as server:
        server.bind(("127.0.0.1", port))


def test_first_token_candidates_reads_top_logprobs():
    from bench.backends.llamacpp import _first_token_candidates

    reply = {"completion_probabilities": [{"top_logprobs": [{"token": " A", "logprob": -0.1}]}]}
    assert _first_token_candidates(reply) == [{"token": " A", "logprob": -0.1}]


def test_symlinked_shard_shares_the_checksum_cache_of_its_target(tmp_path: Path):
    original = tmp_path / "original"
    edited = tmp_path / "edited"
    original.mkdir()
    edited.mkdir()
    (original / "shard.safetensors").write_bytes(b"unchanged weights")
    (edited / "shard.safetensors").symlink_to(original / "shard.safetensors")

    assert cached_sha256(edited / "shard.safetensors") == cached_sha256(original / "shard.safetensors")
    assert (original / SIDECAR_NAME).exists()
    assert not (edited / SIDECAR_NAME).exists(), "the digest is cached next to the real file only"
    assert set(folder_checksums(edited)) == {"shard.safetensors"}


def test_hf_model_class_falls_back_when_the_architecture_is_unknown(tmp_path: Path):
    import pytest

    transformers = pytest.importorskip("transformers")
    from bench.backends.hf import _model_class

    (tmp_path / "config.json").write_text('{"architectures": ["NoSuchModelClass"]}', encoding="utf-8")
    assert _model_class(tmp_path) is transformers.AutoModelForCausalLM
