"""bench.compress without llama.cpp: the two commands are replaced by stand-ins that write small files."""

import json
from pathlib import Path

import pytest

from bench import compress


@pytest.fixture
def fake_tools(monkeypatch):
    """Replace the converter and the quantizer; record the commands that would have run."""
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> float:
        commands.append(command)
        if "--outfile" in command:                       # the converter
            source = Path(command[2])
            weights = b"".join(shard.read_bytes() for shard in sorted(source.glob("*.safetensors")))
            Path(command[command.index("--outfile") + 1]).write_bytes(weights)
        else:                                            # llama-quantize: input, output, type, threads
            Path(command[2]).write_bytes(Path(command[1]).read_bytes() + command[3].encode())
        return 0.1

    monkeypatch.setattr(compress, "_run", fake_run)
    monkeypatch.setattr(compress, "llama_cpp_version", lambda: {"tag": "test"})
    monkeypatch.setattr(compress, "converter_package_versions", lambda: {"gguf": "test"})
    return commands


def make_model_dir(folder: Path, weights: bytes = b"weights") -> Path:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(weights)
    return folder


def test_compress_writes_output_and_record_and_removes_intermediate(tmp_path, fake_tools):
    hf_dir = make_model_dir(tmp_path / "model")
    out = tmp_path / "compressed" / "model-Q4_K_M.gguf"

    assert compress.compress_model(hf_dir, out) == out

    assert out.read_bytes() == b"weightsQ4_K_M"
    visible_and_hidden = {path.name for path in out.parent.iterdir()}
    assert visible_and_hidden == {out.name, out.name + ".json", ".bench_sha256.json"}, "no temporary file is left"
    record = json.loads((out.parent / (out.name + ".json")).read_text(encoding="utf-8"))
    assert record["quant"] == "Q4_K_M"
    assert record["intermediate_type"] == compress.INTERMEDIATE_TYPE
    assert set(record["source_files"]) == {"config.json", "model.safetensors"}
    assert len(record["output_sha256"]) == 64
    convert, quantize = fake_tools
    assert convert[-2:] == ["--outtype", compress.INTERMEDIATE_TYPE]
    assert quantize[-2:] == ["Q4_K_M", str(compress.QUANTIZE_THREADS)]


def test_compress_follows_symlinked_shards(tmp_path, fake_tools):
    original = make_model_dir(tmp_path / "original")
    edited = tmp_path / "edited"
    edited.mkdir()
    (edited / "config.json").write_text("{}", encoding="utf-8")
    (edited / "model.safetensors").symlink_to(original / "model.safetensors")

    out = compress.compress_model(edited, tmp_path / "edited.gguf")

    assert out.read_bytes() == b"weightsQ4_K_M"


def test_finished_output_is_kept_and_not_recomputed(tmp_path, fake_tools):
    hf_dir = make_model_dir(tmp_path / "model")
    out = tmp_path / "model.gguf"
    compress.compress_model(hf_dir, out)
    fake_tools.clear()

    assert compress.compress_model(hf_dir, out) == out
    assert fake_tools == [], "a finished output is reused"


def test_existing_output_from_another_request_is_an_error(tmp_path, fake_tools):
    out = tmp_path / "model.gguf"
    compress.compress_model(make_model_dir(tmp_path / "model"), out)

    with pytest.raises(FileExistsError):
        compress.compress_model(make_model_dir(tmp_path / "other"), out)
    with pytest.raises(FileExistsError):
        compress.compress_model(tmp_path / "model", out, quant="Q8_0")


def test_failed_step_leaves_no_files(tmp_path, monkeypatch):
    def failing_run(command: list[str]) -> float:
        Path(command[command.index("--outfile") + 1]).write_bytes(b"partial")
        raise RuntimeError("converter failed")

    monkeypatch.setattr(compress, "_run", failing_run)
    out = tmp_path / "out" / "model.gguf"

    with pytest.raises(RuntimeError):
        compress.compress_model(make_model_dir(tmp_path / "model"), out)
    assert list(out.parent.iterdir()) == []


def test_not_a_model_folder(tmp_path, fake_tools):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        compress.compress_model(tmp_path / "empty", tmp_path / "model.gguf")
