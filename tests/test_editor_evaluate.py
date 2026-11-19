"""The editor-edit scoring driver (bench.editor.evaluate).

Two kinds of checks:

* the edit itself, on small torch tensors: reading dense and sparse edits in both file layouts,
  the random control (same pattern, same norms), applying a change in memory with an exact
  restore, and writing a merged model folder with Gemma-style stored names;
* the whole flow on dry backends in a temporary project folder, as in test_group_experiment.py:
  a dry "gemma-dry" model with a dry 4-bit entry, the group experiment's split, and a stand-in for
  `applied` whose editor edit makes the dry model answer some questions correctly (the random
  control does nothing). The damage checks are replaced by a stub.
"""

import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from bench import cli, damage, group_experiment as ge, paths, run  # noqa: E402
from bench.editor import evaluate as ev  # noqa: E402
from bench.records import Question  # noqa: E402

DEV_DATES = [f"20{year}-04-0{day}" for year, day in
             ((13, 1), (14, 2), (15, 3), (16, 4), (17, 5), (18, 6), (19, 7), (20, 8), (21, 9))]
TEST_DATE = "2025-10-19"
EXTRA_DATE = "2024-10-20"
PER_SECTION = {"dev": 6, "test": 4, "gemma_extra": 4}
MODELS_TOML = """
[gemma-dry]
backend = "random"
seed = 0

[gemma-dry-q4]
backend = "random"
seed = 0
parent = "gemma-dry"
"""
RUN = "editor10"

DOWN = "model.language_model.layers.{}.mlp.down_proj.weight"
EMBED = "model.language_model.embed_tokens.weight"


# ------------------------------------------------------------------ a small edit

def sample_mapping() -> dict:
    """Dense change of layer 1 (with some exact zeros), rows of layer 2, columns of the embedding."""
    generator = torch.Generator().manual_seed(0)
    dense = torch.randn(6, 8, generator=generator)
    dense[0] = 0.0
    dense[:, 3] = 0.0
    return {
        DOWN.format(1): dense,
        DOWN.format(2): {"rows": torch.tensor([4, 0]), "values": torch.randn(2, 8, generator=generator)},
        EMBED: {"cols": torch.tensor([1, 5, 2]), "values": torch.randn(8, 3, generator=generator)},
    }


def write_suffix_layout(mapping: dict, path: Path) -> None:
    """The hand-written layout: `<name>` for dense, `<name>.rows` / `<name>.values` for sparse."""
    flat = {}
    for name, change in mapping.items():
        if isinstance(change, dict):
            kind = "rows" if "rows" in change else "cols"
            flat[f"{name}.{kind}"] = change[kind]
            flat[f"{name}.values"] = change["values"]
        else:
            flat[name] = change
    save_file(flat, str(path))


def assert_same_edit(edit: ev.EditorEdit, mapping: dict) -> None:
    assert [t.name for t in edit.tensors] == sorted(mapping)
    for part in edit.tensors:
        change = mapping[part.name]
        if isinstance(change, dict):
            kind = "rows" if "rows" in change else "cols"
            assert part.kind == kind
            assert torch.equal(part.index, change[kind])
            assert torch.equal(part.values, change["values"])
        else:
            assert part.kind == "dense"
            assert torch.equal(part.values, change)


def test_reads_dense_and_sparse_edits_in_both_layouts(tmp_path):
    mapping = sample_mapping()
    edit = ev.EditorEdit(ev.tensor_edits(mapping))

    ev.save_edit(edit, tmp_path / "workbench_layout")      # values:/rows:/cols: keys, as the workbench writes
    assert any(key.startswith("values:") for key in load_file(str(tmp_path / "workbench_layout.safetensors")))
    loaded = ev.load_edit(tmp_path / "workbench_layout")
    assert_same_edit(loaded, mapping)
    assert_same_edit(ev.EditorEdit(ev.tensor_edits(ev.read_edit_file(tmp_path / "workbench_layout.safetensors"))),
                     mapping)

    write_suffix_layout(mapping, tmp_path / "suffix_layout.safetensors")
    assert_same_edit(ev.EditorEdit(ev.tensor_edits(ev.read_edit_file(tmp_path / "suffix_layout.safetensors"))),
                     mapping)
    assert ev.split_sparse_key("a.b.weight::rows") == ("a.b.weight", "rows")
    assert ev.split_sparse_key("a.b.weight") == ("a.b.weight", None)


def test_the_workbench_reader_is_used_for_its_own_files(tmp_path):
    workbench = pytest.importorskip("bench.editor.workbench")
    mapping = sample_mapping()
    workbench.save_edit(mapping, tmp_path / "final_edit.safetensors")
    (tmp_path / "final_edit.json").write_text(json.dumps({"explanation": "x", "budget": 10}))
    edit = ev.load_edit(tmp_path / "final_edit")
    assert edit.loader == "bench.editor.workbench.load_edit"
    assert edit.info["budget"] == 10
    assert_same_edit(edit, mapping)


@pytest.mark.parametrize("bad, message", [
    ({"w": {"rows": torch.tensor([1, 1]), "values": torch.zeros(2, 3)}}, "repeated"),
    ({"w": {"rows": torch.tensor([1, 2]), "values": torch.zeros(3, 3)}}, "2 rows"),
    ({"w": {"cols": torch.tensor([1]), "values": torch.zeros(3, 2)}}, "1 columns"),
    ({"w": {"rows": torch.tensor([1]), "cols": torch.tensor([1]), "values": torch.zeros(1, 1)}}, "exactly one"),
    ({}, "no weights"),
])
def test_malformed_edits_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        ev.tensor_edits(bad)


def test_control_has_the_same_pattern_and_norms_but_other_values():
    edit = ev.EditorEdit(ev.tensor_edits(sample_mapping()))
    control = ev.random_control(edit, seed=1)
    assert control.kind == "random"
    for e, c in zip(edit.tensors, control.tensors):
        assert (c.name, c.kind) == (e.name, e.kind)
        if e.index is not None:
            assert torch.equal(c.index, e.index)
        assert torch.equal(c.values != 0, e.values != 0)          # same nonzero pattern
        assert torch.linalg.norm(c.values).item() == pytest.approx(torch.linalg.norm(e.values).item(), rel=1e-5)
        assert not torch.allclose(c.values, e.values)

    again = ev.random_control(edit, seed=1)
    assert all(torch.equal(a.values, b.values) for a, b in zip(control.tensors, again.tensors))
    reordered = ev.random_control(ev.EditorEdit(list(reversed(edit.tensors))), seed=1)
    by_name = {t.name: t.values for t in reordered.tensors}
    assert all(torch.equal(t.values, by_name[t.name]) for t in control.tensors)   # order does not matter
    other = ev.random_control(edit, seed=2)
    assert not torch.equal(other.tensors[0].values, control.tensors[0].values)


# ------------------------------------------------------------------ applying, merging, describing

class TinyBackend:
    """Something with `.model`: a module whose parameter names look like Gemma 3's in memory."""

    def __init__(self):
        generator = torch.Generator().manual_seed(1)
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.language_model = torch.nn.Module()
        lm = model.model.language_model
        lm.embed_tokens = torch.nn.Embedding(8, 10)
        lm.layers = torch.nn.ModuleList()
        for _ in range(3):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.down_proj = torch.nn.Linear(8, 6, bias=False)
            lm.layers.append(layer)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        self.model = model.to(torch.bfloat16)


def expected_weight(original: torch.Tensor, part: ev.TensorEdit) -> torch.Tensor:
    full = original.to(torch.float32).clone()
    if part.kind == "dense":
        full += part.values
    elif part.kind == "rows":
        full[part.index] += part.values
    else:
        full[:, part.index] += part.values
    return full.to(original.dtype)


def test_applied_adds_in_float32_and_restores_exactly():
    backend = TinyBackend()
    edit = ev.EditorEdit(ev.tensor_edits(sample_mapping()))
    before = {name: p.detach().clone() for name, p in backend.model.named_parameters()}
    with ev.applied(backend, edit) as landed:
        for part in edit.tensors:
            now = backend.model.get_parameter(part.name)
            assert torch.equal(now, expected_weight(before[part.name], part))
        assert torch.equal(backend.model.get_parameter(DOWN.format(0)), before[DOWN.format(0)])
        assert [row["name"] for row in landed] == [t.name for t in edit.tensors]
        assert all(0 < row["realized_fraction"] <= 1.01 for row in landed)
    for name, parameter in backend.model.named_parameters():
        assert torch.equal(parameter, before[name]), name

    with pytest.raises(RuntimeError):
        with ev.applied(backend, edit):
            raise RuntimeError("boom")
    for name, parameter in backend.model.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_applied_does_nothing_on_a_dry_backend():
    edit = ev.EditorEdit(ev.tensor_edits(sample_mapping()))
    with ev.applied(object(), edit) as landed:
        assert "nothing was applied" in landed[0]["note"]


def write_model_folder(backend: TinyBackend, folder: Path) -> None:
    """Two shards with Gemma-style stored names (language_model.model.layers...)."""
    folder.mkdir()
    stored = {}
    for name, parameter in backend.model.named_parameters():
        key = name.replace("model.language_model.", "language_model.model.")
        stored[key] = parameter.detach().clone()
    first = {k: v for k, v in stored.items() if "layers.0" in k or "layers.1" in k}
    second = {k: v for k, v in stored.items() if k not in first}
    save_file(first, str(folder / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(second, str(folder / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    (folder / "config.json").write_text("{}")


def test_merged_model_holds_exactly_the_numbers_scored_in_memory(tmp_path):
    backend = TinyBackend()
    source = tmp_path / "source"
    write_model_folder(backend, source)
    # Only layer 0 is edited: the second shard (embedding, layer 2) must stay a symlink.
    edit = ev.EditorEdit(ev.tensor_edits({DOWN.format(0): {"rows": torch.tensor([3]), "values": torch.ones(1, 8)}}))
    out = ev.save_merged_model(source, edit, tmp_path / "merged")
    assert (out / "model-00002-of-00002.safetensors").is_symlink()
    assert not (out / "model-00001-of-00002.safetensors").is_symlink()
    assert (out / "config.json").exists()
    with ev.applied(backend, edit):
        in_memory = backend.model.get_parameter(DOWN.format(0)).detach().clone()
    on_disk = load_file(str(out / "model-00001-of-00002.safetensors"))["language_model.model.layers.0.mlp.down_proj.weight"]
    assert torch.equal(on_disk, in_memory)
    provenance = json.loads((out / "selma_editor_edit.json").read_text())
    assert provenance["rewritten"][0]["stored_as"] == "language_model.model.layers.0.mlp.down_proj.weight"

    # The whole sample edit touches both shards; every stored tensor matches the in-memory model.
    full_edit = ev.EditorEdit(ev.tensor_edits(sample_mapping()))
    out2 = ev.save_merged_model(source, full_edit, tmp_path / "merged2")
    with ev.applied(backend, full_edit):
        memory = {n: p.detach().clone() for n, p in backend.model.named_parameters()}
    disk = {}
    for shard in out2.glob("*.safetensors"):
        disk.update(load_file(str(shard)))
    for name, tensor in memory.items():
        assert torch.equal(disk[name.replace("model.language_model.", "language_model.model.")], tensor), name
    with pytest.raises(FileExistsError):
        ev.save_merged_model(source, edit, out)


def test_stored_names_follow_the_workbench_table_first():
    keys = ["language_model.model.layers.0.mlp.down_proj.weight", "odd.place.for.it"]
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    assert ev.stored_key(keys, name) == keys[0]
    assert ev.stored_key(keys, name, {name: {"key": "odd.place.for.it"}}) == "odd.place.for.it"
    with pytest.raises(ValueError, match="cannot find"):
        ev.stored_key(keys, "model.language_model.norm.weight")


def test_describe_edit_with_the_model_files(tmp_path):
    backend = TinyBackend()
    write_model_folder(backend, tmp_path / "source")
    edit = ev.EditorEdit(ev.tensor_edits(sample_mapping()))
    description = ev.describe_edit(edit, tmp_path / "source")
    total = sum(p.numel() for p in backend.model.parameters())
    rows = {row["name"]: row for row in description["tensors"]}
    dense = rows[DOWN.format(1)]
    assert dense["weights_touched"] == 5 * 7                       # one row and one column are zero
    assert (dense["rows_touched"], dense["cols_touched"]) == (5, 7)
    assert dense["layer"] == 1 and dense["module"] == "mlp.down_proj"
    weight = backend.model.get_parameter(DOWN.format(1)).float()
    assert dense["relative_norm"] == pytest.approx(torch.linalg.norm(sample_mapping()[DOWN.format(1)]).item()
                                                   / torch.linalg.norm(weight).item(), rel=1e-4)
    assert rows[DOWN.format(2)]["rows_indices"] == [0, 4]
    assert rows[EMBED]["cols_touched"] == 3 and rows[EMBED]["layer"] is None
    assert description["model_weights"] == total
    assert description["weights_touched"] == 35 + 16 + 24
    assert description["share_of_model"] == pytest.approx(75 / total)
    assert 0 < description["weights_changed"] <= 75
    assert description["layers"] == [1, 2]

    dry = ev.describe_edit(edit, None)
    assert dry["model_weights"] is None and dry["share_of_model"] is None and dry["weights_changed"] is None


# ------------------------------------------------------------------ the temporary project

def make_questions() -> dict[str, list[Question]]:
    splits: dict[str, list[Question]] = {"dev": [], "test": [], "gemma_extra": []}
    for split, dates in (("dev", DEV_DATES), ("test", [TEST_DATE]), ("gemma_extra", [EXTRA_DATE])):
        for date in dates:
            for section in ("ORD", "LAS", "MEK"):
                letters = ["A", "B", "C", "D", "E"] if section == "ORD" else ["A", "B", "C", "D"]
                for number in range(1, PER_SECTION[split] + 1):
                    splits[split].append(Question(
                        id=Question.make_id(date, 1, section, number), exam_date=date, exam_pass=1,
                        section=section, number=number, question=f"fråga {number}",
                        options={letter: f"alternativ {letter}" for letter in letters},
                        answer=letters[number % len(letters)],
                        passage="En text." if section == "LAS" else None,
                        passage_title="Titel" if section == "LAS" else None,
                    ))
    return splits


def fake_damage_checks(backend, run_dir, data_dir=None):
    result = {"perplexity": backend.perplexity(["x"]), "n_perplexity_docs": 1,
              "loop_rate_percent": 0.0, "n_prompts": 1, "loop_rule": "test stub"}
    (run_dir / "damage.json").write_text(json.dumps(result), encoding="utf-8")
    return result


@pytest.fixture
def project(tmp_path, monkeypatch):
    splits = make_questions()
    exams = tmp_path / "exams"
    exams.mkdir()
    questions_file = exams / "questions.jsonl"
    questions_file.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n"
                                      for questions in splits.values() for q in questions), encoding="utf-8")
    split_file = exams / "split.json"
    split_file.write_text(json.dumps({
        "splits": {name: {"question_ids": [q.id for q in questions]} for name, questions in splits.items()},
        "questions_sha256": None,
    }), encoding="utf-8")
    models_file = tmp_path / "models.toml"
    models_file.write_text(MODELS_TOML, encoding="utf-8")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "QUESTIONS", questions_file)
    monkeypatch.setattr(paths, "SPLIT", split_file)
    monkeypatch.setattr(paths, "MODELS_CONFIG", models_file)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(damage, "run_damage_checks", fake_damage_checks)
    return tmp_path


@pytest.fixture
def phase1(project):
    """Phase 1 runs and the group experiment's split (what the editor experiment starts from)."""
    runs = {"dev": run.run_model("gemma-dry", "dev"),
            "q4_test": run.run_model("gemma-dry-q4", "test"),
            "q4_gemma_extra": run.run_model("gemma-dry-q4", "gemma_extra")}
    ge.make_split("gemma-dry")
    return runs


@pytest.fixture
def group_baseline(phase1):
    """The unedited full-precision runs of the group experiment (no 4-bit hidden run)."""
    ge.evaluate("gemma-dry", conditions=(ge.BASELINE,))
    return ge.condition_dir("gemma-dry", ge.BASELINE)


def write_editor_run(project: Path, name: str = RUN, budget: int = 10, mapping: dict | None = None) -> Path:
    folder = project / "editor_runs" / name
    folder.mkdir(parents=True)
    ev.save_edit(ev.EditorEdit(ev.tensor_edits(mapping or sample_mapping())), folder / "final_edit")
    (folder / "final_edit.json").write_text(json.dumps({
        "run_name": name, "model": "gemma-dry", "budget": budget, "passes_used": budget - 1,
        "explanation": "Strengthened the reading neurons.\nSecond line.",
        "size": {"weights_touched": 81, "weights_changed": 80, "model_weights": 1000, "share_of_model": 0.081,
                 "tensors": {"x": {"shape": [1]}}},
    }), encoding="utf-8")
    return folder


def fixes(question_id: str, generalise: bool) -> bool:
    return generalise and hashlib.sha256(question_id.encode()).hexdigest()[0] in "0123456789"


def fake_applied(generalise: bool):
    """The editor's edit makes the dry model answer some questions right; the random control does nothing."""

    @contextmanager
    def applied(backend, edit):
        if edit is None or edit.kind != "editor":
            yield [] if edit is None else [{"name": t.name, "realized_fraction": 1.0} for t in edit.tensors]
            return
        original = backend.letter_probabilities

        def patched(prompt, letters, question):
            if fixes(question.id, generalise):
                return {letter: 0.9 if letter == question.answer else 0.01 for letter in letters}
            return original(prompt, letters, question)

        backend.letter_probabilities = patched
        try:
            yield [{"name": t.name, "realized_fraction": 0.97, "entries_changed": 3} for t in edit.tensors]
        finally:
            del backend.letter_probabilities

    return applied


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the whole flow

def test_whole_flow_reuses_the_group_baseline(project, group_baseline, phase1, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=True))
    write_editor_run(project)
    ev.evaluate_run("gemma-dry", RUN)
    folder = project / "experiments" / "editor" / RUN

    for condition in ev.CONDITIONS:
        for name in ("hidden", "drill", "validation", "test", "gemma_extra"):
            assert ge.run_is_complete(folder / condition / name), (condition, name)
    assert (folder / "edited" / "test" / "damage.json").exists()
    for name in ("q4_hidden", "q4_test", "q4_gemma_extra"):
        assert ge.run_is_complete(folder / "edited" / name)
        assert not (folder / "control" / name).exists()          # no 4-bit control without the flag
    own = project / "experiments" / "editor" / "_baseline" / "gemma-dry"
    assert sorted(p.name for p in own.iterdir()) == ["q4_hidden"]   # everything else was reused
    assert not any(path.is_file() for path in (project / "work").rglob("*"))
    info = read_json(folder / "edited" / "hidden" / "run.json")
    assert info["model"] == f"gemma-dry+editor[{RUN}]"
    assert info["experiment"]["edit"]["kind"] == "editor"
    assert read_json(folder / "control" / "hidden" / "run.json")["experiment"]["edit"]["kind"] == "random"
    assert (folder / "control" / "control_edit.safetensors").exists()

    results = read_json(folder / "results.json")
    assert results["editor"]["budget"] == 10 and results["editor"]["passes_used"] == 9
    assert results["editor"]["size"] == {"weights_touched": 81, "weights_changed": 80, "model_weights": 1000,
                                         "share_of_model": 0.081}
    assert results["edit"]["weights_touched"] == 75
    full = results["full"]
    assert full["baselines"]["hidden"] == "experiments/group/gemma-dry/baseline/hidden"
    assert full["hidden_check"]["disagreements"] == 0
    hidden = full["hidden"]
    assert hidden["edited"]["broken"] == 0 and hidden["edited"]["fixed"] > 0
    assert hidden["control"]["fixed"] == 0 and hidden["control"]["broken"] == 0
    assert hidden["edited_vs_control"]["net"]["ci"][0] > 0
    assert set(hidden["edited_by_section"]) == {"ORD", "LAS", "MEK"}
    assert full["practice"]["edited"]["n"] == len(ge.questions_of(read_json(ge.split_path("gemma-dry")),
                                                                  ["drill_wrong", "drill_right",
                                                                   "validation_wrong", "validation_right"]))
    assert full["test"]["edited_vs_baseline"]["control"] == "test"
    assert full["test"]["edited_vs_baseline"]["damage"] is not None
    assert full["gemma_extra"]["edited_vs_control"] is not None
    assert results["q4"]["baselines"]["test"] == f"runs/{phase1['q4_test'].name}"
    assert results["q4"]["baselines"]["hidden"] == "experiments/editor/_baseline/gemma-dry/q4_hidden"
    # The dry 4-bit model does not carry the edit, so the gain does not survive compression.
    assert results["survival"]["status"] == "lost"
    assert results["verdict"]["label"] == "improvement"

    text = results["verdict"]["paragraph"]
    assert "a real net gain" in text and "does better than a random edit" in text
    assert "does not survive compression" in text
    markdown = (folder / "results.md").read_text()
    assert "> Strengthened the reading neurons." in markdown and "> Second line." in markdown
    assert "**Overall label: improvement**" in markdown
    assert "On the hidden papers (54 questions the editor never saw)" in text
    assert "| `mlp.down_proj` | 2 | rows | 2 rows: 0, 4 |" in markdown
    assert "97.0% / 100.0%" in markdown          # how much landed, edit / random


def test_an_edit_that_does_nothing_is_never_called_an_improvement(project, group_baseline, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=False))
    write_editor_run(project)
    ev.evaluate_run("gemma-dry", RUN, with_control_q4=True)
    folder = project / "experiments" / "editor" / RUN
    results = read_json(folder / "results.json")
    assert results["full"]["hidden"]["edited"]["net"]["accuracy_change"] == 0
    assert results["verdict"]["label"] == "no evidence"
    text = results["verdict"]["paragraph"]
    assert "includes zero" in text
    assert "real net gain" not in text and "does better than a random edit" not in text
    assert "an improvement on unseen papers" not in text
    assert "NO IMPROVEMENT" in text
    assert results["survival"]["status"] == "no_full_precision_gain"
    assert ge.run_is_complete(folder / "control" / "q4_hidden")


def test_missing_unedited_runs_are_scored_once_into_the_editor_folder(project, phase1, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=True))
    write_editor_run(project)
    ev.evaluate_run("gemma-dry", RUN)
    own = project / "experiments" / "editor" / "_baseline" / "gemma-dry"
    assert sorted(p.name for p in own.iterdir()) == ["drill", "gemma_extra", "hidden", "q4_hidden", "test",
                                                     "validation"]
    assert not ge.condition_dir("gemma-dry", ge.BASELINE).exists()   # the group folder is not touched

    # A second editor run reuses those unedited runs.
    write_editor_run(project, "editor25", budget=25)
    calls = []
    real = ev.run_with_backend

    def counting(backend, label, *args, **kwargs):
        calls.append(label)
        return real(backend, label, *args, **kwargs)

    monkeypatch.setattr(ev, "run_with_backend", counting)
    ev.evaluate_run("gemma-dry", "editor25")
    assert not any(label in ("gemma-dry", "gemma-dry-q4") for label in calls)


def test_second_run_does_no_work(project, group_baseline, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=True))
    write_editor_run(project)
    ev.evaluate_run("gemma-dry", RUN, with_control_q4=True)
    calls = []

    def no_backend(name):
        raise AssertionError(f"no model should be loaded when every step is done (asked for {name})")

    monkeypatch.setattr(ev, "run_with_backend", lambda *a, **k: calls.append(k.get("run_dir")))
    monkeypatch.setattr(ev, "make_backend", no_backend)
    monkeypatch.setattr(ge, "make_backend", no_backend)
    ev.evaluate_run("gemma-dry", RUN, with_control_q4=True)
    assert calls == []


def test_an_unfinished_run_folder_is_redone(project, group_baseline, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=True))
    write_editor_run(project)
    ev.evaluate_run("gemma-dry", RUN)
    hidden = project / "experiments" / "editor" / RUN / "edited" / "hidden"
    info = read_json(hidden / "run.json")
    del info["finished_at"]
    (hidden / "run.json").write_text(json.dumps(info))
    redone = []
    real = ev.run_with_backend

    def counting(*args, **kwargs):
        redone.append(kwargs["run_dir"].name)
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "run_with_backend", counting)
    ev.evaluate_run("gemma-dry", RUN)
    assert redone == ["hidden"]
    assert ge.run_is_complete(hidden)


def test_a_changed_edit_file_stops_the_driver(project, group_baseline, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=True))
    write_editor_run(project)
    ev.describe("gemma-dry", RUN)
    other = sample_mapping()
    other[DOWN.format(1)] = other[DOWN.format(1)] * 2
    ev.save_edit(ev.EditorEdit(ev.tensor_edits(other)), project / "editor_runs" / RUN / "final_edit")
    with pytest.raises(ValueError, match="changed since"):
        ev.evaluate_run("gemma-dry", RUN)


def test_the_hidden_papers_come_from_the_group_split(project, monkeypatch):
    run.run_model("gemma-dry", "dev")
    write_editor_run(project)
    with pytest.raises(FileNotFoundError, match="bench group split --model gemma-dry"):
        ev.evaluate_run("gemma-dry", RUN)


def test_an_edit_of_another_model_is_refused(project, phase1):
    write_editor_run(project)
    record = project / "editor_runs" / RUN / "final_edit.json"
    record.write_text(json.dumps({**read_json(record), "model": "qwen-9b-full"}))
    with pytest.raises(ValueError, match="use --model qwen-9b-full"):
        ev.describe("gemma-dry", RUN)


# ------------------------------------------------------------------ summary and CLI

def write_group_results(project: Path) -> None:
    hidden = {"fixed": 10, "n_wrong": 117, "broken": 4, "n_right": 363, "net_questions": 6,
              "net": {"accuracy_change": 0.0125, "ci": [-0.002, 0.027]}}
    comparison = {"verdict": "NO IMPROVEMENT", "headline": {"metric_change": 0.01},
                  "overall": {"accuracy_change": 0.01, "accuracy_ci": [-0.02, 0.04]},
                  "damage": {"perplexity": {"relative_change": 0.01}, "loop_rate_percent": {"change_points": 0.0}}}
    results = {"model": "gemma-dry",
               "full": {"trained": {"hidden": hidden, "test_vs_baseline": comparison},
                        "trained_vs_control": {"hidden_net": {"change": 0.01, "ci": [-0.01, 0.03]}}},
               "q4": None}
    path = ge.model_dir("gemma-dry") / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results))


def test_cli_needs_a_run(project, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "editor-eval"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_runs_one_editor_run(project, group_baseline, monkeypatch):
    monkeypatch.setattr(ev, "applied", fake_applied(generalise=False))
    write_editor_run(project)
    monkeypatch.setattr(sys, "argv", ["bench", "editor-eval", "--model", "gemma-dry", "--run", RUN])
    cli.main()
    assert (project / "experiments" / "editor" / RUN / "results.md").exists()


# ------------------------------------------------------------------ the verdict wording

def flip_numbers(fixed: int, broken: int, net_ci: list[float]) -> dict:
    return {"n": 400, "accuracy_before": 0.75, "accuracy_after": 0.75 + (fixed - broken) / 400,
            "n_wrong": 100, "fixed": fixed, "n_right": 300, "broken": broken, "net_questions": fixed - broken,
            "net": {"accuracy_change": (fixed - broken) / 400, "ci": net_ci}, "fixed_ids": [], "broken_ids": []}


def synthetic_results(net_ci, versus_ci, damage_passed=True) -> dict:
    rules = {"1_metric_improves": {"passed": False}, "2_beats_control": {"passed": False},
             "3a_perplexity": {"passed": damage_passed}, "3b_loop_rate": {"passed": True}}
    test = {"verdict": "NO IMPROVEMENT", "rules": rules,
            "headline": {"metric_change": 0.001, "metric_ci": [-0.01, 0.012]},
            "damage": {"perplexity": {"relative_change": 0.01 if damage_passed else 0.2},
                       "loop_rate_percent": {"change_points": 0.0}}}
    return {"edit": {"weights_touched": 1234, "n_tensors": 2, "share_of_model": 0.0001, "layers": [20, 21]},
            "editor": {"budget": 25, "passes_used": 25},
            "full": {"hidden": {"edited": flip_numbers(12, 5, net_ci), "control": flip_numbers(3, 3, [-0.01, 0.01]),
                                "edited_vs_control": {"net": {"change": sum(versus_ci) / 2, "ci": versus_ci}}},
                     "test": {"edited_vs_baseline": test}},
            "q4": None, "survival": None}


@pytest.mark.parametrize("net_ci, versus_ci, damage_passed, label, expected, forbidden", [
    ([-0.01, 0.04], [-0.01, 0.05], True, "no evidence",
     ["cannot be told apart from no change", "no better than a random edit"], ["real net gain", "an improvement"]),
    ([0.005, 0.04], [-0.01, 0.05], True, "no evidence",
     ["a real net gain", "no better than a random edit"], ["an improvement on unseen papers"]),
    ([0.005, 0.04], [0.002, 0.05], True, "improvement",
     ["a real net gain", "does better than a random edit", "an improvement on unseen papers"], ["includes zero"]),
    ([0.005, 0.04], [0.002, 0.05], False, "harm",
     ["outside the limits", "damages the model"], ["an improvement on unseen papers"]),
    ([-0.04, -0.005], [-0.05, -0.002], True, "harm",
     ["a real net loss", "worse than a random edit"], ["real net gain", "an improvement on unseen papers"]),
])
def test_verdict_follows_the_confidence_intervals(net_ci, versus_ci, damage_passed, label, expected, forbidden):
    result = ev.verdict(synthetic_results(net_ci, versus_ci, damage_passed))
    assert result["label"] == label
    for words in expected:
        assert words in result["paragraph"]
    for words in forbidden:
        assert words not in result["paragraph"]
    assert "budget of 25 forward passes (used 25)" in result["paragraph"]
