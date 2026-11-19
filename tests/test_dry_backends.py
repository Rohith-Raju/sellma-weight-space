"""Dry-run backends, the backend factory and the models.toml reader. No model, no GPU."""

import pytest

from bench import models, paths
from bench.backends import make_backend
from bench.backends.dry import OracleBackend, RandomBackend
from bench.records import Question

LETTERS_4 = ["A", "B", "C", "D"]


def make_question(number: int, answer: str = "C", n_options: int = 4) -> Question:
    letters = ["A", "B", "C", "D", "E"][:n_options]
    return Question(
        id=Question.make_id("2025-10-19", 1, "ORD", number),
        exam_date="2025-10-19",
        exam_pass=1,
        section="ORD",
        number=number,
        question=f"ord{number}",
        options={letter: f"alt{letter}" for letter in letters},
        answer=answer,
    )


def test_random_backend_is_repeatable_for_same_seed_and_question():
    question = make_question(1)
    first = RandomBackend(seed=0).letter_probabilities("prompt", LETTERS_4, question)
    second = RandomBackend(seed=0).letter_probabilities("prompt", LETTERS_4, question)
    assert first == second


def test_random_backend_depends_on_seed_and_question():
    question = make_question(1)
    other_question = make_question(2)
    assert RandomBackend(0).letter_probabilities("p", LETTERS_4, question) != \
        RandomBackend(1).letter_probabilities("p", LETTERS_4, question)
    assert RandomBackend(0).letter_probabilities("p", LETTERS_4, question) != \
        RandomBackend(0).letter_probabilities("p", LETTERS_4, other_question)


def test_random_backend_puts_about_point_nine_on_the_letters():
    probs = RandomBackend(0).letter_probabilities("p", LETTERS_4, make_question(1))
    assert set(probs) == set(LETTERS_4)
    assert sum(probs.values()) == pytest.approx(0.9)
    assert all(p >= 0 for p in probs.values())


def test_random_backend_is_near_chance_over_many_questions():
    backend = RandomBackend(seed=0)
    questions = [make_question(n, answer=LETTERS_4[n % 4]) for n in range(1, 801)]
    correct = 0
    for question in questions:
        probs = backend.letter_probabilities("p", LETTERS_4, question)
        chosen = max(probs, key=probs.get)
        correct += chosen == question.answer
    accuracy = correct / len(questions)
    assert 0.15 < accuracy < 0.35   # chance with 4 options is 25%


def test_oracle_backend_always_picks_the_answer_key():
    backend = OracleBackend()
    for answer in LETTERS_4:
        question = make_question(1, answer=answer)
        probs = backend.letter_probabilities("p", LETTERS_4, question)
        assert max(probs, key=probs.get) == answer
        assert probs[answer] == pytest.approx(0.97)
        assert sum(probs.values()) == pytest.approx(1.0)


def test_oracle_handles_five_options():
    question = make_question(1, answer="E", n_options=5)
    probs = OracleBackend().letter_probabilities("p", list("ABCDE"), question)
    assert probs["E"] == pytest.approx(0.97)
    assert len(probs) == 5


def test_dry_generate_is_deterministic_and_does_not_loop():
    backend = RandomBackend(seed=0)
    prompt = "Välj det alternativ som har närmast samma betydelse som ordet snabb snabb snabb"
    text = backend.generate(prompt, max_new_tokens=50)
    assert text == backend.generate(prompt, max_new_tokens=50)
    words = text.split()
    assert words, "some text is produced"
    assert len(words) == len(set(words)), "no word is repeated"
    assert set(words) <= set(prompt.split())


def test_dry_generate_respects_max_new_tokens():
    text = OracleBackend().generate("ett två tre fyra fem sex sju", max_new_tokens=3)
    assert len(text.split()) == 3


def test_dry_describe_names_the_backend():
    assert RandomBackend(seed=7).describe()["seed"] == 7
    assert OracleBackend().describe()["backend"] == "oracle"


def test_dry_perplexity_is_a_positive_number():
    assert RandomBackend(seed=0).perplexity(["en text"]) > 0
    assert OracleBackend().perplexity(["en text"]) > 0


def test_models_toml_is_read_with_relative_paths_resolved(tmp_path, monkeypatch):
    config = tmp_path / "models.toml"
    config.write_text(
        '[dry-random]\nbackend = "random"\nseed = 3\n\n'
        '[full]\nbackend = "hf"\npath = "models/x"\ntokenizer = "models/x"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "MODELS_CONFIG", config)
    monkeypatch.setattr(paths, "ROOT", tmp_path / "project")

    specs = models.load_model_specs()
    assert specs["dry-random"].path is None
    assert specs["dry-random"].setting("seed") == 3
    assert specs["full"].path == tmp_path / "project" / "models" / "x"
    assert specs["full"].tokenizer == tmp_path / "project" / "models" / "x"


def test_unknown_model_name_lists_the_known_ones(tmp_path, monkeypatch):
    config = tmp_path / "models.toml"
    config.write_text('[dry-oracle]\nbackend = "oracle"\n', encoding="utf-8")
    monkeypatch.setattr(paths, "MODELS_CONFIG", config)

    with pytest.raises(KeyError, match="dry-oracle"):
        models.get_model("nope")


def test_factory_builds_dry_backends_from_config(tmp_path, monkeypatch):
    config = tmp_path / "models.toml"
    config.write_text('[dry-random]\nbackend = "random"\nseed = 5\n\n[dry-oracle]\nbackend = "oracle"\n',
                      encoding="utf-8")
    monkeypatch.setattr(paths, "MODELS_CONFIG", config)

    assert isinstance(make_backend("dry-oracle"), OracleBackend)
    random_backend = make_backend("dry-random")
    assert isinstance(random_backend, RandomBackend)
    assert random_backend.seed == 5


def test_cached_sha256_reuses_and_refreshes_on_change(tmp_path, monkeypatch):
    from bench.backends import SIDECAR_NAME, cached_sha256, folder_checksums

    model_file = tmp_path / "weights.bin"
    model_file.write_bytes(b"first")
    first = cached_sha256(model_file)
    assert (tmp_path / SIDECAR_NAME).exists()

    # Same size and mtime: the stored value is returned without hashing again.
    import bench.backends as backends
    monkeypatch.setattr(backends, "_hash_file", lambda file: pytest.fail("should not rehash"))
    assert cached_sha256(model_file) == first
    monkeypatch.undo()

    # A change in content (here the size also changes) forces a fresh digest.
    model_file.write_bytes(b"second, longer")
    assert cached_sha256(model_file) != first
    assert set(folder_checksums(tmp_path)) == {"weights.bin"}, "hidden sidecar is not hashed"
