"""Prompt text per section, and the offline rendering used by dry runs."""

from bench.prompts import build_prompt, render_prompt
from bench.records import Question


def make_question(section: str, options: dict[str, str], **extra) -> Question:
    return Question(
        id=Question.make_id("2025-10-19", 3, section, 1),
        exam_date="2025-10-19",
        exam_pass=3,
        section=section,
        number=1,
        question=extra.pop("question", "text"),
        options=options,
        answer="B",
        **extra,
    )


def user_text(question: Question) -> str:
    messages = build_prompt(question)
    assert len(messages) == 1 and messages[0]["role"] == "user", "one user message, no system message"
    return messages[0]["content"]


def test_ord_prompt_has_word_options_and_answer_line():
    question = make_question("ORD", {"A": "snabb", "B": "långsam", "C": "tyst", "D": "stor", "E": "ljus"},
                             question="rask")
    text = user_text(question)

    assert "Ord eller uttryck: rask" in text
    assert "närmast samma betydelse" in text
    for line in ["A. snabb", "B. långsam", "C. tyst", "D. stor", "E. ljus"]:
        assert line in text.splitlines()
    assert "A, B, C, D eller E" in text


def test_las_prompt_has_title_passage_then_question_then_options():
    question = make_question(
        "LAS",
        {"A": "alternativ ett", "B": "alternativ två", "C": "alternativ tre", "D": "alternativ fyra"},
        question="Vad vill författaren säga?",
        passage="Hela texten om något viktigt.",
        passage_title="Om havet",
    )
    text = user_text(question)

    title_at = text.index("Titel: Om havet")
    passage_at = text.index("Hela texten om något viktigt.")
    question_at = text.index("Fråga: Vad vill författaren säga?")
    options_at = text.index("A. alternativ ett")
    assert title_at < passage_at < question_at < options_at
    assert "B. alternativ två" in text and "D. alternativ fyra" in text
    assert "A, B, C eller D" in text


def test_las_prompt_without_title_has_no_title_line():
    question = make_question("LAS", {"A": "a", "B": "b", "C": "c", "D": "d"},
                             question="Fråga?", passage="Texten.", passage_title=None)
    assert "Titel:" not in user_text(question)


def test_mek_prompt_keeps_gaps_and_multi_word_options():
    question = make_question(
        "MEK",
        {"A": "avgränsa – kognitiva – enhetlighet", "B": "begränsa – kognitiva – likhet",
         "C": "avgränsa – konkreta – enhetlighet", "D": "utvidga – kognitiva – mångfald"},
        question="Forskningen _____ området, men _____ resultaten är svårare att _____.",
    )
    text = user_text(question)

    assert "(_____)" in text and "Mening: Forskningen _____ området" in text
    assert "A. avgränsa – kognitiva – enhetlighet" in text
    assert "ordningen från vänster till höger" in text


def test_options_are_listed_in_letter_order_whatever_the_dict_order():
    question = make_question("MEK", {"C": "tredje", "A": "första", "D": "fjärde", "B": "andra"})
    option_lines = [line for line in user_text(question).splitlines() if line[:3] in {"A. ", "B. ", "C. ", "D. "}]
    assert option_lines == ["A. första", "B. andra", "C. tredje", "D. fjärde"]


def test_five_options_are_all_shown_and_counted_in_answer_line():
    question = make_question("ORD", {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}, question="ord")
    text = user_text(question)
    assert "E. e" in text
    assert "A, B, C, D eller E" in text


def test_prompt_is_identical_each_time():
    question = make_question("ORD", {"A": "a", "B": "b", "C": "c", "D": "d"}, question="ord")
    assert build_prompt(question) == build_prompt(question)


def test_plain_rendering_without_tokenizer_contains_everything():
    question = make_question("MEK", {"A": "första", "B": "andra", "C": "tredje", "D": "fjärde"},
                             question="Ett _____ två.")
    rendered = render_prompt(build_prompt(question), tokenizer_path=None)

    assert "A. första" in rendered and "D. fjärde" in rendered
    assert "Mening: Ett _____ två." in rendered
    assert rendered.rstrip().endswith("assistant:")
    assert rendered == render_prompt(build_prompt(question), tokenizer_path=None)
