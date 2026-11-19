"""Tests for the booklet parser, the answer-key parser, hyphen joining and validation.

The parser tests build text rows by hand, so they need no PDF. The answer-key tests draw a small
key PDF with PyMuPDF. The tests that read the real 2025-10-19 files are skipped when those files
have not been downloaded.
"""

from __future__ import annotations

import hashlib

from pathlib import Path

import pymupdf
import pytest

from bench.extract import (
    KeyFile,
    PassageGroup,
    TextRow,
    BookletParser,
    join_wrapped_lines,
    read_answer_key,
    read_booklet,
    to_question,
    validate_pass,
    _fill_span_texts,
    _group_lines_into_rows,
    _verbal_answers,
)
from bench.fetch import KIND_ANSWER_KEY, ManifestEntry
from bench.paths import RAW
from bench.records import Question


def _fp(text: str) -> str:
    """Fingerprint of a text: the first 16 hex digits of its sha256."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _contains_fp(text: str, length: int, fingerprint: str) -> bool:
    """True if some part of text with this length has this fingerprint."""
    return any(_fp(text[i:i + length]) == fingerprint for i in range(len(text) - length + 1))

HP_2025_KEY = RAW / "2025-10-19" / "hp-25b.pdf"
HP_2025_PASS3 = RAW / "2025-10-19" / "provpass-3-verb-utan-elf.pdf"

# Answers to 2025-10-19, verbal provpass 3, questions 1-30, from the official key.
HP_2025_PASS3_ANSWERS = "BDEADECAEBBCAABBDACABDBCAABACD"


# ---------------------------------------------------------------------------
# Hyphen joining
# ---------------------------------------------------------------------------

def test_hyphenated_word_is_joined_without_the_hyphen():
    assert join_wrapped_lines(["gam-", "malt bröd"]) == "gammalt bröd"


def test_soft_hyphen_at_line_end_is_dropped():
    assert join_wrapped_lines(["forn­", "tidens flora"]) == "forntidens flora"


def test_hyphen_before_och_or_eller_is_kept():
    # "djur- och växtdelar": the word is completed later, so this is not a broken word.
    assert join_wrapped_lines(["gamla djur- ", "och växtdelar."]) == "gamla djur- och växtdelar."
    assert join_wrapped_lines(["blod-", "eller urinprov"]) == "blod- eller urinprov"


def test_hyphenated_word_with_a_space_after_the_hyphen_is_joined():
    # Many PDF lines end "arbets- " with a space although the word is only broken.
    assert join_wrapped_lines(["byta arbets- ", "givare."]) == "byta arbetsgivare."


def test_hyphen_after_a_digit_or_a_capital_is_kept():
    assert join_wrapped_lines(["från 1200- ", "talet"]) == "från 1200-talet"
    assert join_wrapped_lines(["svenska EU-", "vännerna"]) == "svenska EU-vännerna"


def test_soft_hyphen_at_line_end_after_a_digit_is_a_real_hyphen():
    assert join_wrapped_lines(["under 1950­ ", "talet"]) == "under 1950-talet"


def test_soft_hyphen_at_line_end_with_a_space_after_it_still_joins_the_word():
    assert join_wrapped_lines(["en större för­ ", "ståelse."]) == "en större förståelse."


def test_hyphen_before_a_capital_is_a_compound_and_is_kept():
    assert join_wrapped_lines(["Bengt-", "Samuelsson"]) == "Bengt-Samuelsson"


def test_plain_lines_are_joined_with_a_space():
    assert join_wrapped_lines(["första raden", "andra raden"]) == "första raden andra raden"


# ---------------------------------------------------------------------------
# Reading PDF lines into rows
# ---------------------------------------------------------------------------

def span(text: str, *, x0: float, top: float, baseline: float, font: str = "Cambria") -> dict:
    return {"text": text, "bbox": (x0, top, x0 + 5 * len(text), baseline + 3), "origin": (x0, baseline),
            "font": font, "size": 10.0}


def test_rows_are_ordered_by_baseline_not_by_box_top():
    # 2026-10-18 layout: the box of regular Cambria starts 31 pt above its baseline, that of
    # Cambria Bold 10 pt. By box top the subtitle would land inside the first line of its text.
    lines = [
        [span("DIKTEN", x0=43, top=310, baseline=320, font="Cambria-Bold")],
        [span("Varje vår händer samma sak.", x0=43, top=311, baseline=342)],
        [span("Marken har väntat på det.", x0=43, top=325, baseline=356)],
    ]
    rows = _group_lines_into_rows(lines, page_number=3, column="left")
    assert [r.text for r in rows] == ["DIKTEN", "Varje vår händer samma sak.", "Marken har väntat på det."]
    assert rows[0].is_subtitle


def chars_line(text: str, gaps: dict[int, float]) -> dict:
    """A rawdict line with one span. Characters are 5 pt wide; gaps[i] is the width of character i."""
    chars, x = [], 0.0
    for index, c in enumerate(text):
        width = gaps.get(index, 5.0)
        chars.append({"c": c, "bbox": (x, 0, x + width, 10)})
        x += width
    return {"spans": [{"chars": chars}]}


def test_soft_hyphen_printed_as_a_space_becomes_a_space():
    # 2019-04-06 has "Vilken\u00adrisk": a soft hyphen where the page shows a space.
    line = chars_line("Vilken­risk", gaps={6: 2.5})
    _fill_span_texts(line)
    assert line["spans"][0]["text"] == "Vilken risk"


def test_soft_hyphen_printed_after_a_digit_becomes_a_hyphen():
    line = chars_line("1990­talet", gaps={4: 3.5})
    _fill_span_texts(line)
    assert line["spans"][0]["text"] == "1990-talet"


def test_invisible_soft_hyphen_inside_a_word_is_left_for_removal():
    line = chars_line("forsknings­resultat", gaps={10: 0.1})
    _fill_span_texts(line)
    assert join_wrapped_lines([line["spans"][0]["text"]]) == "forskningsresultat"


# ---------------------------------------------------------------------------
# Booklet parsing from text rows
# ---------------------------------------------------------------------------

def row(text: str, *, x0: float = 43.0, y0: float = 100.0, column: str = "left",
        font: str = "Calibri", size: float = 11.0, page: int = 1) -> TextRow:
    """A text row with its column edge set to its own x, so that it counts as edge-aligned."""
    return TextRow(page=page, column=column, x0=x0, y0=y0, text=text, font=font, size=size,
                   x1=x0 + 200, edge=x0, right_edge=x0 + 240)


def parse(rows: list[TextRow]) -> dict[int, object]:
    parser = BookletParser()
    for item in rows:
        parser.feed(item)
    parser.finish()
    return {draft.number: draft for draft in parser.drafts}


def test_questions_in_two_columns_are_found_when_numbered_across_the_page():
    # 2013-2018 layout: question 1 top left, 2 top right, 3 below 1.
    rows = [
        row("1. hospitaliserad", x0=76, y0=85),
        row("A inbillningssjuk", x0=76, y0=114),
        row("2. alpinism", x0=302, y0=85, column="right"),
        row("A promenad", x0=302, y0=114, column="right"),
        row("3. gestalta", x0=76, y0=228),
        row("A fundera", x0=76, y0=257),
    ]
    drafts = parse(rows)
    assert sorted(drafts) == [1, 2, 3]
    assert drafts[2].options == {"A": ["promenad"]}
    assert drafts[3].options["A"] == ["fundera"]


def test_a_number_inside_passage_text_does_not_start_a_question():
    rows = [
        row("En gammal trädgård", x0=43, y0=56, size=22.0),
        row("2. Det var en gång", font="Cambria", size=10.0, y0=100),
        row("11. Hur har forskarna kommit fram?", y0=300, font="Calibri-Bold"),
        row("A Genom fynd.", y0=320),
    ]
    drafts = parse(rows)
    assert sorted(drafts) == [11]
    assert drafts[11].parts == ["Hur har forskarna kommit fram?"]


def test_passage_text_after_a_question_is_not_added_to_its_last_option():
    rows = [
        row("En gammal trädgård", x0=43, y0=56, size=22.0),
        row("Det traditionella sättet.", font="Cambria", size=10.0, y0=80),
        row("11. Hur?", y0=300, font="Calibri-Bold"),
        row("A Genom fynd.", y0=320),
        row("B Genom dna.", y0=340),
        row("Forskarna bakom studien.", font="Cambria", size=10.0, y0=360, column="right", x0=312),
    ]
    drafts = parse(rows)
    assert drafts[11].options == {"A": ["Genom fynd."], "B": ["Genom dna."]}


def test_a_passage_with_two_texts_under_one_heading_gives_both_texts_to_its_questions():
    rows = [
        row("Två texter om minnen", x0=43, y0=56, size=22.0),
        row("FÖRSTA DELEN", font="Cambria-Bold", size=10.0, y0=102),
        row("Första texten om minnen.", font="Cambria", size=10.0, y0=120),
        row("ANDRA DELEN", font="Cambria-Bold", size=10.0, y0=200),
        row("Andra texten om ett annat minne.", font="Cambria", size=10.0, y0=220),
        row("17. Vad kommer textförfattaren fram till?", y0=400, font="Calibri-Bold"),
        row("A Ett svar.", y0=420),
    ]
    drafts = parse(rows)
    group = drafts[17].group
    assert group.title == "Två texter om minnen"
    text = group.text()
    assert "Första delen" in text and "Första texten om minnen." in text
    assert "Andra delen" in text and "Andra texten om ett annat minne." in text


def test_author_line_is_kept_as_its_own_paragraph():
    # Questions name the authors ("Anna Exempels text"), so the names must stay with their texts.
    author = row("Anna Exempel", font="Cambria", size=10.0, x0=220)
    author.attribution = True
    group = PassageGroup(title="Två inlägg")
    for item in [row("Sista raden i texten.", font="Cambria", size=10.0), author,
                 row("floskel = tom fras", font="Cambria", size=10.0)]:
        group.add(item)
    assert group.text() == "Sista raden i texten.\n\nAnna Exempel\n\nfloskel = tom fras"


def test_question_31_closes_its_column_so_elf_text_does_not_join_question_30():
    rows = [
        row("30. Sista frågan.", y0=300, font="GillAltOneMT-Bold"),
        row("A första", y0=320),
        row("B andra", y0=340),
        row("C tredje", y0=360),
        row("D sista svaret", y0=380),
        row("31. English question", y0=400),
        row("A an english option", y0=420),
        row("More English text", y0=440),
    ]
    drafts = parse(rows)
    assert 31 not in drafts
    assert drafts[30].options["D"] == ["sista svaret"]


def test_an_option_letter_out_of_order_is_text_not_an_option():
    # "C" comes before "B": it is not taken as an option, so it stays in the text of option A.
    rows = [
        row("1. ord", y0=100),
        row("A första", y0=120),
        row("C tredje", y0=140),
        row("B andra", y0=160),
    ]
    drafts = parse(rows)
    assert drafts[1].options == {"A": ["första", "C tredje"], "B": ["andra"]}


# ---------------------------------------------------------------------------
# Answer keys
# ---------------------------------------------------------------------------

def make_key_pdf(path: Path, columns: list[tuple[int, str, dict[int, str]]]) -> Path:
    """A one-page key laid out like the real ones: a "Provpass N" header, a part label, and rows
    of number and letter. Numbers from 10 up are drawn as two touching digits, as in some years."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for index, (exam_pass, part, answers) in enumerate(columns):
        x = 62 + 130 * index
        page.insert_text((x, 98), f"Provpass {exam_pass}", fontsize=10)
        page.insert_text((x, 117), f"{part} del xy", fontsize=10)
        for number, letter in answers.items():
            y = 143 + 14 * (number - 1)
            number_x = x + 15
            digits = str(number)
            for offset, digit in enumerate(digits):
                page.insert_text((number_x + 6 * offset, y), digit, fontsize=10)
            page.insert_text((number_x + 17, y), letter, fontsize=10)
    page.insert_text((62, 800), "Provpass 2 ingår ej! (Utprövningspass)", fontsize=9)
    doc.save(path)
    doc.close()
    return path


def test_key_columns_are_read_by_header_and_part(tmp_path):
    answers_a = {n: "ABCDE"[n % 5] for n in range(1, 31)}
    answers_b = {n: "EDCBA"[n % 5] for n in range(1, 31)}
    key = make_key_pdf(tmp_path / "key.pdf", [(1, "Kvantitativ", answers_a), (3, "Verbal", answers_b)])
    columns = {(c.exam_pass, c.part): c.answers for c in read_answer_key(key)}
    assert columns[(1, "Kvantitativ")] == answers_a
    assert columns[(3, "Verbal")] == answers_b          # two-digit numbers (10 to 30) included
    assert (2, "Verbal") not in columns                  # the footnote "Provpass 2" is not a column


def test_real_2025_key_gives_the_official_verbal_answers_for_pass_3():
    if not HP_2025_KEY.exists():
        pytest.skip("2025-10-19 answer key not downloaded; run `bench fetch`")
    verbal = {c.exam_pass: c.answers for c in read_answer_key(HP_2025_KEY) if c.part == "Verbal"}
    assert "".join(verbal[3][n] for n in range(1, 31)) == HP_2025_PASS3_ANSWERS


def test_real_2025_booklet_gives_questions_passages_and_options():
    if not HP_2025_PASS3.exists():
        pytest.skip("2025-10-19 booklet not downloaded; run `bench fetch`")
    exam_pass, drafts = read_booklet(HP_2025_PASS3)
    assert exam_pass == 3
    by_number = {d.number: d for d in drafts}
    assert sorted(by_number) == list(range(1, 31))
    # Question 11 is in the first text; questions 17 to 20 are about two texts under one heading.
    # Expected exam texts are stored as fingerprints (sha256, first 16 hex digits), so the test stays
    # exact without publishing the exam's text.
    assert _fp(by_number[11].group.title) == "1b2de2a5ef2d5efd"
    assert _fp(by_number[17].group.title) == "7eb96c54716f14db"
    assert by_number[17].group is by_number[20].group
    assert _fp(join_wrapped_lines(by_number[11].options["A"])) == "4c86d84909004ad2"
    assert len(by_number[21].options) == 4


def test_real_2026_booklet_keeps_subtitles_and_authors_with_their_texts():
    booklet = RAW / "2026-10-18" / "provpass-2-verb-utan-elf.pdf"
    if not booklet.exists():
        pytest.skip("2026-10-18 booklet not downloaded; run `bench fetch`")
    _, drafts = read_booklet(booklet)
    text = {d.number: d for d in drafts}[11].group.text()
    # Question 11 asks about a poem by its title: the subtitle must head its own text, and the poet's and
    # the author's names must stay as their own paragraphs. Checked with fingerprints, not the exam text.
    assert _fp(text[:25]) == "401a50a65c8db69c"
    assert _contains_fp(text, 95, "63db8d1ee9e594f5")
    assert _fp(text[-28:]) == "80225021d3befdb8"


def test_verbal_answers_are_used_only_when_all_keys_agree():
    entry = ManifestEntry(url="u", path="p", sha256="s", size=1, fetched_at="t",
                          kind=KIND_ANSWER_KEY, exam_date="2019-10-20", exam_pass=None)
    from bench.extract import KeyColumn
    key_a = KeyFile(entry, [KeyColumn(5, "Verbal", {1: "A"})])
    key_b = KeyFile(entry, [KeyColumn(5, "Verbal", {1: "B"})])
    answers, _, problem = _verbal_answers([key_a, key_b], 5)
    assert answers is None and "different answers" in problem
    answers, _, problem = _verbal_answers([key_a, key_a], 5)
    assert answers == {1: "A"} and problem is None
    _, _, problem = _verbal_answers([key_a], 4)
    assert "no verbal column for provpass 4" in problem


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def make_question(number: int, section: str, *, question: str = "Fråga", options: dict | None = None,
                  answer: str = "A", passage: str | None = None) -> Question:
    if options is None:
        options = {letter: f"alternativ {letter}" for letter in "ABCDE"[: 5 if section == "ORD" else 4]}
    return Question(
        id=Question.make_id("2025-10-19", 3, section, number), exam_date="2025-10-19", exam_pass=3,
        section=section, number=number, question=question, options=options, answer=answer,
        passage=passage, passage_title="Titel" if passage else None,
        source={"booklet": "b.pdf", "page": 1, "answer_key": "k.pdf"},
    )


def make_valid_pass() -> list[Question]:
    questions = [make_question(n, "ORD") for n in range(1, 11)]
    questions += [make_question(n, "LAS", passage="En text.") for n in range(11, 21)]
    questions += [make_question(n, "MEK", question="Det blev _____ här.",
                                options={"A": "a", "B": "b", "C": "c", "D": "d"}) for n in range(21, 31)]
    return questions


def test_a_complete_pass_has_no_problems():
    assert validate_pass(make_valid_pass()) == []


def test_a_pass_with_nine_ord_questions_is_rejected():
    questions = [q for q in make_valid_pass() if q.number != 10]
    problems = validate_pass(questions)
    assert any("ORD: expected questions 1-10" in p for p in problems)


def test_answer_that_is_not_an_option_is_rejected():
    questions = make_valid_pass()
    questions[0].answer = "F"
    assert any("question 1: answer F is not one of the options" in p for p in validate_pass(questions))


def test_options_must_run_from_a():
    questions = make_valid_pass()
    questions[0].options = {"A": "a", "C": "c", "D": "d", "E": "e"}
    assert any("not A, B, C, ... from A" in p for p in validate_pass(questions))


def test_a_reading_question_without_passage_is_rejected():
    questions = make_valid_pass()
    for q in questions:
        if q.section == "LAS":
            q.passage = None
    assert any("no reading passage" in p for p in validate_pass(questions))


def test_multi_gap_options_must_have_one_part_per_gap():
    questions = make_valid_pass()
    mek = next(q for q in questions if q.number == 21)
    mek.question = "Först _____ och sedan _____."
    mek.options = {"A": "a – b – c", "B": "d – e", "C": "f – g", "D": "h – i"}
    assert any("option A has 3 parts for 2 gaps" in p for p in validate_pass(questions))


def test_a_sentence_without_a_gap_is_rejected():
    questions = make_valid_pass()
    mek = next(q for q in questions if q.number == 22)
    mek.question = "Ingen lucka här."
    assert any("no gap" in p for p in validate_pass(questions))


def test_to_question_takes_the_answer_from_the_key():
    draft = parse([row("1. ord", y0=100), row("A första", y0=120)])[1]
    question = to_question(draft, "2025-10-19", 3, {1: "A"}, "b.pdf", "k.pdf")
    assert question.id == "2025-10-19/p3/ORD/1"
    assert question.answer == "A" and question.options == {"A": "första"}
    assert question.source == {"booklet": "b.pdf", "page": 1, "answer_key": "k.pdf"}
