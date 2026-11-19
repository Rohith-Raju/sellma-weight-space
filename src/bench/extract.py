"""Turn the downloaded verbal booklets and answer keys into one Question record per question.

Booklets
--------
A booklet is a two-column PDF. Plain text extraction scrambles the reading order (ORD comes
out as 2, 1, 4, 3, ...), so this module reads each text line with its coordinates and puts
the lines back in order itself:

- Text is grouped into lines per column. A line belongs to the column where it starts.
- Running headers ("LÄS", "MEK – Meningskomplettering") and page numbers sit in the top and
  bottom margins and are dropped by position, as are "Uppgifter" labels and blank-page notices.
- A question starts with "N." at the column edge, in the question font (Calibri, or GillAlt in
  the 2013-2018 booklets). Question numbers do not always follow reading order: the 2013-2018
  booklets run across the page. So each column keeps its own open question, and a number is
  accepted when it is in range and has not been seen yet. Question 31 and up is ELF, which
  closes its column for the rest of the booklet.
- Options are letters A to E at the start of a line, in order. Other lines continue the
  question or option before them.
- Reading passages are set in Cambria (2019 on) or AGaramond (2013-2018). A heading of 18 pt or
  more starts a passage group. A group can hold two texts, each with a bold uppercase subtitle
  (2025-10-19: "SÄKERHETSLINA" and "HJÄRTATS ROP" under one heading). A question belongs to the
  group whose heading came last before it, so a question about two texts gets both texts.
- Author names come at the end of a text, set flush right in its column. They are kept as a
  paragraph of their own, because questions name the authors ("Daniel Åbergs text").
- Words split by hyphenation ("gam-" + "malt") are joined; soft hyphens are removed.

Answer keys
-----------
A key is a one-page table with one column per provpass. Each column's header says whether
it is verbal ("Verbal del") or maths ("Kvantitativ del"). Each entry is a question number and
a letter on one row ("12  C"). The PDFs do not keep those two in the same order in the text
stream, so each letter is paired with the number just left of it on the same row.

Setting aside
-------------
A pass is kept only if validate_pass finds no problem with it. Otherwise the whole pass is
set aside, with the reasons written to the report. Nothing is patched by hand. Two booklets
that claim the same provpass in one exam are both set aside, because nothing printed on the
booklets says which answer key goes with which version.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cmp_to_key
from pathlib import Path
from typing import NamedTuple

import pymupdf

from bench.fetch import KIND_ANSWER_KEY, KIND_BOOKLET, ManifestEntry, load_manifest
from bench.paths import EXTRACT_REPORT, QUESTIONS, ROOT, SPLIT_CONFIG
from bench.records import LETTERS, Question, write_jsonl
from bench.split import load_split_config, split_name_of

# Question numbers per section. ELF (31-40) is not used.
SECTION_NUMBERS: dict[str, range] = {"ORD": range(1, 11), "LAS": range(11, 21), "MEK": range(21, 31)}
LAST_NUMBER = 30
FIRST_ELF_NUMBER = 31

# Passages are set in Cambria (2019 on) or AGaramond (2013-2018); questions in Calibri or GillAlt.
PASSAGE_FONT_PREFIXES = ("Cambria", "AGaramond")
HEADING_MIN_SIZE = 18.0        # points; the passage headings are 22 pt, "Uppgifter" is 16 pt
HEADER_MARGIN_Y = 45.0         # rows above this are running headers
FOOTER_MARGIN_Y = 790.0        # rows below this are page numbers
ROW_TOLERANCE_Y = 3.0          # lines whose baselines are this close belong to one visual line
SPACE_GAP_X = 1.5              # a horizontal gap wider than this is a space between spans
INDENT_POINTS = 5.0            # a first line this far right of the column edge starts a paragraph
NUMBER_INDENT = 6.0            # a question number starts at the column edge, give or take this
ATTRIBUTION_MAX_CHARS = 45     # author lines are short; a justified body line is about 55 characters
ATTRIBUTION_EDGE_TOLERANCE = 5.0   # and set flush with the right edge of their column
BODY_TEXT_MAX_SIZE = 12.0      # body text and labels are at most this size

SOFT_HYPHEN = "\u00ad"          # invisible in the middle of a word, shown as a hyphen at a line end
SOFT_HYPHEN_PRINTED_GAP = 1.3  # a soft hyphen inside a line with a wider gap than this around it is printed
SENTENCE_END_CHARACTERS = ".?!:;\"”)"
NAME_CONNECTORS = {"och", "&", "and", "von", "van", "de", "af", "av"}
WORDS_AFTER_A_KEPT_HYPHEN = {"och", "eller"}   # "djur- och växtdelar": the hyphen is not a line break
GAP_PATTERN = re.compile(r"_{3,}")   # the gap is 4 or 5 underscores, depending on the year
MULTI_GAP_SEPARATOR = " – "

QUESTION_START = re.compile(r"^(\d{1,2})\.\s*(.*)$")
OPTION_START = re.compile(r"^([A-E])(?:\s+(.*))?$")
PASS_NUMBER_ON_COVER = re.compile(r"Provpass\s+(\d)")
KEY_NUMBER_TO_LETTER_GAP = 30.0   # points between a question number and its answer letter in a key


# ---------------------------------------------------------------------------
# Reading the booklet PDF into text rows
# ---------------------------------------------------------------------------

@dataclass
class TextRow:
    """One visual line of text, with the information needed to order and classify it."""
    page: int
    column: str            # "left" or "right"
    x0: float
    y0: float
    text: str              # raw: keeps a trailing space, which tells a real hyphen from a line break
    font: str
    size: float
    x1: float = 0.0            # right edge of the text
    edge: float = 0.0          # left edge of the column's body text, set by _mark_indents
    right_edge: float = 0.0    # right edge of the column's body text, set by _mark_indents
    indented: bool = False
    attribution: bool = False  # an author line under a text, set by _mark_attributions

    @property
    def is_heading(self) -> bool:
        return self.size >= HEADING_MIN_SIZE

    @property
    def is_passage(self) -> bool:
        return self.font.startswith(PASSAGE_FONT_PREFIXES)

    @property
    def is_subtitle(self) -> bool:
        # The text subtitles are bold uppercase Cambria, e.g. "SÄKERHETSLINA".
        return self.is_passage and "Bold" in self.font and self.text.strip().isupper()

    @property
    def at_column_edge(self) -> bool:
        return self.x0 - self.edge <= NUMBER_INDENT


def read_rows(page: pymupdf.Page, page_number: int) -> list[TextRow]:
    """All text rows of one page in reading order: left column top to bottom, then right."""
    middle = page.rect.width / 2
    lines_by_column: dict[str, list[list[dict]]] = {"left": [], "right": []}
    for block in page.get_text("rawdict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            _fill_span_texts(line)
            # Whitespace-only spans are kept: a space before a line break tells a real
            # hyphen ("djur- och") from a hyphenated word ("gam-" + "malt").
            spans = [s for s in line["spans"] if s["text"]]
            if not spans or not _in_body_margins(line["bbox"][1]):
                continue
            # A text line is assigned to a column by where it starts. A span can start right of
            # the middle while belonging to the left column (2016 layout), so spans are not used.
            column = "left" if line["bbox"][0] < middle else "right"
            lines_by_column[column].append(spans)

    rows: list[TextRow] = []
    for column in ("left", "right"):
        column_rows = _group_lines_into_rows(lines_by_column[column], page_number, column)
        column_rows = [row for row in column_rows if row.text.strip()]
        rows.extend(_mark_attributions(_drop_labels(_mark_indents(column_rows))))
    return rows


def _fill_span_texts(line: dict) -> None:
    """Give every span of a line its text. PyMuPDF's "rawdict" lists characters instead.

    The characters are needed for one thing: a soft hyphen inside a line is normally an
    invisible break point, but some lines have one where something is printed. It then has a
    visible gap between its neighbours. After a digit it is a hyphen ("1990\u00adtalet"); elsewhere
    it is a space ("Vilken\u00adrisk\u00adser" in 2019-04-06 and 2023-03-25).
    """
    chars = [(span, char) for span in line["spans"] for char in span["chars"]]
    for span in line["spans"]:
        span["text"] = ""
    for index, (span, char) in enumerate(chars):
        text = char["c"]
        if text == SOFT_HYPHEN and 0 < index < len(chars) - 1:
            before, after = chars[index - 1][1], chars[index + 1][1]
            if after["c"].strip() and after["bbox"][0] - before["bbox"][2] > SOFT_HYPHEN_PRINTED_GAP:
                text = "-" if before["c"].isdigit() else " "
        span["text"] += text


def _in_body_margins(y: float) -> bool:
    return HEADER_MARGIN_Y <= y <= FOOTER_MARGIN_Y


def _group_lines_into_rows(lines: list[list[dict]], page_number: int, column: str) -> list[TextRow]:
    """Text lines at the same height are one visual row: the label "A" and its text are separate lines.

    `lines` is in the order the PDF draws them. Within a row the lines keep that order, and the
    rows are sorted by height (the sort is stable, so equal heights keep their order too).

    Height is the baseline the text stands on, not the top of its box. The box top depends on
    the font: in the 2019-2026 booklets the box of regular Cambria starts 21 pt higher than that
    of Cambria Bold or Italic on the same baseline, so by box top a bold subtitle ("LERAN")
    would be placed one or two lines too far down, in the middle of the text it heads.
    """
    by_height = sorted(enumerate(lines), key=lambda item: _baseline(item[1]))
    groups: list[list[tuple[int, list[dict]]]] = []
    for index, spans in by_height:
        if groups and abs(_baseline(spans) - _baseline(groups[-1][-1][1])) <= ROW_TOLERANCE_Y:
            groups[-1].append((index, spans))
        else:
            groups.append([(index, spans)])
    rows = []
    for group in groups:
        in_pdf_order = [spans for _, spans in sorted(group, key=lambda item: item[0])]
        rows.append(_row_from_lines(in_pdf_order, page_number, column))
    return rows


def _baseline(spans: list[dict]) -> float:
    """The baseline of a text line: that of its longest piece, so a footnote mark does not move it."""
    return max(spans, key=lambda span: len(span["text"]))["origin"][1]


def _row_from_lines(lines: list[list[dict]], page_number: int, column: str) -> TextRow:
    text = ""
    previous_right_edge = None
    for spans in _ordered_lines(lines):
        for span in spans:
            # A gap between two pieces is a space. Overlapping pieces get none.
            if previous_right_edge is not None and span["bbox"][0] - previous_right_edge > SPACE_GAP_X:
                text += " "
            text += span["text"].replace("\u00a0", " ")
            previous_right_edge = span["bbox"][2]
    all_spans = [span for spans in lines for span in spans]
    return TextRow(
        page=page_number,
        column=column,
        x0=min(span["bbox"][0] for span in all_spans),
        x1=max(span["bbox"][2] for span in all_spans),
        y0=min(span["bbox"][1] for span in all_spans),
        text=text,
        font=lines[0][0]["font"],
        size=max(span["size"] for span in all_spans),
    )


def _ordered_lines(lines: list[list[dict]]) -> list[list[dict]]:
    """Lines of one row, left to right. Lines drawn over each other keep their order in the PDF.

    A bold "30." can come out as the line "3" and the line "0. I ...", which overlap in x:
    sorting by x would put them the wrong way round. The label "A" and its text never overlap,
    so they are sorted by x as usual.
    """
    def compare(a: tuple[int, list[dict]], b: tuple[int, list[dict]]) -> int:
        (index_a, line_a), (index_b, line_b) = a, b
        a_start, a_end = _x_extent(line_a)
        b_start, b_end = _x_extent(line_b)
        if a_start < b_end and b_start < a_end:
            return index_a - index_b
        if a_start != b_start:
            return -1 if a_start < b_start else 1
        return index_a - index_b

    return [line for _, line in sorted(enumerate(lines), key=cmp_to_key(compare))]


def _x_extent(spans: list[dict]) -> tuple[float, float]:
    return min(s["bbox"][0] for s in spans), max(s["bbox"][2] for s in spans)


def _mark_indents(rows: list[TextRow]) -> list[TextRow]:
    """Mark first lines of paragraphs: they start further right than the column's body text."""
    body = [row.x0 for row in rows if row.size <= BODY_TEXT_MAX_SIZE]
    if not body:
        return rows
    edge = min(body)
    right_edge = max(row.x1 for row in rows if row.size <= BODY_TEXT_MAX_SIZE)
    for row in rows:
        row.edge = edge
        row.right_edge = right_edge
        row.indented = row.x0 - edge > INDENT_POINTS
    return rows


def _drop_labels(rows: list[TextRow]) -> list[TextRow]:
    """Remove labels that are not part of a question or a text."""
    return [row for row in rows if row.text.strip() != "Uppgifter" and "BLANKSIDA" not in row.text]


def _mark_attributions(rows: list[TextRow]) -> list[TextRow]:
    for row in rows:
        row.attribution = _is_attribution(row)
    return rows


def _is_attribution(row: TextRow) -> bool:
    """An author line: short, set flush right in its column, and not ending a sentence.

    Justified text also ends flush right, but its lines are long. Short paragraph ends are
    ragged on the right and usually end with punctuation.
    """
    if not row.is_passage or row.is_subtitle:
        return False
    flush_right = row.x1 >= row.right_edge - ATTRIBUTION_EDGE_TOLERANCE
    return (flush_right
            and len(row.text.strip()) <= ATTRIBUTION_MAX_CHARS
            and row.text.rstrip()[-1:] not in SENTENCE_END_CHARACTERS
            and _looks_like_names(row.text))


def _looks_like_names(text: str) -> bool:
    """Every word capitalised, apart from joining words such as "och" or "&"."""
    words = text.replace(",", " ").split()
    return bool(words) and all(w[0].isupper() or w.lower() in NAME_CONNECTORS for w in words)


# ---------------------------------------------------------------------------
# Joining wrapped lines
# ---------------------------------------------------------------------------

def join_wrapped_lines(lines: list[str]) -> str:
    """Join the lines of one paragraph into running text.

    - A soft hyphen at a line end is dropped: "forn" + "tidens" is "forntidens". After a digit
      it stands for a real hyphen: "1950" + "talet" is "1950-talet".
    - "gam-" followed by a lowercase line is a hyphenated word: "gammalt". The PDFs often have a
      space after such a hyphen ("arbets- " + "tagare"), so the space does not make it a dash.
    - The hyphen is kept when it is a real one: before a capital ("Bengt-Samuelsson"), after a
      digit or a capital ("1200-talet", "EU-vännerna"), and before "och" or "eller", where the
      word is completed later ("djur- och växtdelar").
    """
    text = ""
    for raw in lines:
        line = raw.lstrip().lstrip(SOFT_HYPHEN)   # a soft hyphen at a line start means nothing
        if not line:
            continue
        end = text.rstrip()
        if end.endswith("-" + SOFT_HYPHEN):
            end = end[:-1]
        if not text:
            text = line
        elif end.endswith(SOFT_HYPHEN):
            text = _join_after_soft_hyphen(end[:-1], line)
        elif end.endswith("-"):
            text = _join_after_hyphen(end, line)
        else:
            text = end + " " + line
    return _tidy(text)


def _join_after_soft_hyphen(text: str, line: str) -> str:
    """`text` ended with a soft hyphen, which is already removed; `line` is the next line."""
    if text[-1:].isdigit():
        return text + "-" + line        # "1950-talet"
    if line[0].isupper():
        return text + " " + line        # not a broken word: "Lena" + "Marander-Eklund"
    return text + line


def _join_after_hyphen(text: str, line: str) -> str:
    """`text` ends with a hyphen at a line end; `line` is the next line."""
    before_hyphen = text[-2:-1]
    if not before_hyphen.strip() or line.split()[0] in WORDS_AFTER_A_KEPT_HYPHEN:
        return text + " " + line        # a free-standing dash, or "djur- och växtdelar"
    real_compound = line[0].isupper() or before_hyphen.isdigit() or before_hyphen.isupper()
    return (text if real_compound else text[:-1]) + line


def _tidy(text: str) -> str:
    text = text.replace(SOFT_HYPHEN, "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------

@dataclass
class PassageText:
    subtitle: str | None
    rows: list[TextRow] = field(default_factory=list)

    def render(self) -> str:
        paragraphs = _paragraphs(self.rows)
        body = "\n\n".join(join_wrapped_lines(p) for p in paragraphs)
        return f"{self.subtitle}\n\n{body}" if self.subtitle else body


@dataclass
class PassageGroup:
    """Everything printed under one 22 pt heading: one or more texts."""
    title: str
    texts: list[PassageText] = field(default_factory=list)

    def add(self, row: TextRow) -> None:
        if row.is_subtitle:
            self.texts.append(PassageText(subtitle=row.text.strip().capitalize()))
            return
        if not self.texts:
            self.texts.append(PassageText(subtitle=None))
        self.texts[-1].rows.append(row)

    def text(self) -> str:
        return "\n\n".join(t.render() for t in self.texts if t.rows)


def _paragraphs(rows: list[TextRow]) -> list[list[str]]:
    """Rows grouped into paragraphs. An author line is a paragraph of its own, so the name is
    not read as part of the text before it or of the word list after it."""
    paragraphs: list[list[str]] = []
    previous = None
    for row in rows:
        if row.indented or previous is None or row.attribution != previous.attribution:
            paragraphs.append([])
        paragraphs[-1].append(row.text)
        previous = row
    return paragraphs


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

@dataclass
class DraftQuestion:
    number: int
    page: int
    group: PassageGroup | None
    parts: list[str] = field(default_factory=list)        # lines of the question text
    options: dict[str, list[str]] = field(default_factory=dict)

    def next_letter(self) -> str:
        return LETTERS[len(self.options)] if len(self.options) < len(LETTERS) else ""

    def start_option(self, letter: str, first_line: str) -> None:
        self.options[letter] = [first_line] if first_line else []

    def add_line(self, text: str) -> None:
        if self.options:
            last = next(reversed(self.options))
            self.options[last].append(text)
        else:
            self.parts.append(text)


class BookletParser:
    """Feeds text rows in reading order and collects the questions 1 to 30.

    Each column keeps its own open question. The numbering is not strictly in reading order:
    the 2013-2018 booklets go across the page, so question 2 can come after question 3 in the
    stream. A number is therefore accepted when it is in range, edge-aligned, and not seen yet.
    """

    def __init__(self) -> None:
        self.drafts: list[DraftQuestion] = []
        self._group: PassageGroup | None = None
        self._open: dict[str, DraftQuestion | None] = {"left": None, "right": None}
        self._started: set[int] = set()
        self._ignored_columns: set[str] = set()

    def feed(self, row: TextRow) -> None:
        if row.is_heading:
            self._close_all()
            self._group = PassageGroup(title=row.text.strip())
            return
        if row.column in self._ignored_columns:
            return
        if row.is_passage:
            self._close(row.column)
            if self._group is not None:
                self._group.add(row)
            return
        if self._start_question_if_numbered(row):
            return
        draft = self._open[row.column]
        if draft is not None:
            self._continue_draft(draft, row.text)

    def finish(self) -> None:
        self._close_all()

    def _start_question_if_numbered(self, row: TextRow) -> bool:
        match = QUESTION_START.match(row.text.lstrip())
        if match is None or row.is_passage or not row.at_column_edge:
            return False
        number = int(match.group(1))
        if number >= FIRST_ELF_NUMBER:
            # ELF. Nothing after it in this column belongs to ORD, LÄS or MEK.
            self._close(row.column)
            self._ignored_columns.add(row.column)
            return True
        if number in self._started or not 1 <= number <= LAST_NUMBER:
            return False
        self._close(row.column)
        draft = DraftQuestion(number=number, page=row.page, group=self._group)
        self._started.add(number)
        if match.group(2).strip():
            draft.parts.append(match.group(2))
        self._open[row.column] = draft
        return True

    def _continue_draft(self, draft: DraftQuestion, raw: str) -> None:
        # The raw text is kept, trailing space included; only the label is matched on stripped text.
        option = OPTION_START.match(raw.lstrip())
        if option is not None and option.group(1) == draft.next_letter():
            first_line = option.group(2) or ""
            draft.start_option(option.group(1), first_line if first_line.strip() else "")
            return
        if raw.strip():
            draft.add_line(raw)

    def _close(self, column: str) -> None:
        draft = self._open[column]
        if draft is not None:
            self.drafts.append(draft)
            self._open[column] = None

    def _close_all(self) -> None:
        for column in self._open:
            self._close(column)


def read_booklet(path: Path) -> tuple[int | None, list[DraftQuestion]]:
    """The provpass number printed on the cover (None if not printed) and the drafted questions."""
    parser = BookletParser()
    with pymupdf.open(path) as doc:
        cover = doc[0].get_text()
        match = PASS_NUMBER_ON_COVER.search(cover)
        exam_pass = int(match.group(1)) if match else None
        for page_number, page in enumerate(doc, start=1):
            for row in read_rows(page, page_number):
                parser.feed(row)
    parser.finish()
    return exam_pass, parser.drafts


def section_of(number: int) -> str | None:
    for section, numbers in SECTION_NUMBERS.items():
        if number in numbers:
            return section
    return None


def to_question(draft: DraftQuestion, exam_date: str, exam_pass: int, answers: dict[int, str],
                booklet: str, answer_key: str) -> Question:
    section = section_of(draft.number)
    assert section is not None  # the parser only produces numbers 1 to 30
    passage = draft.group.text() if section == "LAS" and draft.group else None
    passage_title = draft.group.title if section == "LAS" and draft.group else None
    return Question(
        id=Question.make_id(exam_date, exam_pass, section, draft.number),
        exam_date=exam_date,
        exam_pass=exam_pass,
        section=section,
        number=draft.number,
        question=join_wrapped_lines(draft.parts),
        options={letter: join_wrapped_lines(lines) for letter, lines in draft.options.items()},
        answer=answers.get(draft.number, ""),
        passage=passage,
        passage_title=passage_title,
        source={"booklet": booklet, "page": draft.page, "answer_key": answer_key},
    )


# ---------------------------------------------------------------------------
# Answer keys
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyColumn:
    exam_pass: int
    part: str                   # "Verbal" or "Kvantitativ"
    answers: dict[int, str]     # question number -> letter


class KeyWord(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def middle_y(self) -> float:
        return (self.y0 + self.y1) / 2


def read_answer_key(path: Path) -> list[KeyColumn]:
    """Every provpass column of an answer key, with its answers."""
    columns: dict[tuple[int, str], dict[int, str]] = {}
    with pymupdf.open(path) as doc:
        for page in doc:
            words = [KeyWord(*w[:5]) for w in page.get_text("words")]
            for column in _key_columns_on_page(words):
                merged = columns.setdefault((column.exam_pass, column.part), {})
                merged.update(column.answers)
    return [KeyColumn(exam_pass, part, answers) for (exam_pass, part), answers in columns.items()]


def _key_columns_on_page(words: list[KeyWord]) -> list[KeyColumn]:
    pass_headers = _pass_headers(words)        # (x, provpass)
    if not pass_headers:
        return []
    part_labels = [(w.x0, w.text) for w in words if w.text in ("Verbal", "Kvantitativ")]
    numbers = _question_numbers(words)

    answers_by_header: list[dict[int, str]] = [{} for _ in pass_headers]
    for word in words:
        if word.text not in LETTERS:
            continue
        number = _number_left_of(word, numbers)
        if number is None:
            continue
        header = min(range(len(pass_headers)), key=lambda i: abs(pass_headers[i][0] - word.x0))
        answers_by_header[header][int(number.text)] = word.text

    columns = []
    for (x, exam_pass), answers in zip(pass_headers, answers_by_header):
        nearest_label = min(part_labels, key=lambda label: abs(label[0] - x), default=(x, "Kvantitativ"))
        columns.append(KeyColumn(exam_pass, nearest_label[1], answers))
    return columns


def _question_numbers(words: list[KeyWord]) -> list[KeyWord]:
    """The digit words, with two-digit numbers joined: "1" and "0" touching on one row is "10"."""
    digits = sorted((w for w in words if w.text.isdigit()), key=lambda w: (round(w.middle_y), w.x0))
    merged: list[KeyWord] = []
    for word in digits:
        previous = merged[-1] if merged else None
        touching = (previous is not None
                    and abs(previous.middle_y - word.middle_y) <= ROW_TOLERANCE_Y
                    and 0 <= word.x0 - previous.x1 < 2.0)
        if touching:
            merged[-1] = KeyWord(previous.x0, previous.y0, word.x1, word.y1, previous.text + word.text)
        else:
            merged.append(word)
    return merged


def _number_left_of(letter: KeyWord, numbers: list[KeyWord]) -> KeyWord | None:
    """The question number printed on the same row, just left of an answer letter.

    The layout is "12  C" on one row. Pairing is by position, not by the order of the words
    in the file, because the PDFs do not keep the two in the same order.
    """
    candidates = [n for n in numbers
                  if abs(n.middle_y - letter.middle_y) <= ROW_TOLERANCE_Y
                  and 0 <= letter.x0 - n.x1 <= KEY_NUMBER_TO_LETTER_GAP]
    return max(candidates, key=lambda n: n.x1, default=None)


def _pass_headers(words: list[KeyWord]) -> list[tuple[float, int]]:
    """The "Provpass N" headers of the table, as (x, provpass). The footnote "Provpass 2 ingår ej"
    is on another row and is left out."""
    found = []
    for word, following in zip(words, words[1:]):
        if word.text == "Provpass" and following.text.isdigit():
            found.append((word.x0, word.y0, int(following.text)))
    if not found:
        return []
    table_row_y = found[0][1]
    return [(x, number) for x, y, number in found if abs(y - table_row_y) < ROW_TOLERANCE_Y]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pass(questions: list[Question]) -> list[str]:
    """Every reason this pass cannot be used. An empty list means the pass is good."""
    problems: list[str] = []
    for section, numbers in SECTION_NUMBERS.items():
        found = sorted(q.number for q in questions if q.section == section)
        if found != list(numbers):
            problems.append(f"{section}: expected questions {numbers.start}-{numbers.stop - 1}, found {found}")
    for question in questions:
        problems.extend(_question_problems(question))
    return problems


def _question_problems(q: Question) -> list[str]:
    where = f"question {q.number}"
    problems = []
    if not q.question.strip():
        problems.append(f"{where}: empty question text")
    letters = sorted(q.options)
    if len(letters) < 4 or letters != list(LETTERS[:len(letters)]):
        problems.append(f"{where}: options {letters} are not A, B, C, ... from A")
    if any(not text.strip() for text in q.options.values()):
        problems.append(f"{where}: an option is empty")
    if not q.answer:
        problems.append(f"{where}: no answer in the answer key")
    elif q.answer not in q.options:
        problems.append(f"{where}: answer {q.answer} is not one of the options")
    if q.section == "LAS" and not (q.passage or "").strip():
        problems.append(f"{where}: no reading passage")
    if q.section == "MEK":
        problems.extend(_gap_problems(q))
    return problems


def _gap_problems(q: Question) -> list[str]:
    where = f"question {q.number}"
    gaps = len(GAP_PATTERN.findall(q.question))
    if gaps == 0:
        return [f"{where}: no gap (a run of underscores) in the sentence"]
    if gaps == 1:
        return []
    problems = []
    for letter, text in q.options.items():
        parts = text.split(MULTI_GAP_SEPARATOR)
        if len(parts) != gaps:
            problems.append(f"{where}: option {letter} has {len(parts)} parts for {gaps} gaps")
    return problems


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

@dataclass
class PassResult:
    exam_date: str
    exam_pass: int | None
    booklet: str
    answer_keys: list[str]
    questions: list[Question]        # everything parsed, kept or not
    problems: list[str]              # empty when the pass is kept

    @property
    def kept(self) -> bool:
        return not self.problems

    def count(self, section: str) -> int:
        return sum(1 for q in self.questions if q.section == section)


@dataclass
class ParsedBooklet:
    entry: ManifestEntry
    exam_pass: int | None
    drafts: list[DraftQuestion]
    error: str | None = None


@dataclass
class KeyFile:
    entry: ManifestEntry
    columns: list[KeyColumn]


def extract_all() -> None:
    """Build exams/questions.jsonl and exams/extract_report.md from everything in the manifest."""
    manifest = list(load_manifest().values())
    exam_dates = sorted({entry.exam_date for entry in manifest})
    results: list[PassResult] = []
    for exam_date in exam_dates:
        entries = [entry for entry in manifest if entry.exam_date == exam_date]
        booklets = [e for e in entries if e.kind == KIND_BOOKLET]
        keys = [_read_key_entry(e) for e in entries if e.kind == KIND_ANSWER_KEY]
        print(f"{exam_date}: {len(booklets)} booklet(s), {len(keys)} answer key(s)")
        results.extend(_extract_exam(exam_date, booklets, keys))

    kept = sorted((q for r in results if r.kept for q in r.questions),
                  key=lambda q: (q.exam_date, q.exam_pass, q.number))
    write_jsonl(QUESTIONS, kept)
    split_config = load_split_config(SPLIT_CONFIG)
    EXTRACT_REPORT.write_text(render_report(exam_dates, results, split_config), encoding="utf-8")
    kept_passes = sum(1 for r in results if r.kept)
    print(f"Kept {kept_passes} of {len(results)} passes, {len(kept)} questions. "
          f"Set aside: {len(results) - kept_passes}. Report: {EXTRACT_REPORT}")


def _read_key_entry(entry: ManifestEntry) -> KeyFile:
    return KeyFile(entry=entry, columns=read_answer_key(ROOT / entry.path))


def _parse_booklet(entry: ManifestEntry) -> ParsedBooklet:
    try:
        exam_pass, drafts = read_booklet(ROOT / entry.path)
    except RuntimeError as error:  # PyMuPDF raises RuntimeError for unreadable files
        return ParsedBooklet(entry, entry.exam_pass, [], error=str(error))
    return ParsedBooklet(entry, exam_pass if exam_pass is not None else entry.exam_pass, drafts)


def _extract_exam(exam_date: str, booklet_entries: list[ManifestEntry],
                  keys: list[KeyFile]) -> list[PassResult]:
    parsed = [_parse_booklet(entry) for entry in booklet_entries]
    claims = Counter(p.exam_pass for p in parsed if p.exam_pass is not None)
    return [_pass_result(exam_date, p, keys, claims) for p in parsed]


def _pass_result(exam_date: str, parsed: ParsedBooklet, keys: list[KeyFile],
                 claims: Counter) -> PassResult:
    booklet = parsed.entry.path
    key_paths = [k.entry.path for k in keys]
    if parsed.error:
        return PassResult(exam_date, parsed.exam_pass, booklet, key_paths, [],
                          [f"booklet cannot be read: {parsed.error}"])
    if parsed.exam_pass is None:
        return PassResult(exam_date, None, booklet, key_paths, [],
                          ["cannot tell which provpass this booklet is"])

    problems: list[str] = []
    if claims[parsed.exam_pass] > 1:
        problems.append(f"{claims[parsed.exam_pass]} booklets claim provpass {parsed.exam_pass}; "
                        "cannot tell which answer key goes with which")
    answers, answer_key, key_problem = _verbal_answers(keys, parsed.exam_pass)
    if key_problem:
        problems.append(key_problem)

    questions = [to_question(d, exam_date, parsed.exam_pass, answers or {}, booklet, answer_key or "")
                 for d in parsed.drafts]
    # Without an answer key the questions have no answers, so their checks would only repeat the key problem.
    if answers is not None:
        problems.extend(validate_pass(questions))
    return PassResult(exam_date, parsed.exam_pass, booklet, key_paths, questions, problems)


def _verbal_answers(keys: list[KeyFile], exam_pass: int) -> tuple[dict[int, str] | None, str | None, str | None]:
    """The verbal answers for one provpass, the key they came from, and a problem if there is one.

    Several answer keys can carry the same provpass (the 2018 spring exam printed eight
    versions). Their answers are used only if they all agree.
    """
    candidates = [(k.entry.path, c.answers) for k in keys for c in k.columns
                  if c.exam_pass == exam_pass and c.part == "Verbal"]
    if not candidates:
        return None, None, f"no verbal column for provpass {exam_pass} in the answer keys"
    first_path, first_answers = candidates[0]
    if any(answers != first_answers for _, answers in candidates):
        return None, None, (f"{len(candidates)} answer key columns give different answers for provpass "
                            f"{exam_pass}; cannot tell which version this booklet is")
    return first_answers, first_path, None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(exam_dates: list[str], results: list[PassResult], split_config: dict) -> str:
    kept = [r for r in results if r.kept]
    lines = [
        "# Extraction report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `bench extract`.",
        "",
        f"- Exams with files: {len(exam_dates)}",
        f"- Passes found: {len(results)}; kept: {len(kept)}; set aside: {len(results) - len(kept)}",
        f"- Questions kept: {sum(len(r.questions) for r in kept)}",
        "",
        "## Passes",
        "",
        "| Exam | Split | Pass | Booklet | ORD | LÄS | MEK | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(_pass_row(result, split_config))
    lines += ["", "## Set aside", ""]
    set_aside = [r for r in results if not r.kept]
    if not set_aside:
        lines.append("None.")
    for result in set_aside:
        lines.append(f"- {result.exam_date}, provpass {result.exam_pass}, `{result.booklet}`:")
        lines += [f"  - {problem}" for problem in result.problems]
    lines += ["", "## Exams without a verbal booklet", ""]
    lines += _exams_without_booklets(exam_dates, results)
    kept_questions = [q for result in kept for q in result.questions]
    lines += ["", "## Questions kept per split", "", "| Split | Exams | Questions | ORD | LÄS | MEK |",
              "|---|---|---|---|---|---|"]
    lines += _split_totals(kept_questions, split_config)
    lines += _configured_dates_warnings(kept_questions, split_config)
    return "\n".join(lines) + "\n"


def _pass_row(result: PassResult, split_config: dict) -> str:
    status = "kept" if result.kept else f"set aside ({len(result.problems)} problem(s))"
    split = split_name_of(result.exam_date, split_config)
    counts = [str(result.count(s)) for s in ("ORD", "LAS", "MEK")]
    return (f"| {result.exam_date} | {split} | {result.exam_pass or '?'} | `{Path(result.booklet).name}` | "
            f"{' | '.join(counts)} | {status} |")


def _exams_without_booklets(exam_dates: list[str], results: list[PassResult]) -> list[str]:
    with_booklets = {r.exam_date for r in results}
    missing = [d for d in exam_dates if d not in with_booklets]
    return [f"- {d}" for d in missing] or ["None."]


def _split_totals(kept: list[Question], split_config: dict) -> list[str]:
    rows = []
    for split in ("test", "gemma_extra", "dev"):
        in_split = [q for q in kept if split_name_of(q.exam_date, split_config) == split]
        exams = len({q.exam_date for q in in_split})
        counts = [sum(1 for q in in_split if q.section == s) for s in ("ORD", "LAS", "MEK")]
        rows.append(f"| {split} | {exams} | {len(in_split)} | {' | '.join(map(str, counts))} |")
    return rows


def _configured_dates_warnings(kept: list[Question], split_config: dict) -> list[str]:
    kept_dates = {q.exam_date for q in kept}
    lines = []
    for split in ("test", "gemma_extra"):
        for exam_date in split_config.get(split, []):
            if exam_date not in kept_dates:
                lines.append(f"\nWARNING: {split} exam {exam_date} has no kept questions.")
    return lines
