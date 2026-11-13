"""Cheap structural and heading-cue features for page-level section prediction.

These features complement word TF-IDF with layout signals (position in the
manuscript, page length, digit/uppercase ratios, numbered headings, citations,
figures/tables) and 17 heading-cue flags ("abstract", "references", ...).

They are deliberately simple, deterministic and cheap: no model is needed to
extract them, and Random Forest does not require scaling.
"""

from __future__ import annotations

import math
import re

import numpy as np

NUMBERED_HEADING = re.compile(r"^\s*\d{1,3}(?:\.\d{1,3})*[\.\)]?\s*[A-Z]")
LEAD_NUMBER = re.compile(r"^(?:\d{1,3}(?:\.\d{1,3})*[\.\)]?)\s*")
CITATION = re.compile(r"\[\d{1,3}(?:\s*[-,–]\s*\d{1,3})*\]")
FIGURE = re.compile(r"\bfig(?:ure)?\.?\s*\d+", re.I)
TABLE = re.compile(r"\btable\s*\d+", re.I)
EQUATION_LINE = re.compile(r"^\s*\(\d+\)\s*$")
WORD = re.compile(r"\w+")

LAYOUT_NAMES = [
    "page",
    "n_pages",
    "rel_pos",
    "remaining_ratio",
    "is_first_page",
    "is_last_page",
    "is_front_3_pages",
    "is_back_3_pages",
    "log_chars",
    "log_words",
    "log_lines",
    "mean_line_len",
    "digit_ratio",
    "upper_ratio",
    "alpha_ratio",
    "space_ratio",
    "punct_ratio",
    "log_numbered_headings",
    "log_citations",
    "log_figures",
    "log_tables",
    "log_equation_lines",
]

# One flag per canonical body label (title has no textual heading cue).
HEADING_PATTERNS = [
    ("cue_abstract", r"abstract\b"),
    ("cue_keywords", r"(?:key\s*words?|index\s+terms)\b"),
    ("cue_introduction", r"introduction\b"),
    ("cue_background", r"background\b"),
    ("cue_objectives", r"(?:objectives?|aims?|purpose)\b"),
    ("cue_analysis", r"(?:analysis|data\s+analysis|statistical\s+analysis)\b"),
    ("cue_study_design", r"(?:study\s+design|design|participants?|procedure)\b"),
    ("cue_materials", r"(?:materials?|materials?\s+and\s+methods?)\b"),
    ("cue_results", r"(?:results?|findings)\b"),
    ("cue_method", r"(?:methods?|methodology|materials?\s+and\s+methods?|proposed\s+method|approach)\b"),
    ("cue_experiments", r"(?:experiments?|experimental\s+(?:setup|results|evaluation))\b"),
    ("cue_evaluation", r"(?:evaluation|performance\s+evaluation)\b"),
    ("cue_discussion", r"(?:discussion|results?\s+and\s+discussion)\b"),
    ("cue_conclusion", r"(?:conclusions?|concluding\s+remarks)\b"),
    ("cue_references", r"(?:references?|bibliography)\b"),
    ("cue_declarations", r"(?:declarations?|acknowledg(?:e)?ments?|funding|conflict\s+of\s+interest|competing\s+interests|author\s+contributions)\b"),
    ("cue_supplementary", r"(?:supplementary|supplemental|appendix|appendices)\b"),
]
HEADING_NAMES = [name for name, _ in HEADING_PATTERNS]
_HEADING_REGEXES = [(name, re.compile(pattern, re.I)) for name, pattern in HEADING_PATTERNS]


def feature_names() -> list[str]:
    return LAYOUT_NAMES + HEADING_NAMES


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _has_heading_cue(lines: list[str], regex: re.Pattern) -> float:
    """True if a short line starts with the cue (numbering is stripped)."""
    for line in lines[:80]:
        if len(line) > 80:
            continue
        stripped = LEAD_NUMBER.sub("", line).strip()
        if regex.match(stripped):
            return 1.0
    return 0.0


def page_features(row: dict) -> np.ndarray:
    """Return the structural/heading feature vector for one page row."""
    text = row.get("text") or ""
    n_pages = _as_float(row.get("n_pages"))
    page = _as_float(row.get("page"))
    rel_pos = _as_float(
        row.get("rel_pos"), page / n_pages if n_pages else 0.0
    )
    remaining = max(0.0, 1.0 - rel_pos)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    n_lines = len(lines)
    n_chars = len(text)
    n_words = len(WORD.findall(text))

    letters = uppers = digits = spaces = punct = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if ch.isupper():
                uppers += 1
        elif ch.isdigit():
            digits += 1
        elif ch.isspace():
            spaces += 1
        else:
            punct += 1

    numbered = sum(
        1 for line in lines if len(line) <= 120 and NUMBERED_HEADING.match(line)
    )
    citations = len(CITATION.findall(text))
    figures = len(FIGURE.findall(text))
    tables = len(TABLE.findall(text))
    equations = sum(1 for line in lines if EQUATION_LINE.match(line))

    layout = [
        page,
        n_pages,
        rel_pos,
        remaining,
        float(page == 1.0),
        float(n_pages > 1.0 and page == n_pages),
        float(0.0 < page <= 3.0),
        float(n_pages > 1.0 and page >= n_pages - 2.0),
        math.log1p(n_chars),
        math.log1p(n_words),
        math.log1p(n_lines),
        _ratio(n_chars, n_lines),
        _ratio(digits, n_chars),
        _ratio(uppers, letters),
        _ratio(letters, n_chars),
        _ratio(spaces, n_chars),
        _ratio(punct, n_chars),
        math.log1p(numbered),
        math.log1p(citations),
        math.log1p(figures),
        math.log1p(tables),
        math.log1p(equations),
    ]
    cues = [_has_heading_cue(lines, regex) for _, regex in _HEADING_REGEXES]
    return np.asarray(layout + cues, dtype=np.float32)


def page_feature_matrix(rows: list[dict]) -> np.ndarray:
    """Stack page_features over rows."""
    return np.vstack([page_features(row) for row in rows]).astype(np.float32)
