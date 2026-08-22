import json

import pytest

from src.ingest import build_sections, extract_illustrations, validate
from src.config import load_act_config
from src.parse_index import parse_index


@pytest.fixture(scope="module")
def sections():
    return build_sections("bns")


def test_exactly_358_sections(sections):
    assert len(sections) == 358


def test_no_chunk_below_minimum(sections):
    cfg = load_act_config("bns")
    short = [s.section_number for s in sections if s.char_count < cfg["min_chunk_chars"]]
    assert short == []


def test_no_chunk_above_maximum(sections):
    cfg = load_act_config("bns")
    long = [s.section_number for s in sections if s.char_count > cfg["max_chunk_chars"]]
    assert long == []


def test_definitions_section_is_kept_whole(sections):
    """Section 2 is 12,904 characters. SPEC.md's original 6000 ceiling would
    have forced a split, violating the more important rule that a section is
    never split. The ceiling moved; the section stays whole."""
    section_2 = next(s for s in sections if s.section_number == "2")
    assert section_2.char_count > 12000
    assert "wrongful loss" in section_2.text
    assert "words and expressions used but not defined" in section_2.text


def test_theft_section_shape(sections):
    theft = next(s for s in sections if s.section_number == "303")
    assert theft.id == "bns-303"
    assert theft.section_title == "Theft"
    assert theft.chapter_number == "XVII"
    assert theft.chapter_title == "OF OFFENCES AGAINST PROPERTY"
    assert theft.source_page == 88
    assert theft.act_number == "45 of 2023"
    assert theft.status == "enacted"
    assert theft.as_of_date == "2025-10-06"


def test_theft_illustrations_are_extracted(sections):
    theft = next(s for s in sections if s.section_number == "303")
    assert len(theft.illustrations) >= 7
    joined = " ".join(theft.illustrations)
    assert "cuts down a tree" in joined
    assert "finds a ring belonging to Z" in joined


def test_illustrations_stay_in_body_text_too(sections):
    """Illustrations are the highest-value text for semantic matching, so
    they are duplicated into the array rather than moved out of the body."""
    theft = next(s for s in sections if s.section_number == "303")
    assert "finds a ring belonging to Z" in theft.text


def test_section_text_does_not_leak_into_the_next_section(sections):
    theft = next(s for s in sections if s.section_number == "303")
    assert "304." not in theft.text
    assert "Snatching" not in theft.text


def test_footnote_absent_from_section_2(sections):
    section_2 = next(s for s in sections if s.section_number == "2")
    assert "1st day of July, 2024" not in section_2.text


def test_extract_illustrations_splits_lettered_items():
    text = (
        "Some body text.\n"
        "Illustrations.\n"
        "(a) A does a thing.\n"
        "(b) B does another thing.\n"
    )
    items = extract_illustrations(text)
    assert len(items) == 2
    assert items[0].startswith("(a) A does a thing")


def test_extract_illustrations_stops_at_explanation():
    text = (
        "Illustration.\n"
        "(a) A does a thing.\n"
        "Explanation.-Something else entirely.\n"
    )
    items = extract_illustrations(text)
    assert len(items) == 1
    assert "Something else entirely" not in items[0]


def test_validate_reports_index_diffs_not_errors(sections):
    import pymupdf

    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    index = parse_index(doc, cfg)
    errors, diffs = validate(sections, index, cfg)
    assert errors == []
    assert len(diffs) <= cfg["max_title_diffs"]


def test_body_wins_over_index_typo(sections):
    """Index says 'hous-ebreaking'; the body says 'house-breaking'."""
    section = next(s for s in sections if s.section_number == "330")
    assert "house-breaking" in section.section_title
