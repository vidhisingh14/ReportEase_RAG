import pymupdf
import pytest

from src.config import load_act_config
from src.pdf_text import body_start_page, joined_body, page_text


@pytest.fixture(scope="module")
def doc():
    return pymupdf.open("data/raw/a202345.pdf")


@pytest.fixture(scope="module")
def cfg():
    return load_act_config("bns")


def test_body_starts_at_page_16(doc, cfg):
    # 0-indexed 15 == printed page 16, where the enacting formula appears
    assert body_start_page(doc, cfg["body_start_marker"]) == 15


def test_page_number_line_is_stripped(doc, cfg):
    text = page_text(doc, 15, cfg["footnote_max_size"])
    assert not text.lstrip().startswith("16")
    assert "THE BHARATIYA NYAYA SANHITA, 2023" in text


def test_footnote_is_stripped_by_font_size(doc, cfg):
    """The one footnote in the Act sits mid-section, between Explanation 1
    and Explanation 2 of section 2, because it is page-16 furniture and
    section 2 spans that page. A positional rule cannot remove it."""
    text = page_text(doc, 15, cfg["footnote_max_size"])
    assert "1st day of July, 2024" not in text
    assert "S.O. 850(E)" not in text


def test_superscript_footnote_marker_is_stripped(doc, cfg):
    text = page_text(doc, 15, cfg["footnote_max_size"])
    assert "such date as the Central Government" in " ".join(text.split())


def test_illustrations_delimiter_survives(doc, cfg):
    """Guards against reinstating the deleted line-frequency heuristic."""
    text, _, _ = joined_body(doc, cfg)
    assert text.count("Illustrations.") > 30
    assert text.count("Illustration.") > 20


def test_joined_body_offsets_align_to_pages(doc, cfg):
    text, offsets, start = joined_body(doc, cfg)
    assert start == 15
    assert offsets[0] == (0, 15)
    for offset, pno in offsets:
        assert 0 <= offset <= len(text)
    assert [p for _, p in offsets] == list(range(15, doc.page_count))
