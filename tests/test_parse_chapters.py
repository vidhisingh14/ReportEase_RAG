import pymupdf
import pytest

from src.config import load_act_config
from src.parse_chapters import chapter_for_page, parse_chapters
from src.pdf_text import body_start_page


@pytest.fixture(scope="module")
def chapters():
    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    return parse_chapters(doc, body_start_page(doc, cfg["body_start_marker"]))


def test_finds_20_chapters(chapters):
    assert len(chapters) == 20


def test_first_and_last_chapters(chapters):
    assert chapters[0] == ("I", "PRELIMINARY", 16)
    assert chapters[-1] == ("XX", "REPEAL AND SAVINGS", 110)


def test_property_offences_chapter(chapters):
    numerals = {c[0]: c for c in chapters}
    assert numerals["XVII"][1] == "OF OFFENCES AGAINST PROPERTY"
    assert numerals["XVII"][2] == 88


def test_chapter_for_page_assigns_theft_correctly(chapters):
    """Section 303 Theft is on page 88, in Chapter XVII."""
    number, title = chapter_for_page(chapters, 88)
    assert number == "XVII"
    assert title == "OF OFFENCES AGAINST PROPERTY"


def test_chapter_for_page_uses_the_most_recent_chapter(chapters):
    number, _ = chapter_for_page(chapters, 95)
    assert number == "XVII"
