import pymupdf
import pytest

from src.config import load_act_config
from src.parse_sections import merge_runs, parse_headings
from src.pdf_text import body_start_page


@pytest.fixture(scope="module")
def headings():
    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    return parse_headings(doc, cfg, body_start_page(doc, cfg["body_start_marker"]))


def test_finds_exactly_358_headings(headings):
    assert len(headings) == 358


def test_headings_are_unique_and_monotonic(headings):
    numbers = [n for n, _, _ in headings]
    assert len(set(numbers)) == 358
    assert numbers == sorted(numbers)
    assert numbers == list(range(1, 359))


def test_known_titles(headings):
    by_number = {n: t for n, t, _ in headings}
    assert by_number[303] == "Theft"
    assert by_number[318] == "Cheating"


def test_wrapped_title_is_reassembled(headings):
    """Section 10 places its em-dash on the following line."""
    by_number = {n: t for n, t, _ in headings}
    assert by_number[10].startswith("Punishment of person guilty")


def test_dash_before_title_heading_form(headings):
    """Section 255 is '255.-Title', with the dash before the title rather
    than after it. No title-then-dash pattern can match it."""
    by_number = {n: t for n, t, _ in headings}
    assert by_number[255].startswith("Public servant disobeying direction of law")


def test_page_numbers_are_plausible(headings):
    by_number = {n: p for n, _, p in headings}
    assert by_number[303] == 88
    assert by_number[1] == 16


def test_merge_runs_does_not_swallow_the_next_section():
    """A run that starts with 'N.' always begins a new heading, even when
    the previous run did not end with a dash."""
    runs = ["5. Commutation of sentence.", "6. Fractions of terms of punishment.—"]
    assert merge_runs(runs) == runs


def test_merge_runs_joins_a_wrapped_title():
    runs = ["253. Harbouring offender who has escaped from custody", "or whose apprehension has been ordered.—"]
    merged = merge_runs(runs)
    assert len(merged) == 1
    assert "apprehension has been ordered" in merged[0]


def test_cross_page_title_wrap_is_detected_not_silently_dropped(monkeypatch):
    """A title wrapping across a page break cannot be merged, because
    merge_runs restarts per page. That must raise rather than drop a section
    silently — a dropped section is a law the system will never cite."""
    import src.parse_sections as ps

    fake_pages = {
        0: ["1. Short title.—"],
        1: ["continued title text with no leading number.—"],
    }

    def fake_bold_runs(doc, pno, cfg):
        return fake_pages.get(pno, [])

    monkeypatch.setattr(ps, "bold_runs", fake_bold_runs)

    class FakeDoc:
        page_count = 2

    with pytest.raises(ValueError, match="did not parse as headings"):
        ps.parse_headings(FakeDoc(), {"heading_font_contains": "Bold",
                                      "heading_min_size": 10.0}, 0)
