import pymupdf
import pytest

from src.config import load_act_config
from src.parse_index import parse_index


@pytest.fixture(scope="module")
def index():
    doc = pymupdf.open("data/raw/a202345.pdf")
    return parse_index(doc, load_act_config("bns"))


def test_index_has_exactly_358_entries(index):
    assert len(index) == 358


def test_index_is_contiguous_and_unique(index):
    assert set(index.keys()) == set(range(1, 359))


def test_known_titles(index):
    assert index[1] == "Short title, commencement and application"
    assert index[303] == "Theft"
    assert index[318] == "Cheating"
    assert index[358] == "Repeal and savings"


def test_wrapped_title_is_reassembled(index):
    """Section 10's title wraps across two lines in the index."""
    assert index[10].startswith("Punishment of person guilty of one of several offences")
    assert index[10].endswith("doubtful of which")


def test_index_typo_is_preserved_not_corrected(index):
    """The PDF's own index misspells section 330. The oracle reports what
    the index says; reconciliation against the body happens in ingest."""
    assert index[330] == "House-trespass and hous-ebreaking"
