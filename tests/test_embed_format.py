from src.embed_format import DIMENSION, MODEL_NAME, format_document, format_query
from src.models import Section


def _section(**kwargs):
    defaults = dict(
        id="bns-303", act="BNS", act_number="45 of 2023", status="enacted",
        as_of_date="2025-10-06", section_number="303", section_title="Theft",
        chapter_number="XVII", chapter_title="OF OFFENCES AGAINST PROPERTY",
        text="Whoever, intending to take dishonestly...", source_page=88, char_count=40,
    )
    defaults.update(kwargs)
    return Section(**defaults)


def test_document_format_is_exact():
    out = format_document(_section())
    assert out == "title: Theft | text: Whoever, intending to take dishonestly..."


def test_query_format_is_exact():
    assert format_query("what is theft") == "task: question answering | query: what is theft"


def test_model_constants():
    assert MODEL_NAME == "gemini-embedding-2"
    assert DIMENSION == 768


def test_document_format_does_not_include_section_numbers():
    """Section numbers embed to number identity, not to nothing (measured:
    dense retrieval confidently returns the wrong section by number
    coincidence, e.g. "IPC 34" -> BNS 34 instead of BNS 3). That is a reason
    to keep them out of the embedding text, not a reason to think they are
    harmless here. They are matched by BM25 via the FTS column instead."""
    assert "303" not in format_document(_section())
