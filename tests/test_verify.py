from src.retrieve import Retrieved
from src.verify import extract_citations, verify_citations


def _results(*numbers):
    return [
        Retrieved(
            section_id=f"bns-{n}", section_number=n, section_title="T",
            text="body", score=0.9, rank=i + 1,
        )
        for i, n in enumerate(numbers)
    ]


def test_extract_citations_finds_bracketed_sections():
    assert extract_citations("Theft is [BNS 303] and snatching is [BNS 304].") == ["303", "304"]


def test_extract_citations_deduplicates_preserving_order():
    assert extract_citations("[BNS 303] then [BNS 303] again and [BNS 101]") == ["303", "101"]


def test_extract_citations_ignores_bare_numbers():
    assert extract_citations("Section 303 says a thing about 420.") == []


def test_all_cited_sections_retrieved_is_valid():
    result = verify_citations("Theft is [BNS 303].", _results("303", "304"))
    assert result.valid == ["303"]
    assert result.fabricated == []


def test_fabricated_citation_is_detected():
    result = verify_citations("Cheating is [BNS 999].", _results("303"))
    assert result.fabricated == ["999"]
    assert result.valid == []


def test_fabricated_citation_is_stripped_from_the_text():
    result = verify_citations(
        "Theft is [BNS 303] and fraud is [BNS 999].", _results("303")
    )
    assert "[BNS 999]" not in result.cleaned_text
    assert "[BNS 303]" in result.cleaned_text


def test_answer_with_no_citations_reports_none():
    result = verify_citations("I cannot answer that.", _results("303"))
    assert result.cited == []
    assert result.fabricated == []
