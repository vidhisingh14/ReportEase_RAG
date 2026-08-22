from src.cli import format_output
from src.generate import Answer
from src.retrieve import Retrieved
from src.verify import verify_citations


def _answer(text, numbers):
    results = [
        Retrieved(
            section_id=f"bns-{n}", section_number=n, section_title="Theft",
            text="body", score=0.9, rank=i + 1,
        )
        for i, n in enumerate(numbers)
    ]
    return Answer(text=text, prompt_version="grounded_v1", retrieved=results)


def test_output_includes_the_disclaimer():
    answer = _answer("Theft is [BNS 303].", ["303"])
    out = format_output(answer, verify_citations(answer.text, answer.retrieved))
    assert "not legal advice" in out.lower()


def test_output_lists_the_retrieved_sources():
    answer = _answer("Theft is [BNS 303].", ["303", "304"])
    out = format_output(answer, verify_citations(answer.text, answer.retrieved))
    assert "BNS 303" in out
    assert "BNS 304" in out


def test_output_flags_fabricated_citations():
    answer = _answer("Fraud is [BNS 999].", ["303"])
    out = format_output(answer, verify_citations(answer.text, answer.retrieved))
    assert "999" in out
    assert "not retrieved" in out.lower()
