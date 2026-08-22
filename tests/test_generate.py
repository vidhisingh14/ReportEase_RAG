import pytest

from src.generate import PROMPT_VERSION, build_prompt, get_provider
from src.retrieve import Retrieved


def _result(number, title, text):
    return Retrieved(
        section_id=f"bns-{number}", section_number=number,
        section_title=title, text=text, score=0.9, rank=1,
    )


def test_prompt_includes_every_retrieved_section():
    results = [
        _result("303", "Theft", "Whoever intending to take dishonestly..."),
        _result("304", "Snatching", "Theft is snatching if..."),
    ]
    prompt = build_prompt("what is theft", results)
    assert "BNS 303" in prompt
    assert "BNS 304" in prompt
    assert "Whoever intending to take dishonestly" in prompt
    assert "what is theft" in prompt


def test_prompt_forbids_uncited_sections():
    prompt = build_prompt("q", [_result("303", "Theft", "body")])
    assert "Never cite a section that is not there" in prompt


def test_prompt_version_is_recorded():
    assert PROMPT_VERSION == "grounded_v1"


def test_gemini_provider_is_the_default():
    provider = get_provider()
    assert provider.name == "gemini-2.5-flash"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("not-a-real-provider")


def test_provider_conforms_to_the_interface():
    """The provider boundary is what makes Claude a drop-in later. Verifying
    the contract does not require a real API call, so this stays a unit test."""

    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.calls = []

        def generate(self, prompt: str, max_tokens: int) -> str:
            self.calls.append((prompt, max_tokens))
            return "Theft is [BNS 303]."

    provider = FakeProvider()
    out = provider.generate("a prompt", max_tokens=800)
    assert isinstance(out, str)
    assert provider.calls == [("a prompt", 800)]


@pytest.mark.integration
def test_gemini_provider_returns_text_live():
    provider = get_provider()
    out = provider.generate("Reply with exactly the word: acknowledged", max_tokens=20)
    assert isinstance(out, str)
    assert out.strip()


def test_generate_module_declares_no_provider_model_ids():
    """The provider boundary is the point of this module. A model id string in
    generate.py means adding Claude would require editing it, which is exactly
    what the interface exists to prevent."""
    import pathlib
    source = pathlib.Path("src/generate.py").read_text(encoding="utf-8")
    for marker in ("gemini-", "claude-", "gpt-"):
        assert marker not in source, f"provider model id {marker!r} leaked into generate.py"
