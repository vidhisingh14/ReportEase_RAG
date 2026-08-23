from dataclasses import dataclass, field
from pathlib import Path

from src.providers import GenerationProvider, get_provider
from src.retrieve import dense_search

PROMPT_VERSION = "grounded_v1"
PROMPT_PATH = Path(f"prompts/{PROMPT_VERSION}.txt")

MAX_GROUNDED_TOKENS = 800
TOP_K = 8

DISCLAIMER = (
    "This is general information about the text of Indian criminal statutes, "
    "not legal advice. Consult a qualified lawyer about your situation."
)


@dataclass
class Answer:
    text: str
    prompt_version: str
    retrieved: list = field(default_factory=list)


def build_prompt(question: str, results: list) -> str:
    blocks = []
    for result in results:
        blocks.append(
            f"[BNS {result.section_number}] {result.section_title}\n{result.text}"
        )
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(sections="\n\n---\n\n".join(blocks), question=question)


def answer(conn, question: str, provider: GenerationProvider = None, k: int = TOP_K) -> Answer:
    provider = provider or get_provider()
    results = dense_search(conn, question, k=k)
    text = provider.generate(build_prompt(question, results), MAX_GROUNDED_TOKENS)
    return Answer(text=text, prompt_version=PROMPT_VERSION, retrieved=results)
