from typing import Protocol


class GenerationProvider(Protocol):
    """The only surface generation code may depend on.

    Swapping providers must not require re-embedding and must not touch
    anything outside this package.
    """

    name: str

    def generate(self, prompt: str, max_tokens: int) -> str:
        ...
