from src.providers.base import GenerationProvider
from src.providers.gemini import GeminiProvider

# The registry lives here, not in generate.py, so a provider's model ids never
# appear outside this package. Adding Claude means adding a module here and one
# entry below — nothing in the generation module changes.
_PROVIDERS = {
    "gemini": GeminiProvider,
}


def get_provider(name: str = "gemini") -> GenerationProvider:
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider: {name!r}. Known: {sorted(_PROVIDERS)}"
        ) from None
    return factory()


__all__ = ["GenerationProvider", "GeminiProvider", "get_provider"]
