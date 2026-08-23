import pytest

from src.embed import embed_texts
from src.embed_format import DIMENSION

# Every test here spends embedding quota against a live API key.
pytestmark = pytest.mark.integration


def test_embed_texts_returns_one_vector_per_input():
    vectors = embed_texts(["theft of movable property", "criminal intimidation"])
    assert len(vectors) == 2
    assert all(len(v) == DIMENSION for v in vectors)


def test_distinct_inputs_produce_distinct_vectors():
    """The tripwire for silent aggregation. If the model returned one
    averaged vector for a batch, these would be identical."""
    a, b = embed_texts(["theft of movable property", "abetment of suicide"])
    assert a != b


def test_vectors_are_normalised():
    """gemini-embedding-2 normalises truncated outputs, which is what makes
    cosine distance valid without renormalising."""
    (vector,) = embed_texts(["theft"])
    magnitude = sum(x * x for x in vector) ** 0.5
    assert abs(magnitude - 1.0) < 0.01


def test_cache_makes_a_repeat_call_free(tmp_path, monkeypatch):
    from src import embed

    monkeypatch.setattr(embed, "CACHE_DIR", tmp_path)
    first = embed_texts(["a test sentence about theft"])
    calls_before = embed.API_CALLS
    second = embed_texts(["a test sentence about theft"])
    assert first == second
    assert embed.API_CALLS == calls_before
