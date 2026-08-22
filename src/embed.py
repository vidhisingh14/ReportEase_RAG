import hashlib
import json
import logging
import random
import time
from pathlib import Path

from google import genai
from google.genai import types

from src.config import require_env
from src.embed_format import DIMENSION, MODEL_NAME, format_document, format_query

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/embeddings")
MAX_RETRIES = 6

# Test-visible counter so the cache can be proven to avoid API calls.
API_CALLS = 0

_client = None


class EmbeddingCountMismatch(Exception):
    """Raised when the API returns a different number of vectors than sent."""


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    return _client


def _cache_path(text: str) -> Path:
    key = hashlib.sha256(f"{MODEL_NAME}:{DIMENSION}:{text}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _embed_one(text: str) -> list:
    """One text, one request.

    Deliberately not batched. The Gemini documentation states that
    gemini-embedding-2 "produces a single aggregated embedding" when multiple
    inputs are passed directly in `contents`, and that failure is silent — it
    yields one averaged vector where N were expected, with no error. At 358
    sections against a 1,000/day quota the round trips are affordable and the
    correctness guarantee is not negotiable.
    """
    global API_CALLS
    client = _get_client()
    for attempt in range(MAX_RETRIES):
        try:
            API_CALLS += 1
            result = client.models.embed_content(
                model=MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSION),
            )
            if len(result.embeddings) != 1:
                raise EmbeddingCountMismatch(
                    f"sent 1 input, received {len(result.embeddings)} embeddings"
                )
            values = list(result.embeddings[0].values)
            if len(values) != DIMENSION:
                raise EmbeddingCountMismatch(
                    f"expected {DIMENSION} dimensions, received {len(values)}"
                )
            return values
        except EmbeddingCountMismatch:
            raise
        except Exception as exc:  # rate limits and transient transport errors
            if attempt == MAX_RETRIES - 1:
                raise
            delay = (2 ** attempt) + random.random()
            log.warning("embed retry %d after %.1fs: %s", attempt + 1, delay, exc)
            time.sleep(delay)


def embed_texts(texts: list) -> list:
    """Embed a list of texts, one request each, cached by content hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors = []
    for text in texts:
        path = _cache_path(text)
        if path.exists():
            vectors.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        vector = _embed_one(text)
        path.write_text(json.dumps(vector), encoding="utf-8")
        vectors.append(vector)

    if len(vectors) != len(texts):
        raise EmbeddingCountMismatch(
            f"sent {len(texts)} texts, produced {len(vectors)} vectors"
        )
    return vectors


def embed_sections(sections: list) -> dict:
    """Embed every section. Returns {section_id: vector}."""
    texts = [format_document(s) for s in sections]
    vectors = embed_texts(texts)
    if len(vectors) != len(sections):
        raise EmbeddingCountMismatch(
            f"sent {len(sections)} sections, produced {len(vectors)} vectors"
        )
    return {section.id: vector for section, vector in zip(sections, vectors)}


def embed_query(question: str) -> list:
    (vector,) = embed_texts([format_query(question)])
    return vector
