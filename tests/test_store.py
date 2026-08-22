import json

import pytest

from src.db import connect, create_schema
from src.embed_format import DIMENSION, MODEL_NAME
from src.models import Section
from src.store import load_sections, verify_embedding_model, EmbeddingModelMismatch

# Every test here needs a live database and the processed data files.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        create_schema(connection)
        yield connection


@pytest.fixture(scope="module", autouse=True)
def embeddings_count_guard(conn):
    """Fail loudly if any test in this module leaves the embeddings table
    changed. The two verify_embedding_model tests need whole-table control
    (DELETE + INSERT) to exercise verify_embedding_model, which reads across
    the whole table -- they roll that back rather than committing it, and
    this guard is the tripwire in case a future edit reintroduces a commit
    or a new destructive statement that isn't rolled back.
    """
    before = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    yield
    after = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert after == before, (
        f"tests/test_store.py left the embeddings table changed "
        f"({before} -> {after} rows) -- a destructive statement was not "
        "rolled back"
    )


@pytest.fixture(scope="module")
def loaded(conn):
    sections = [
        Section.from_dict(d)
        for d in json.load(open("data/processed/sections.json", encoding="utf-8"))
    ]
    mappings = json.load(open("data/processed/mappings.json", encoding="utf-8"))
    return load_sections(conn, sections, mappings)


def test_all_358_sections_load(conn, loaded):
    assert loaded == 358
    count = conn.execute("SELECT count(*) FROM sections WHERE act = 'BNS'").fetchone()[0]
    assert count == 358


def test_maps_to_text_is_populated_and_inspectable(conn, loaded):
    row = conn.execute(
        "SELECT maps_to_text FROM sections WHERE id = 'bns-303'"
    ).fetchone()
    assert row[0] == "IPC 378 IPC 379"


def test_maps_to_jsonb_matches_maps_to_text(conn, loaded):
    row = conn.execute(
        "SELECT maps_to, maps_to_text FROM sections WHERE id = 'bns-318'"
    ).fetchone()
    assert "420" in row[0]["sections"]
    assert "IPC 420" in row[1]


def test_reload_is_idempotent(conn, loaded):
    sections = [
        Section.from_dict(d)
        for d in json.load(open("data/processed/sections.json", encoding="utf-8"))
    ]
    mappings = json.load(open("data/processed/mappings.json", encoding="utf-8"))
    again = load_sections(conn, sections, mappings)
    assert again == 358
    count = conn.execute("SELECT count(*) FROM sections").fetchone()[0]
    assert count == 358


def test_verify_embedding_model_passes_on_matching_rows(conn):
    # verify_embedding_model reads across the whole embeddings table, so this
    # test genuinely needs whole-table control (DELETE + INSERT). It must
    # never persist that: everything below runs inside the connection's
    # already-open transaction and is rolled back in `finally`, including on
    # assertion failure, so the real embeddings survive this test either way.
    try:
        conn.execute("DELETE FROM embeddings")
        conn.execute(
            "INSERT INTO embeddings (section_id, vector, model_name, dimension)"
            " VALUES ('bns-303', %s, %s, %s)",
            ([0.0] * DIMENSION, MODEL_NAME, DIMENSION),
        )
        verify_embedding_model(conn)
    finally:
        conn.rollback()


def test_verify_embedding_model_fails_on_stale_rows(conn):
    try:
        conn.execute("DELETE FROM embeddings")
        conn.execute(
            "INSERT INTO embeddings (section_id, vector, model_name, dimension)"
            " VALUES ('bns-303', %s, %s, %s)",
            ([0.0] * DIMENSION, "gemini-embedding-001", DIMENSION),
        )
        with pytest.raises(EmbeddingModelMismatch, match="gemini-embedding-001"):
            verify_embedding_model(conn)
    finally:
        conn.rollback()
