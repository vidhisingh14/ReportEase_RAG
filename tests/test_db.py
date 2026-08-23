import pytest

from src.db import connect, create_schema

# Every test here needs a live database.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        create_schema(connection)
        yield connection


def test_pgvector_is_available(conn):
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None


def test_all_tables_exist(conn):
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {
        "sections", "embeddings", "queries", "retrievals",
        "answers", "eval_runs", "eval_results",
    } <= names


def test_embedding_dimension_is_768(conn):
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'embeddings'::regclass AND attname = 'vector'"
    ).fetchone()
    assert row[0] == 768


def test_non_enacted_status_is_rejected_by_the_database(conn):
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO sections (id, act, act_number, status, as_of_date,"
            " section_number, section_title, chapter_number, chapter_title,"
            " text, source_page, char_count)"
            " VALUES ('x-1','X','1 of 2023','bill','2025-10-06','1','T','I','C','body',1,4)"
        )
    conn.rollback()


def test_fts_includes_mapping_tokens(conn):
    conn.execute(
        "INSERT INTO sections (id, act, act_number, status, as_of_date,"
        " section_number, section_title, chapter_number, chapter_title,"
        " text, maps_to_text, source_page, char_count)"
        " VALUES ('test-1','TEST','1 of 2023','enacted','2025-10-06','1',"
        " 'Test','I','Chapter','some body text','IPC 420',1,14)"
        " ON CONFLICT (id) DO NOTHING"
    )
    try:
        row = conn.execute(
            "SELECT id FROM sections WHERE fts @@ plainto_tsquery('english', 'IPC 420')"
        ).fetchone()
        assert row is not None
    finally:
        # Must run even if the assertion above fails, or 'test-1' survives
        # into the module-scoped transaction and a later commit() persists
        # it, leaving 359 sections and breaking acceptance.
        conn.execute("DELETE FROM sections WHERE id = 'test-1'")
        conn.commit()


def test_queries_text_is_nullable_for_sensitive_routes(conn):
    import uuid

    qid = uuid.uuid4()
    conn.execute(
        "INSERT INTO queries (query_id, text, route) VALUES (%s, NULL, 'SENSITIVE')",
        (qid,),
    )
    row = conn.execute(
        "SELECT text, route FROM queries WHERE query_id = %s", (qid,)
    ).fetchone()
    assert row[0] is None
    assert row[1] == "SENSITIVE"
    conn.execute("DELETE FROM queries WHERE query_id = %s", (qid,))
    conn.commit()
