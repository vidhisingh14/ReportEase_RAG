import logging
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from src.config import require_env

log = logging.getLogger(__name__)

SCHEMA_PATH = Path("sql/schema.sql")

TABLES = [
    "eval_results", "eval_runs", "answers", "retrievals",
    "queries", "embeddings", "sections",
]


def resolve_database_url() -> str:
    """The database to connect to. TEST_DATABASE_URL wins when set.

    It exists so integration tests can run against a disposable local
    container instead of the shared Neon dev project. Tests are destructive by
    nature -- one of them wiped the embeddings table during the build -- and a
    container that can be thrown away makes that harmless. Dev and production
    both read DATABASE_URL and both point at Neon.
    """
    test_url = os.environ.get("TEST_DATABASE_URL")
    if test_url:
        log.info("resolve_database_url: using TEST_DATABASE_URL")
        return test_url
    log.info("resolve_database_url: using DATABASE_URL")
    return require_env("DATABASE_URL")


@contextmanager
def connect():
    """A connection to the database named by resolve_database_url().

    Plain Postgres and pgvector only. Nothing here may depend on a Neon
    feature, so moving to self-hosted Postgres stays a configuration change.
    """
    conn = psycopg.connect(resolve_database_url(), autocommit=False)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def create_schema(conn) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def drop_schema(conn) -> None:
    for table in TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    conn.commit()
