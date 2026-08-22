from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from src.config import require_env

SCHEMA_PATH = Path("sql/schema.sql")

TABLES = [
    "eval_results", "eval_runs", "answers", "retrievals",
    "queries", "embeddings", "sections",
]


@contextmanager
def connect():
    """A connection to the database named by DATABASE_URL.

    Plain Postgres and pgvector only. Nothing here may depend on a Neon
    feature, so moving to self-hosted Postgres stays a configuration change.
    """
    conn = psycopg.connect(require_env("DATABASE_URL"), autocommit=False)
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
