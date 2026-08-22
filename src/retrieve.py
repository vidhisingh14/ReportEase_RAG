from dataclasses import dataclass

from src.embed import embed_query

DENSE_SQL = """
SELECT s.id, s.section_number, s.section_title, s.text,
       1 - (e.vector <=> %s::vector) AS score
FROM embeddings e
JOIN sections s ON s.id = e.section_id
ORDER BY e.vector <=> %s::vector
LIMIT %s
"""

SPARSE_SQL = """
SELECT s.id, s.section_number, s.section_title, s.text,
       ts_rank(s.fts, plainto_tsquery('english', %s)) AS score
FROM sections s
WHERE s.fts @@ plainto_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s
"""


@dataclass
class Retrieved:
    section_id: str
    section_number: str
    section_title: str
    text: str
    score: float
    rank: int


def _rows_to_results(rows: list) -> list:
    return [
        Retrieved(
            section_id=row[0],
            section_number=row[1],
            section_title=row[2],
            text=row[3],
            score=float(row[4]),
            rank=i + 1,
        )
        for i, row in enumerate(rows)
    ]


def dense_search(conn, question: str, k: int = 20) -> list:
    """Cosine similarity over section embeddings."""
    vector = embed_query(question)
    rows = conn.execute(DENSE_SQL, (vector, vector, k)).fetchall()
    return _rows_to_results(rows)


def sparse_search(conn, question: str, k: int = 20) -> list:
    """Postgres full-text search over title, body, illustrations, and mappings.

    The mapping tokens are what make migration queries work. 'IPC 420' does
    not appear anywhere in the gazette, so without maps_to_text in the fts
    column this returns nothing for the single most common migration query.
    """
    rows = conn.execute(SPARSE_SQL, (question, question, k)).fetchall()
    return _rows_to_results(rows)
