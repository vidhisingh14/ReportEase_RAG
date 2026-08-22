import json
import logging
from pathlib import Path

from src.config import load_act_config
from src.db import connect, create_schema
from src.embed import embed_sections
from src.embed_format import DIMENSION, MODEL_NAME
from src.mapping import mapping_text
from src.models import Section

log = logging.getLogger(__name__)

SECTIONS_PATH = Path("data/processed/sections.json")
MAPPINGS_PATH = Path("data/processed/mappings.json")

INSERT_SECTION = """
INSERT INTO sections (
    id, act, act_number, status, as_of_date, section_number, section_title,
    chapter_number, chapter_title, text, illustrations, illustrations_text,
    maps_to, maps_to_text, source_page, char_count
) VALUES (
    %(id)s, %(act)s, %(act_number)s, %(status)s, %(as_of_date)s,
    %(section_number)s, %(section_title)s, %(chapter_number)s,
    %(chapter_title)s, %(text)s, %(illustrations)s, %(illustrations_text)s,
    %(maps_to)s, %(maps_to_text)s, %(source_page)s, %(char_count)s
)
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    section_title = EXCLUDED.section_title,
    chapter_number = EXCLUDED.chapter_number,
    chapter_title = EXCLUDED.chapter_title,
    illustrations = EXCLUDED.illustrations,
    illustrations_text = EXCLUDED.illustrations_text,
    maps_to = EXCLUDED.maps_to,
    maps_to_text = EXCLUDED.maps_to_text,
    source_page = EXCLUDED.source_page,
    char_count = EXCLUDED.char_count
"""

INSERT_EMBEDDING = """
INSERT INTO embeddings (section_id, vector, model_name, dimension)
VALUES (%s, %s, %s, %s)
ON CONFLICT (section_id) DO UPDATE SET
    vector = EXCLUDED.vector,
    model_name = EXCLUDED.model_name,
    dimension = EXCLUDED.dimension,
    created_at = now()
"""


class EmbeddingModelMismatch(Exception):
    """Raised when stored vectors came from a different embedding model."""


def load_sections(conn, sections: list, mappings: dict, target_act: str = "IPC") -> int:
    """Insert or update every section, joining in its mapping."""
    for section in sections:
        ipc = mappings.get(section.section_number, [])
        params = section.to_dict()
        params["illustrations"] = json.dumps(section.illustrations, ensure_ascii=False)
        params["maps_to"] = json.dumps(
            {"act": target_act, "sections": ipc} if ipc else {}, ensure_ascii=False
        )
        params["maps_to_text"] = mapping_text(ipc, target_act)
        conn.execute(INSERT_SECTION, params)
    conn.commit()
    return len(sections)


def load_embeddings(conn, vectors: dict) -> int:
    for section_id, vector in vectors.items():
        conn.execute(INSERT_EMBEDDING, (section_id, vector, MODEL_NAME, DIMENSION))
    conn.commit()
    return len(vectors)


def verify_embedding_model(conn) -> None:
    """Fail loudly if stored vectors came from a different model.

    The embedding-001 and embedding-2 vector spaces are incompatible. Mixing
    them degrades retrieval invisibly, with no error and no obvious symptom,
    so this runs at startup rather than being left to chance.
    """
    rows = conn.execute(
        "SELECT DISTINCT model_name, dimension FROM embeddings"
    ).fetchall()
    for model_name, dimension in rows:
        if model_name != MODEL_NAME or dimension != DIMENSION:
            raise EmbeddingModelMismatch(
                f"stored embeddings are {model_name} at {dimension} dimensions, "
                f"but query time uses {MODEL_NAME} at {DIMENSION}. Re-run embedding."
            )


def main(act_key: str = "bns") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_act_config(act_key)  # fail early if config is missing

    sections = [
        Section.from_dict(d)
        for d in json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    ]
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))

    with connect() as conn:
        create_schema(conn)
        count = load_sections(conn, sections, mappings)
        log.info("loaded %d sections", count)

        vectors = embed_sections(sections)
        if len(vectors) != len(sections):
            raise SystemExit(
                f"embedded {len(vectors)} vectors for {len(sections)} sections"
            )
        log.info("loaded %d embeddings", load_embeddings(conn, vectors))
        verify_embedding_model(conn)


if __name__ == "__main__":
    main()
