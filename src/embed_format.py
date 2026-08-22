from src.models import Section

MODEL_NAME = "gemini-embedding-2"

# 768, never the 3072 default: pgvector's HNSW index caps at 2000 dimensions.
# gemini-embedding-2 normalises truncated outputs automatically, so cosine
# distance is valid without renormalising.
DIMENSION = 768


def format_document(section: Section) -> str:
    """Index-time text for one section.

    The title is prepended because it is a strong semantic signal that the
    body text often does not contain in plain form — the word 'Theft' rarely
    appears in the section that defines theft.
    """
    return f"title: {section.section_title} | text: {section.text}"


def format_query(question: str) -> str:
    """Query-time text.

    This function and format_document live in the same module deliberately.
    If index-time and query-time formatting drift apart, retrieval degrades
    in a way that is invisible from the outside.
    """
    return f"task: question answering | query: {question}"
