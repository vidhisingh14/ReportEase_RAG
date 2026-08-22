import pytest

from src.db import connect
from src.retrieve import dense_search, sparse_search

# Every test here needs a live database and a populated index.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        yield connection


def test_dense_search_finds_theft(conn):
    results = dense_search(conn, "what is the punishment for theft", k=3)
    assert "bns-303" in [r.section_id for r in results]


def test_dense_search_handles_a_plain_language_situation(conn):
    """Illustrations are why this works — they read like real situations."""
    results = dense_search(conn, "he took my phone while I was asleep", k=8)
    assert "bns-303" in [r.section_id for r in results]


def test_dense_search_returns_k_ranked_results(conn):
    results = dense_search(conn, "criminal intimidation", k=5)
    assert len(results) == 5
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_sparse_search_finds_ipc_420_via_the_mapping_column(conn):
    """The design doc's headline retrieval fix, tested directly with no LLM
    in the loop. 'IPC 420' appears nowhere in the gazette text — it exists
    only in the NCRB mapping table. If this fails, the Phase 2 migration
    comparison measures nothing."""
    results = sparse_search(conn, "IPC 420", k=5)
    assert "bns-318" in [r.section_id for r in results]


def test_sparse_search_finds_ipc_378_maps_to_theft(conn):
    results = sparse_search(conn, "IPC 378", k=5)
    assert "bns-303" in [r.section_id for r in results]


def test_dense_retrieval_collides_on_section_numbers(conn):
    """Section numbers do not embed to nothing — they embed to number
    identity. Asked for 'IPC 34', dense retrieval confidently returns BNS 34;
    the correct answer is BNS 3. This is worse than returning nothing: a
    confidence threshold catches an empty result, but not a plausible wrong
    law. It is the measured justification for hybrid retrieval."""
    results = dense_search(conn, "IPC 34", k=1)
    assert results[0].section_id == "bns-34"      # the collision
    assert results[0].section_id != "bns-3"       # the correct mapping


def test_sparse_retrieval_gets_the_migration_right_where_dense_gets_it_wrong(conn):
    """The same query, through the FTS document that carries maps_to_text."""
    assert "bns-3" in [r.section_id for r in sparse_search(conn, "IPC 34", k=5)]


def test_ipc_420_is_the_documented_exception(conn):
    """'420' is Indian-English slang for a cheat, derived from IPC 420 itself,
    so the model learned the association from ordinary text. It is the one IPC
    number where dense retrieval happens to be right, and the reason the
    original spec's argument was built on a misleading example."""
    assert dense_search(conn, "IPC 420", k=1)[0].section_id == "bns-318"
