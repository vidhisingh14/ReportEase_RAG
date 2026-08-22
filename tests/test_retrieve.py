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


def test_dense_search_cannot_find_ipc_420(conn):
    """The premise of hybrid retrieval, stated as a test rather than an
    assertion: section numbers embed to nothing useful."""
    results = dense_search(conn, "IPC 420", k=3)
    assert "bns-318" not in [r.section_id for r in results]
