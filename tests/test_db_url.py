import pytest

from src.db import resolve_database_url

# Deliberately NOT integration-marked: this needs no live database or API
# key, so it must run in the default secret-free suite. src.config calls
# load_dotenv() at import time, which may already have populated
# DATABASE_URL / TEST_DATABASE_URL from a real .env -- monkeypatch.delenv
# with raising=False clears whatever is actually there before each case.


def test_test_database_url_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://nyaya:nyaya@localhost:55432/nyaya_test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://neon-should-not-be-picked")
    assert resolve_database_url() == "postgresql://nyaya:nyaya@localhost:55432/nyaya_test"


def test_database_url_is_used_when_test_database_url_is_unset(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://neon-dev-project")
    assert resolve_database_url() == "postgresql://neon-dev-project"


def test_raises_when_neither_is_set(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        resolve_database_url()
