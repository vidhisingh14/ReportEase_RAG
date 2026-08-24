# Nyaya

Grounded question answering over Indian criminal law. Answers come only from
the text of the Bharatiya Nyaya Sanhita 2023, and cite the sections they came
from; any citation to a section that was not retrieved is removed before the
answer is shown.

This is not legal advice.

## Setup

    python -m venv .venv
    .venv/Scripts/python.exe -m pip install -r requirements.txt
    cp .env.example .env      # then fill in DATABASE_URL and GEMINI_API_KEY

## Running the pipeline

    python -m src.ingest        # PDF -> data/processed/sections.json
    python -m src.mapping       # NCRB table -> data/processed/mappings.json
    python -m src.store         # load sections + embeddings into Postgres
    python -m src.cli "what is the punishment for theft"

## Tests

    pytest tests/                 # unit tests. No database, no API key, no cost.
    pytest tests/ -m integration  # live database and API key required
    pytest tests/ -m ""           # everything

The default run is unit-only by design. Integration tests spend Gemini free-tier
quota, and the Flash budget is 250 requests per day.

### Integration tests: container vs. Neon

Development and production both run on Neon (see design doc §1.1) -- that does
not change. But integration tests are destructive by nature (one test module
wipes and reloads the `embeddings` table), so they can also target a disposable
local Postgres + pgvector container instead of spending time and risk against
the shared Neon dev project:

    # start the disposable local database (not a dev environment -- see the
    # header comment in docker-compose.test.yml)
    docker compose -f docker-compose.test.yml up -d

    # point tests at it instead of DATABASE_URL/Neon
    export TEST_DATABASE_URL=postgresql://nyaya:nyaya@localhost:55432/nyaya_test
    psql "$TEST_DATABASE_URL" -f sql/schema.sql   # or let create_schema() do it
    pytest tests/ -m integration

    docker compose -f docker-compose.test.yml down -v   # throw it away

`TEST_DATABASE_URL`, when set, always wins over `DATABASE_URL` (see
`src/db.py:resolve_database_url`, which logs at INFO which one it picked). Leave
`TEST_DATABASE_URL` unset to run integration tests against Neon as before.

This was verified 2026-08-24 against `pgvector/pgvector:pg18` (Postgres 18.6,
pgvector 0.8.6, matching the Neon dev project's versions exactly): schema
creation, `python -m src.store` (zero embedding API calls -- served from
`.cache/embeddings/`), the full integration suite except the one live-generation
test (excluded to protect the 250/day Flash budget, not because it failed), and
`python -m scripts.acceptance` (16/16) all passed unmodified against the
container. The integration suite ran in roughly 7 seconds against the container,
versus 228s and 384s recorded against Neon for `tests/test_store.py -m
integration` alone -- the gap is network latency to Neon's hosted endpoint. See
design doc §1.1 for the recorded proof.

### CI

`.github/workflows/ci.yml` runs on every push: the default unit suite, the BNS
parser (`src.ingest`, `src.mapping`) against the committed source PDFs with
count assertions (358 sections, >=330 mappings), and the database-only
integration tests (`tests/test_db.py`) against a `pgvector/pgvector:pg18`
service container. CI has no API keys, so nothing needing embeddings or
generation runs there -- see the comment at the top of the workflow file for
exactly what is excluded and why.

## Acceptance

    python -m scripts.acceptance

## Status

Phase 1: ingest, embed, store, dense retrieval, grounded answers with verified
citations. No router, no hybrid retrieval, no evaluation harness yet — those are
Phase 2.
