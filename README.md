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

## Acceptance

    python -m scripts.acceptance

## Status

Phase 1: ingest, embed, store, dense retrieval, grounded answers with verified
citations. No router, no hybrid retrieval, no evaluation harness yet — those are
Phase 2.
