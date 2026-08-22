CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sections (
    id                  text PRIMARY KEY,
    act                 text NOT NULL,
    act_number          text NOT NULL,
    status              text NOT NULL CHECK (status = 'enacted'),
    as_of_date          date NOT NULL,
    section_number      text NOT NULL,
    section_title       text NOT NULL,
    chapter_number      text NOT NULL,
    chapter_title       text NOT NULL,
    text                text NOT NULL,
    illustrations       jsonb NOT NULL DEFAULT '[]'::jsonb,
    illustrations_text  text NOT NULL DEFAULT '',
    maps_to             jsonb NOT NULL DEFAULT '{}'::jsonb,
    maps_to_text        text NOT NULL DEFAULT '',
    source_page         integer NOT NULL,
    char_count          integer NOT NULL,
    fts                 tsvector GENERATED ALWAYS AS (
                            to_tsvector(
                                'english',
                                section_title || ' ' || text || ' ' ||
                                illustrations_text || ' ' || maps_to_text
                            )
                        ) STORED,
    UNIQUE (act, section_number)
);

CREATE INDEX IF NOT EXISTS sections_fts_idx ON sections USING gin (fts);

CREATE TABLE IF NOT EXISTS embeddings (
    section_id   text PRIMARY KEY REFERENCES sections(id) ON DELETE CASCADE,
    vector       vector(768) NOT NULL,
    model_name   text NOT NULL,
    dimension    integer NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- At roughly 358 chunks the index choice makes no measurable difference to
-- latency. HNSW is used because it is the sensible default, not because it
-- was tuned.
CREATE INDEX IF NOT EXISTS embeddings_vector_idx
    ON embeddings USING hnsw (vector vector_cosine_ops);

CREATE TABLE IF NOT EXISTS queries (
    query_id    uuid PRIMARY KEY,
    -- NULL for SENSITIVE-routed queries. Gemini free-tier terms permit
    -- Google to use inputs for model improvement, so question text for
    -- sensitive queries is never persisted.
    text        text,
    route       text NOT NULL,
    latency_ms  integer,
    token_cost  integer,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrievals (
    query_id    uuid REFERENCES queries(query_id) ON DELETE CASCADE,
    section_id  text REFERENCES sections(id) ON DELETE CASCADE,
    dense_rank  integer,
    sparse_rank integer,
    fused_rank  integer,
    score       double precision,
    PRIMARY KEY (query_id, section_id)
);

CREATE TABLE IF NOT EXISTS answers (
    query_id          uuid PRIMARY KEY REFERENCES queries(query_id) ON DELETE CASCADE,
    answer_text       text NOT NULL,
    cited_section_ids text[] NOT NULL DEFAULT '{}',
    prompt_version    text NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      uuid PRIMARY KEY,
    git_sha     text,
    timestamp   timestamptz NOT NULL DEFAULT now(),
    config_json jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id        uuid REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    question_id   text NOT NULL,
    passed        boolean NOT NULL,
    retrieved_ids text[] NOT NULL DEFAULT '{}',
    notes         text,
    PRIMARY KEY (run_id, question_id)
);
