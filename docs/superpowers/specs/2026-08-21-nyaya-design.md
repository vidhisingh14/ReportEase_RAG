# Nyaya — design decisions and parser contract

**Date:** 2026-08-21
**Status:** awaiting review
**Relationship to `SPEC.md`:** this document does not replace SPEC.md. SPEC.md remains the statement of what is being built and why. This document records the decisions taken after SPEC.md was written, the corpus evidence gathered on 2026-08-21, and the two corrections that evidence forces on SPEC.md. Where the two documents disagree, this one wins.

---

## 1. Decisions locked

### 1.1 Storage — Neon Postgres

Development and production both run on Neon, so environments match.

- Two Neon projects: `nyaya-dev` and `nyaya-prod`. Dev code never points at prod.
- `DATABASE_URL` comes from `.env` locally and from host environment variables in production. Never hardcoded, never committed.
- All storage code is written against **plain Postgres + pgvector**, with no Neon-specific features. Nothing may block a later move to self-hosted Postgres.
- `docker-compose.yml` arrives as a **packaging deliverable in Phase 4**, not as the development environment. (SPEC.md defines Phases 1–3 and lists Docker Compose under Phase 3; "Phase 4" here means a packaging step taken after Phase 3's production-shape work, and it is explicitly not a prerequisite for anything earlier.)

`.env.example` lists every variable with blank values. `.env` is in `.gitignore` before any other code is written.

### 1.2 Models — Gemini free tier, no OpenAI anywhere

**Embeddings: `gemini-embedding-2`**

| Setting | Value | Reason |
|---|---|---|
| `output_dimensionality` | **768** | The 3072 default exceeds pgvector's 2000-dimension HNSW index cap. 768 is a Google-recommended truncation size, and truncated outputs are **auto-normalised**, so cosine distance is valid without renormalising. |
| Document format | `title: {section_title} \| text: {text}` | Title carries semantic signal the body often lacks in plain form. |
| Query format | `task: question answering \| query: {question}` | |

Both formatters live in **one module**, so index-time and query-time formats cannot drift apart.

Three hard requirements on the embedding path:

1. **Per-`Content` wrapping.** `gemini-embedding-2` returns ONE aggregated embedding when passed multiple inputs directly in `contents`. During bulk ingest each section must be wrapped in its own `types.Content` object. Passing a plain list of strings silently produces a single averaged vector — a failure that raises no error and ruins the index.
2. **Count assertion.** Number of embeddings returned must equal number of sections sent. The run fails on mismatch. This is the tripwire for (1).
3. **Model identity check.** Every row stores `model_name = "gemini-embedding-2"` and `dimension = 768`. On startup the query-time model is verified against what is in the table and fails loudly on mismatch. The `embedding-001` and `embedding-2` vector spaces are incompatible; a silent mix would degrade retrieval invisibly.

429s are handled with exponential backoff during bulk ingest. Free-tier limits are per Google Cloud project, not per key.

**Generation: Gemini free tier**

| Use | Model | max_tokens |
|---|---|---|
| Grounded answers | `gemini-2.5-flash` | 800 |
| Out-of-scope replies | `gemini-2.5-flash-lite` | 200 |
| Safety classifier | `gemini-2.5-flash-lite` | 150 |

**Swappability.** Generation sits behind a provider interface. Claude will be added later and compared against Gemini on the same 55-question eval set. Switching the generation provider must not require re-embedding and must not touch anything outside the generation module. `ANTHROPIC_API_KEY` is present in `.env.example` as a blank optional value.

`GEMINI_API_KEY` is the only key required to run.

**Prohibited:** `sentence-transformers`, `torch`, or any torch-based model. The deployment target is a 512MB free tier and torch will not fit. This rules out the local cross-encoder reranker named in SPEC.md §8; Phase 3 reranking needs a different approach, decided later.

### 1.3 Corpus scope — BNS only for Phase 1

The highest-risk unknown is parser correctness, and verifying it against one document is a clean test. Two documents with different layout conventions makes failures ambiguous.

- `ingest.py` is generic enough that adding BNSS is a **config entry, not a rewrite**. Per-act config carries the section pattern, page offsets, body-start marker, bold font name and minimum heading size.
- Full acceptance on BNS before BNSS is touched.
- The BNS→IPC mapping table is parsed from NCRB `BNS2023.pdf`, pages 20–73.
- The Phase 1 golden set is BNS-only. Procedure questions arrive with BNSS.
- Every chunk carries its `act`, so a mixed corpus works later with no schema change.

BNSS is **Phase 1.5**, once BNS ingest passes acceptance. The BNSS→CrPC mapping table is not yet acquired and is tracked separately.

### 1.4 Environment

- **Python 3.10.9** (existing venv at `.venv/`). Not migrating to 3.11; nothing in this project requires it. SPEC.md §8 is corrected accordingly.
- **`pymupdf==1.28.2` is pinned.** Span-level font name and size extraction is version sensitive and the parser depends on that behaviour. An unpinned upgrade could silently change heading detection.

---

## 2. Corpus evidence

Gathered 2026-08-21 by throwaway probe against `a202345.pdf`. Recorded here because the parser contract in §4 is derived from it, not assumed.

### 2.1 Provenance — verified enacted, not a bill

Page 16 carries `ACT NO. 45 OF 2023`, `[25th December, 2023.]`, and `BE it enacted by Parliament in the Seventy-fourth Year of the Republic of India`. The title page reads `[As on the 6th October, 2025]`.

| File | SHA256 | Pages |
|---|---|---|
| `a202345.pdf` | `ff92dcc72778944011807644b6033b1140ddbe6d7e9f82ac32fd419dae03aa86` | 112 |
| `A202346.pdf` | `54b27a4f2786dc5867c2cc23391e8359b3b29125684119acbc652d1630a716d6` | 282 |
| `BNS2023.pdf` | `a83f12a93e32c9e0b39a85f75850a448dc4682b1b44b2d6bd680165bb9931549` | 237 |

These go into `data/raw/manifest.json`. Ingest fails if a hash does not match.

### 2.2 Text layer is clean

400,023 characters extracted. Non-ASCII census: `U+2014` em-dash ×608, `U+2019` ×222, `U+201C`/`U+201D` ×128 each, `U+2013` en-dash ×18. **Zero `U+FFFD`.** No mojibake.

**Word spacing is a non-issue.** A regex for run-together words (`[a-z]{3,}[A-Z][a-z]{3,}`) returns 0 hits in both BNS and BNSS; missing-space-after-period returns 0 in both. The run-together words visible in PDF viewer screenshots are glyph-spacing artefacts of the renderer, not defects in the text layer. BM25 keyword matching over section titles is safe.

The real spacing defect is **line-break hyphenation**: 6 occurrences in BNS, **184 in BNSS**. Handled in §4.4.

### 2.3 The section-heading regex in SPEC.md does not work

SPEC.md §5 proposes `^(\d+)\.\s+(.+?)\.\s*[–—-]{2}`. Measured against the real file it matches **2 of 358 sections**. Four independent causes:

| Cause | Evidence |
|---|---|
| Separator is usually a **single** character | PyMuPDF returns one `U+2014`. Only 608 em-dashes exist in the entire 400k-character document; 358 sections each needing two would require 716 for headings alone. |
| Separator is **inconsistent** | §303 is `Theft.—`. §2 is `Definitions. –– ` — two *en*-dashes, with surrounding spaces. |
| Titles **wrap across lines** | 55 sections place the dash on the following line (§10, §25, §44, §93, …). A non-DOTALL line-anchored pattern cannot reach it. |
| A **different heading form** exists | §255 is `255.—Public servant disobeying direction of law…` — dash immediately after the number, title after the dash. There is no title-then-dash to match. |

### 2.4 Bold spans are the reliable anchor

Headings are set in `TimesNewRomanPS-BoldMT`; body text is `TimesNewRomanPSMT`; footnotes are `TimesNewRomanPSMT` at 9pt. Extracting bold runs ≥10pt, merging continuation runs, and parsing a leading `^(\d+)\.` yields:

**358 headings — 358 distinct, monotonically increasing, zero duplicates, zero missing.**

Chapter extraction from `^CHAPTER\s+([IVXL]+)$` yields **20/20 chapters**, matching SPEC.md §3.

### 2.5 The index is a validation oracle

The "Arrangement of Sections" index runs pages 3–15; the body begins page 16. The boundary marker is unambiguous — `BE it enacted` and `ACT NO. 45 OF 2023` occur only at the body start.

The index parses to **exactly 358 entries, numbered 1–358, contiguous, unique, no letter suffixes**. Cross-checking parsed body titles against it gives **352/358 exact agreement**. All six diffs are cosmetic:

| § | Nature of diff |
|---|---|
| 26, 89 | Apostrophe normalisation (`'` vs `U+2019`) |
| 44 | Trailing period and whitespace |
| 179, 180 | Line-break hyphenation in body (`bank- notes`) |
| 330 | **The PDF's own index is wrong** — `hous-ebreaking` vs the body's correct `house-breaking` |

§330 is why body text wins on mismatch.

### 2.6 Footnotes and amendment annotations

This is the consolidated "as on 6 October 2025" text, but it carries almost no amendment furniture: 41 small-font spans in the whole body, `omitted` ×0, bracketed amendment markers ×0, `substituted` ×1. There is effectively **one footnote in the entire Act** — the commencement notification on page 16.

That single footnote is nonetheless dangerous. Because it is page furniture and §2 spans that page, it lands **inside §2's body text, between Explanation 1 and Explanation 2**:

```
Explanation 1.—It is not essential to counterfeiting that the imitation should be exact.

1. 1st day of July, 2024, except the provision of sub-section (2) of section 106, vide
notification No. S.O. 850(E), dated, 23rd February, 2024, see Gazette of India, …

Explanation 2.—When a person causes one thing to resemble another thing, …
```

Two consequences:

1. Footnotes must be stripped **by font size at span level (≤9.5pt)**, never by line position. A positional rule (last N lines of a page) cannot remove text that appears mid-section.
2. That footnote's leading `1. ` would otherwise register as a phantom section 1. The bold-and-≥10pt filter already excludes it, which the 358/358 result confirms.

### 2.7 Page-spanning sections

87 of 358 sections cross at least one page break. Verified §303 (p88→89) and §318 (p95→96): text joins cleanly with no furniture intrusion once the leading page-number line is dropped.

### 2.8 Repeating furniture

The only repeating furniture is the page number — the first line of **97 of 97** body pages. There are no running headers. This is the basis for the SPEC.md correction in §3.1.

---

## 3. Corrections to SPEC.md

### 3.1 Remove the line-frequency header/footer heuristic

SPEC.md §5 currently reads:

> Strip repeating headers and footers by counting lines that appear on more than twenty pages and removing them.

**This is deleted in full.** It is not merely unnecessary here — it is actively destructive. Run against the real corpus, its top hits are:

```
  42x  'Illustrations.'
  36x  'Illustration.'
  17x  'fine.'
  10x  'also be liable to fine.'
   8x  'section.'
   6x  'and shall also be liable to fine.'
```

It would delete the illustration delimiters, which SPEC.md §4 itself calls *"the highest value text in the corpus for semantic matching"* and explicitly warns against stripping as noise. It would also chew through punishment-clause boilerplate that carries real legal meaning. The heuristic is a generic-PDF reflex that this specific corpus does not need and cannot survive.

**Replacement rule, complete:**

1. Strip the first line of a body page when it is a bare number.
2. Strip spans at or below 9.5pt.

Nothing else. No frequency counting, no positional footer rule.

### 3.2 Python version

SPEC.md §8 says Python 3.11. Corrected to **3.10** to match the existing venv. No toolchain migration.

*(Both corrections are applied to `SPEC.md` in the same commit as this document.)*

---

## 4. Parser contract

### 4.1 Primary extraction

Bold-span detection is primary. The regex is a secondary parse **of an already-identified heading run**, not a scanner over raw page text.

Algorithm, per body page:

1. Walk `page.get_text("dict")` spans. Accumulate runs where the font name contains the configured bold marker **and** size ≥ the configured minimum.
2. Normalise whitespace within each run.
3. Merge a run into its predecessor when it does **not** start with `\d+\.` and the predecessor does not already end with a dash — this reconstructs headings split across lines without swallowing the following section.
4. Parse `^(\d+)\.\s*[–—]?\s*(.*)$`; strip trailing dashes and periods from the title.

Section body is the text between one heading's start offset and the next heading's start offset, over a page-joined document with per-page furniture already removed.

### 4.2 Configuration, not hardcoding

The bold font name and minimum heading size go in **per-act config**, not in the parser. BNSS may render differently — a different font subset name, a different heading size — and that failure must be fixable by editing config, not by rewriting the parser.

Per-act config carries at minimum:

```yaml
bns:
  source_pdf: data/raw/a202345.pdf
  sha256: ff92dcc7...
  act_number: "45 of 2023"
  status: enacted
  as_of_date: "2025-10-06"
  index_pages: [3, 15]
  body_start_marker: "BE it enacted"
  heading_font_contains: "Bold"
  heading_min_size: 10.0
  footnote_max_size: 9.5
  expected_section_count: 358
  expected_chapter_count: 20
```

### 4.3 Validation

- **Hard fail on `count != 358`.** Not a warning, not a tolerance band. SPEC.md §7 allows "within 5 of 358"; the index oracle makes an exact figure available, so we hold the exact figure. Same for chapters at 20.
- **The index oracle check runs as a test**, not only inside ingest. It lives in `tests/` and therefore runs in CI on every push. A parser regression is caught by the test suite, not only by someone re-running ingest.
- **Body text wins** on any index/body mismatch — §330 proves the index itself can be wrong.
- **Every diff is logged with its section number.** Cosmetic diffs are expected and must not be silently swallowed, because a real extraction error would hide among them. The log is reviewed, not merely counted.
- Additional invariants, per SPEC.md §7: zero chunks under 50 characters, zero over 6000, section numbers contiguous 1–358 and monotonic in document order.

### 4.4 De-hyphenation

Line-break hyphens must be joined, but genuine hyphenates must survive. `House-trespass`, `house-breaking`, `currency-notes`, `bank-notes`, and `power-of-attorney` are all real legal terms that appear hyphenated in normal running text.

Rule:

1. Build a **keep-list** of genuine hyphenates harvested from the corpus itself — terms that appear hyphenated *mid-line*, where no line break could have caused it.
2. When a hyphen falls at a line break, join only if the resulting token is not on the keep-list.
3. **Log every join performed**, with the before and after token, so the keep-list can be audited rather than trusted.

Volume is small enough to review by hand: 6 joins in BNS, 184 in BNSS.

### 4.5 Illustrations

`Illustration.` and `Illustrations.` are clean delimiters, followed by lettered items `(a)`, `(b)`, … They are extracted into the `illustrations` array and **also retained in the section body text**. They are the highest-value text for semantic matching against how ordinary users phrase questions.

---

## 5. A retrieval problem SPEC.md does not address

SPEC.md §5 justifies hybrid retrieval like this:

> The string "420" embeds to nothing useful, so dense retrieval cannot find it. BM25 matches it exactly.

**BM25 over the gazette text cannot match it either.** "IPC 420" does not appear anywhere in `a202345.pdf`. The BNS gazette does not mention its IPC ancestry at all — that relationship exists only in the NCRB mapping table. So for the query *"What is IPC 420 now?"*, both dense and sparse retrieval over section text return nothing useful, and the migration category would score near zero regardless of retrieval strategy.

**Decision:** denormalise `maps_to` into the full-text search document, via a dedicated plain column.

Ingest populates a `maps_to_text` column — a flat string such as `"IPC 378 IPC 379"` — derived from the same parsed mapping data that fills `maps_to jsonb`. The generated tsvector is then built from that column:

```sql
maps_to_text text,

fts tsvector GENERATED ALWAYS AS (
  to_tsvector('english',
    section_title || ' ' || text || ' ' ||
    coalesce(illustrations_text, '') || ' ' ||
    coalesce(maps_to_text, ''))
) STORED
```

**Why a column rather than transforming `maps_to` jsonb inline in the generated expression.** A jsonb-to-text transformation embedded in a generated column populates without erroring even when it is subtly wrong — a mis-keyed path or a wrong `jsonb_array_elements_text` unnest yields an empty or malformed string, the column fills, no exception is raised, and the only visible symptom is that migration queries quietly retrieve nothing. A plain column is inspectable with `SELECT section_number, maps_to_text FROM sections LIMIT 20`, diffable against the NCRB table by eye, and testable directly. It also keeps the mapping-flattening logic in Python where it can be unit tested, rather than in a DDL expression that only runs inside Postgres.

BM25 then has a literal token to match. This keeps a single retrieval path, needs no special casing in the query router, and makes SPEC.md's stated dense-vs-sparse story actually true rather than merely plausible. The mapping tokens go into the **FTS document only**, not the embedding text, since section numbers embed to nothing useful.

This is what makes the Phase 2 migration-category result meaningful. Without it, the "sparse wins on migration" prediction in SPEC.md §7 would be measuring nothing.

---

## 6. Evaluation — two harnesses

### 6.1 The budget that forces this

Gemini free-tier limits, verified 2026-08-21:

| Model | RPM | RPD |
|---|---|---|
| `gemini-embedding-2` | 100 | 1,000 |
| `gemini-2.5-flash` | 10 | **250** |
| `gemini-2.5-flash-lite` | 15 | 1,000 |

Limits are per Google Cloud **project**, not per key. Daily quotas reset at midnight Pacific.

SPEC.md §7 asks for the golden set (55 questions) to be run three times — dense, sparse, hybrid. Done naively, with a generated answer per question per configuration, that is 165 Flash calls: **66% of the entire daily budget on one experiment**, and roughly 17 minutes of wall clock at 10 RPM. SPEC.md also asks CI to run the golden set on every push, which under that design would exhaust the quota within a few pushes and fail the build for reasons unrelated to code quality.

### 6.2 The split

The resolution is not a workaround. It is a better design that the budget happened to force: **retrieval quality and generation quality are different measurements and belong in different harnesses.**

**Harness A — retrieval eval. Zero generation calls.**

- Metrics: recall@k, MRR, per-category pass rate.
- Flow: golden question → embedding → dense / sparse / RRF-fused ranking → compare retrieved section ids against expected ids.
- Cost: 55 query embeddings on first run, cached by content hash thereafter. Effectively free, and drawn from the 1,000 RPD embedding pool, not the 250 RPD Flash pool.
- **The entire dense-vs-sparse-vs-hybrid comparison — SPEC.md's "comparison that matters" — lives here.** It is a retrieval question and never needed a generated answer to decide it.
- Runs on **every push in CI**.

**Harness B — generation eval. Disk cached.**

- Metrics: faithfulness, answer relevance, citation-verification violations, safety-route correctness.
- Cache key: `(question, prompt_version, retrieved_ids, model_name, generation_params)`.

  `model_name` is included beyond the three fields originally specified, because the stated plan is to add Claude and compare it against Gemini on this same set. Without `model_name` in the key, the cache would serve Gemini's answers when scoring Claude and the comparison would be silently meaningless.
- A run spends quota only on entries whose key changed. Editing a prompt template re-runs 55 questions (22% of the Flash budget); changing nothing costs nothing.
- Runs **on demand and nightly**, not on every push.

**Cost of one full generation eval:**

| Component | Model | Calls | Share of that model's RPD |
|---|---|---|---|
| Grounded answers | Flash | 55 | 22% |
| Safety classification | Flash-Lite | 55 | 5.5% |
| RAGAS faithfulness judge | Flash-Lite | ~110–220 | 11–22% |

The RAGAS judge runs on Flash-Lite against its separate 1,000 RPD pool, never on Flash. If it proves too expensive it runs on a sampled subset; that decision is deferred until there are real numbers.

### 6.3 CI policy

| Trigger | Runs | Generation calls |
|---|---|---|
| Every push | Unit tests, index-oracle parser test, chunk invariants, retrieval eval (all three configs) | 0 |
| Nightly / manual | Generation eval, safety suite, RAGAS | Cached; 0 if nothing changed |

The build fails if retrieval pass rate drops more than 5 points from the last recorded run. Reported **per category**, never as a single aggregate — an aggregate of 78% hides migration sitting at 30%, and that gap is where the work is.

---

## 7. Schema

Per SPEC.md §4, with privacy amendments.

```
sections     id, act, act_number, status, as_of_date, section_number, section_title,
             chapter_number, chapter_title, text, illustrations[], illustrations_text,
             maps_to jsonb, maps_to_text, source_page, char_count,
             fts tsvector GENERATED (from the plain text columns — see §5)
embeddings   section_id, vector vector(768), model_name, dimension, created_at
queries      query_id, text NULLABLE, route, latency_ms, token_cost, created_at
retrievals   query_id, section_id, dense_rank, sparse_rank, fused_rank, score
answers      query_id, answer_text, cited_section_ids, prompt_version
eval_runs    run_id, git_sha, timestamp, config_json
eval_results run_id, question_id, passed, retrieved_ids, notes
```

- HNSW index on `embeddings.vector` with `vector_cosine_ops`. At ~358 chunks the index choice makes no measurable difference to latency. That is worth saying out loud rather than presenting as a tuning decision.
- GIN index on `sections.fts`.
- `queries.text` is **nullable and left NULL for `SENSITIVE`-routed queries**. Route, timestamp and latency are stored; the question text is not. See §8.

---

## 8. Privacy

Gemini free-tier terms permit Google to use inputs for model improvement. Therefore:

- Raw question text is **not persisted for `SENSITIVE`-routed queries**. Only route, timestamp and latency are stored.
- Any landing page or UI carries a line stating that questions are processed by a third-party AI service and are not stored.
- The standing disclaimer from SPEC.md §11 — that this is not legal advice — sits in every response path, including refusals.

Note the ordering consequence: the router classifies **before** retrieval (SPEC.md §6), so the decision not to persist is available before anything is written.

---

## 9. Phase 1 acceptance criteria

Superseding the looser figures in SPEC.md §7 where the index oracle permits exactness.

- [ ] BNS section count is **exactly 358**; chapter count exactly 20
- [ ] Section numbers contiguous 1–358, unique, monotonic in document order
- [ ] Index-oracle title agreement ≥ 350/358, with every diff logged by section number and reviewed
- [ ] Zero chunks under 50 characters; zero over 6000
- [ ] 5 randomly chosen sections verified by hand against the PDF, word for word
- [ ] Ingest fails when run against a PDF whose SHA256 does not match the manifest
- [ ] Embedding count equals section count (the aggregation tripwire fires correctly)
- [ ] Stored `model_name`/`dimension` match query-time config, verified at startup
- [ ] 20 BNS→IPC mappings verified by hand against NCRB pages 20–73
- [ ] "What is the punishment for theft" returns BNS 303 in the top 3
- [ ] **FTS mapping lookup, no LLM involved.** Query the FTS index directly for a known IPC section number drawn from the mapping table and confirm the correct BNS section is returned. This isolates whether the §5 retrieval fix works from whether generation happens to compensate for it. The design doc claims this fix is what makes the Phase 2 migration comparison meaningful, so it carries its own direct test rather than resting on an end-to-end one.
- [ ] "What is IPC 420 now" returns the correct BNS section (end to end)
- [ ] Every answer includes at least one section citation
- [ ] Citation verification runs and logs violations
- [ ] De-hyphenation join log reviewed by hand

---

## 10. Repo layout

Per SPEC.md §8, plus:

```
config/acts.yaml          per-act parser config (§4.2)
config/resources.yaml     SENSITIVE support resources
data/raw/manifest.json    SHA256 + act number + retrieval date
src/providers/            generation provider interface (Gemini now, Claude later)
src/embed_format.py       single module owning both doc and query formatters (§1.2)
eval/retrieval_eval.py    Harness A — zero generation calls
eval/generation_eval.py   Harness B — disk cached
.cache/generation/        gitignored answer cache
docs/superpowers/specs/   this document
```

---

## 11. Open items

1. **BNSS→CrPC mapping table** not yet acquired. Blocks BNSS migration queries only.
2. **Phase 3 reranking** — SPEC.md names a local cross-encoder, which the no-torch constraint rules out. Alternatives (an LLM-based reranker on Flash-Lite, or dropping reranking) to be decided when Phase 3 is reached, measured against Phase 2 numbers.
3. **Deployment host** for the 512MB free tier not yet chosen.
4. **RAGAS judge cost** may force subset sampling; decide against real numbers.
