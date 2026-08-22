# Nyaya: a grounded question answering system over Indian criminal law

**Build specification, v1**
Working name is Nyaya. Rename freely.

This document is the reference for building this project from scratch. It states what is being built, why each piece exists, what is in scope and what is deliberately not, and the acceptance criteria for each phase. Read the whole thing before writing code.

---

## 1. What this is

A retrieval augmented generation system over Indian criminal statutes. A user asks a question in plain language. The system finds the relevant sections of law, answers using only those sections, and cites them.

Two audiences, one system:

- Someone who knows the law and wants a precise citation.
- Someone who does not, who describes a situation in ordinary words and wants to know whether it is a crime and what happens next.

The system must serve the second audience without lying to them.

### The concrete problem it solves

India replaced its entire criminal code on 1 July 2024. The Indian Penal Code 1860 became the Bharatiya Nyaya Sanhita 2023. The Code of Criminal Procedure 1973 became the Bharatiya Nagarik Suraksha Sanhita 2023. Every section number changed. Someone who has used "IPC 420" for twenty years now has to find out what it is called.

So the system answers three shapes of question:

1. **Situation to section.** "Someone broke into a house at night and took jewellery. What applies?"
2. **Migration.** "What is IPC 420 now?"
3. **Section lookup.** "What does BNS 303 say?"

### Why RAG and not fine-tuning

This project exists partly because the prior approach was wrong. An earlier system baked section knowledge into model weights through fine-tuning. Two consequences followed. When the law changed the model was silently wrong, and there was no way to see which text produced any given answer.

Retrieval fixes both. The law lives in a database that can be updated in an afternoon, and every answer points at the exact text it came from. That property, traceability, is the entire product.

---

## 2. Goals and non goals

### Goals

- Every substantive claim in an answer traces to a specific section of a specific act.
- The system refuses when the corpus does not cover the question, rather than inventing a section.
- Answer quality is measured with a repeatable script, not by eyeballing it.
- Retrieval decisions are visible after the fact: what was searched, what came back, what was sent to the model.
- A non lawyer can use it and get a useful answer.

### Non goals

- This is not legal advice and must never present itself as such.
- No case law, no judgments, no precedent. Statutes only in v1.
- No state amendments. Central acts only.
- No Hindi query support in v1. English in, English out. Hindi is a v2 stretch.
- No multi user accounts, no auth, no billing.
- No fine tuning of any model.

---

## 3. Corpus

### Files to acquire

| File | Source | Pages | Purpose |
|---|---|---|---|
| `a202345.pdf` | indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf | 112 | BNS section text. Consolidated "as on 6 October 2025" |
| `A202346.pdf` | indiacode.nic.in/bitstream/123456789/20099/1/A202346.pdf | varies | BNSS section text, Act 46 of 2023 |
| `BNS2023.pdf` | ncrb.gov.in/uploads/SankalanPortal/DownloadPDF/BNS2023.pdf | 237 | BNS to IPC mapping table, **pages 20 to 73 only** |
| BNSS NCRB bundle | NCRB Sankalan portal | varies | BNSS to CrPC mapping table |

Store raw PDFs in `data/raw/`. Never modify them. Everything downstream is regenerated from these.

### Version discipline

This matters more than it sounds. Withdrawn bills for both acts circulate widely online and read almost identically to the enacted versions, with different section numbers. Indexing a bill would produce confident citations to sections that do not legally exist.

Guard against it in two ways:

1. Record a SHA256 hash of each source PDF in `data/raw/manifest.json` alongside its act number and the date it was retrieved. Fail the ingest if a hash does not match.
2. Carry `act_number`, `status` and `as_of_date` on every chunk. A chunk sourced from anything other than an enacted act must be impossible to store.

Acceptance check: an ingest run against a bill PDF should error out, not succeed quietly.

### Expected counts

- BNS: approximately 358 sections across 20 chapters.
- BNSS: 531 sections across 39 chapters.

If the parser produces materially different numbers, the parser is wrong. Do not proceed.

---

## 4. Data model

### Chunk

One chunk equals one section. This is a deliberate choice and it is the most important decision in the project. A section is a complete legal idea. Splitting a section at a token boundary produces fragments that are individually meaningless, for example a punishment clause with no definition of the offence attached.

Sections with subsections stay whole. Do not split 103(1) from 103(2).

```json
{
  "id": "bns-303",
  "act": "BNS",
  "act_number": "45 of 2023",
  "status": "enacted",
  "as_of_date": "2025-10-06",
  "section_number": "303",
  "section_title": "Theft",
  "chapter_number": "XVII",
  "chapter_title": "Of offences against property",
  "text": "full section text including subsections",
  "illustrations": ["A finds a ring belonging to Z..."],
  "maps_to": {"act": "IPC", "sections": ["378", "379"]},
  "source_page": 78,
  "char_count": 1420
}
```

Field notes:

- `section_title` is prepended to the text before embedding. The word "Theft" is a strong semantic signal that the body text often does not contain in plain form.
- `illustrations` are the "A does X to B" examples in the statute. Keep them. They read like real world situations, which is exactly how ordinary users phrase questions. They are the highest value text in the corpus for semantic matching and it would be easy to strip them as noise. Do not.
- `maps_to` powers migration queries and is populated from the NCRB table, not from the gazette PDF.
- `source_page` exists so a human can verify the parser did not fabricate anything.

### Postgres schema

```
sections        -- one row per chunk, all metadata above
embeddings      -- section_id, vector, model_name, created_at
queries         -- query_id, text, route, latency_ms, token_cost, created_at
retrievals      -- query_id, section_id, dense_rank, sparse_rank, fused_rank, score
answers         -- query_id, answer_text, cited_section_ids, prompt_version
eval_runs       -- run_id, git_sha, timestamp, config_json
eval_results    -- run_id, question_id, passed, retrieved_ids, notes
```

Keep `model_name` on embeddings. Switching embedding models later means regenerating, and without that column there is no way to tell which vectors are stale.

`prompt_version` on answers is what makes it possible to say "quality dropped when I changed the prompt" instead of guessing.

---

## 5. Architecture

```
ingest  -->  chunk  -->  embed  -->  store
                                       |
query --> route --> retrieve --> rerank --> generate --> cite
              |         |
              |         +-- dense (pgvector) + sparse (BM25) fused by RRF
              +-- three way classification, see section 6
```

### Stage by stage

**Ingest.** PyMuPDF (`fitz`) for extraction. It handles Indian government PDFs better than pypdf. Extract page by page and keep page numbers attached.

Strip page furniture with exactly two rules, and nothing else:

1. Strip the first line of a body page when it is a bare number.
2. Strip spans at or below 9.5pt.

An earlier version of this spec proposed stripping "lines that appear on more than twenty pages". Do not do this. Measured against the real corpus, that heuristic's top hits are `Illustrations.` (42 pages) and `Illustration.` (36 pages) — the delimiters for the text section 4 calls the highest value in the corpus. It would also remove recurring punishment-clause boilerplate that carries real legal meaning. The only genuinely repeating furniture in these PDFs is the page number, on the first line of every body page. See `docs/superpowers/specs/2026-08-21-nyaya-design.md` §2.8 and §3.1.

**Chunk.** Split on section boundaries using bold-span detection, with the "Arrangement of Sections" index as a validation oracle. See the parser contract in `docs/superpowers/specs/2026-08-21-nyaya-design.md` \u00a74.

**Mapping table.** A separate parser over NCRB pages 20 to 73. Table layout, not prose, so use `page.find_tables()` or column position extraction rather than the section regex. Output a lookup keyed by BNS section number. Join into the chunks after both parsers have run.

**Embed.** Text to embed is `section_title + "\n" + text + "\n" + illustrations joined`. Batch the API calls, do not loop one at a time. Cache by content hash so a re-run of an unchanged section costs nothing.

**Store.** pgvector with an HNSW index. At roughly 900 chunks the index choice makes no practical difference to latency, which is worth knowing and worth saying out loud rather than pretending it was a tuning decision.

**Retrieve.** Two searches in parallel:

- Dense: cosine similarity over embeddings, top 20.
- Sparse: BM25 over raw text, top 20. Postgres full text search is sufficient, no separate search engine needed.

Fuse with Reciprocal Rank Fusion, `score = sum(1 / (k + rank))` with k = 60. Take the top 8 after fusion.

Hybrid is not decoration here and the reason is specific to this corpus. Section numbers carry no semantic meaning. The string "420" embeds to nothing useful, so dense retrieval cannot find it. BM25 matches it exactly. Conversely "he took my phone while I was asleep" has no keyword overlap with the theft section but is semantically close. Each method fails exactly where the other works. Measure this rather than asserting it, see section 8.

**Rerank.** Cross encoder over the top 8, keep the top 4. Phase 3.

**Generate.** Send the question plus the selected sections. The prompt must instruct: answer only from the provided sections, cite section numbers inline, and if the sections do not contain the answer, say so rather than reasoning from general knowledge.

**Cite.** Post generation, verify every section number appearing in the answer exists in the retrieved set. If the model cited something that was not retrieved, that is a fabrication. Log it and strip it. This check is cheap and catches the failure mode that matters most.

---

## 6. The three way router

This is the part that distinguishes this project. Read it carefully.

A naive system has two states, answer or fail. This one has three, because the two state version has a specific and serious failure mode.

| Route | Condition | Behaviour | Tone |
|---|---|---|---|
| `GROUNDED` | Retrieval confident, question is legal | Answer from sections, cite everything | Precise, dry |
| `OUT_OF_SCOPE` | Retrieval weak, no safety signal | Refuse, explain what the corpus covers | Light, plain language, may be playful |
| `SENSITIVE` | Safety signal present, regardless of retrieval score | Full grounded answer plus support resources | Careful, direct, never playful |

### Why SENSITIVE exists

Many real questions sound like ordinary life and are serious offences.

- "He has not let me leave the flat in three days." Wrongful confinement.
- "He says he will hurt my family if I leave." Criminal intimidation.
- "He put a tracker on my phone and turns up wherever I go." Stalking.
- "He posted our private photos after we broke up." A serious offence with substantial punishment.

A system tuned to be casual about relationship questions will pattern match on words like "boyfriend" and produce a light dismissal at precisely the moment someone needed a real answer. That is the worst thing this system can do and it must be designed against from the start, not patched later.

### Implementation

Classification runs **before** retrieval, not after, because a sensitive query must not be routed by retrieval score. A serious question phrased casually may score poorly and would otherwise land in the joke path.

Order of checks:

1. Safety classifier. If it fires, route `SENSITIVE`. This check wins over everything.
2. Retrieval. If the top fused score clears the threshold, route `GROUNDED`.
3. Otherwise route `OUT_OF_SCOPE`.

The safety classifier should be an LLM call with a tightly scoped prompt, not a keyword list. Keyword lists both miss paraphrases and fire on discussions of the law itself. Someone asking "what does the law say about stalking" is asking an academic question and belongs in `GROUNDED`.

Bias the classifier toward false positives. Treating an ordinary question seriously costs a slightly stiff answer. Treating a serious question lightly costs something that cannot be undone.

### Tone separation

Three prompt template files, one per route, selected after classification. Not one prompt with conditional instructions inside it. A single prompt told to be occasionally playful will leak that register into legal answers, and citations start reading like conversation.

Store templates in `prompts/` with version numbers in the filename. Log which version produced each answer.

### Support resources

`SENSITIVE` answers append resources from a config file, not from model generation, so they cannot be hallucinated. India: 112 for emergency, 181 women helpline, 1098 childline. Keep this in `config/resources.yaml`.

### Ordering constraint

Build the router in phase 1 as a bare score threshold with no personality at all. Add the `OUT_OF_SCOPE` voice in phase 2, **after** the eval set exists. The eval set is what catches misroutes. Shipping the playful tone before the safety measurement exists is the wrong order.

---

## 7. Phases

### Phase 1: the spine

Ingest, chunk, embed, store, retrieve, answer with citations. Dense retrieval only. No personality, no reranking, no hybrid.

Deliverables:

- `ingest.py` producing `data/processed/sections.json`
- `mapping.py` producing `data/processed/mappings.json`
- Postgres with pgvector, sections and embeddings populated
- `query.py` taking a question, returning an answer with citations
- CLI is enough. No web UI yet.

**Acceptance criteria:**

- BNS section count within 5 of 358, BNSS within 5 of 531
- Zero chunks under 50 characters
- Zero chunks over 6000 characters
- 5 randomly chosen sections verified by hand against the PDF, word for word
- "What is the punishment for theft" returns BNS 303 in the top 3
- "What is IPC 420 now" returns the correct BNS section
- Every answer includes at least one section citation
- The citation verification check runs and logs violations

After phase 1 the honest claim is "I built a RAG pipeline." Nothing beyond that yet.

### Phase 2: measurement and voice

This is the phase that makes the project worth having.

**Golden dataset**, 55 questions in `eval/golden.yaml`, each with question text, expected section ids, expected route, and a note on why.

Composition:

| Category | Count | Purpose |
|---|---|---|
| Direct section lookup | 10 | Baseline retrieval |
| Situation to section | 15 | Semantic retrieval under paraphrase |
| Migration, IPC to BNS | 10 | Exact number matching, where dense fails |
| Genuinely out of scope | 5 | Refusal behaviour |
| **Looks trivial, is an offence** | **15** | **Safety routing** |

That last category is the red team suite and it is the reason this project has a story. Write questions that sound like relationship complaints, workplace annoyances or family arguments but describe actual offences. Measure how often the router sends them to `OUT_OF_SCOPE`. That count is the safety metric. Target zero.

**Hybrid retrieval.** Add BM25 alongside dense, fuse with RRF.

**The comparison that matters.** Run the golden set three times: dense only, sparse only, hybrid. Record pass rate for each, broken down by category. The expectation is that dense wins on situation questions, sparse wins on migration questions, hybrid wins overall. If the numbers say otherwise, report what they actually say. A surprising result honestly reported is worth more than a predicted one.

**Bucketed reporting.** Never report one aggregate number. Report pass rate per category. An aggregate of 78 percent hides that migration queries are at 30 percent, and that gap is where the engineering work actually is.

**Metrics to compute:** recall@k and MRR for retrieval, faithfulness and answer relevance for generation. RAGAS is worth using here, and worth understanding rather than importing blindly.

**CI.** GitHub Actions runs the golden set on every push. Fail the build if pass rate drops more than 5 points from the last recorded run.

**Then add the OUT_OF_SCOPE voice**, with the red team suite already in place to catch what it breaks.

**Acceptance criteria:**

- 55 questions, all with expected answers recorded
- Three retrieval configurations measured on the same set
- Results reported per category, not aggregated
- Safety misroute count is zero
- CI fails on regression, demonstrated by deliberately breaking chunking and watching it fail

After phase 2 the honest claim is "I built a RAG pipeline and I measured it, and here is what changed when I changed the retrieval strategy." That is the sentence worth reaching.

### Phase 3: production shape

Only if time allows. Phase 2 is a complete stopping point.

- Cross encoder reranking, measured against the phase 2 numbers so the gain is a real figure
- Per query traces: retrieved ids, scores, prompt version, latency by stage, token cost
- Conversation memory in MongoDB with session continuity
- Streaming responses
- Query reformulation and retry when retrieval is weak, which is a ReAct loop in miniature
- Docker Compose for the whole stack
- A minimal web UI

---

## 8. Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.10 | Ecosystem. Matches the existing venv; nothing here needs 3.11 |
| PDF | PyMuPDF | Handles government PDFs |
| DB | Postgres 16 with pgvector | Already known, one database instead of two |
| Sparse search | Postgres full text search | Avoids a second service |
| Embeddings | OpenAI text-embedding-3-small | 1536 dims, cheap, adequate at this scale |
| Generation | Any capable model, configurable | Do not hardcode a provider |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Runs locally, free |
| API | FastAPI | Already known |
| Eval | pytest plus RAGAS | Runs in CI |
| Memory | MongoDB | Phase 3 |
| Containers | Docker Compose | Phase 3 |

Deliberately no LangChain in the first build. Build the pipeline by hand so every step is visible and debuggable. Once it works, rebuilding it in LangGraph is a worthwhile exercise and makes it possible to speak about what the framework abstracts and when skipping it is right.

### Repo layout

```
nyaya/
  data/raw/              PDFs, manifest.json, never edited
  data/processed/        sections.json, mappings.json
  src/ingest.py
  src/mapping.py
  src/chunk.py
  src/embed.py
  src/retrieve.py        dense, sparse, RRF
  src/rerank.py
  src/route.py           three way classifier
  src/generate.py
  src/verify.py          citation verification
  src/api.py
  prompts/grounded_v1.txt
  prompts/out_of_scope_v1.txt
  prompts/sensitive_v1.txt
  prompts/safety_classifier_v1.txt
  config/resources.yaml
  eval/golden.yaml
  eval/run_eval.py
  eval/report.py
  tests/
  .github/workflows/eval.yml
```

---

## 9. What this project is designed to demonstrate

Stated plainly, because it drives the priorities above.

| Capability | Where it appears |
|---|---|
| Chunking strategy and the reasoning behind it | Section 4 |
| Embeddings and vector storage | Section 5 |
| Dense, sparse and hybrid retrieval with RRF | Section 5, measured in phase 2 |
| Reranking | Phase 3 |
| Citations and grounding | Section 5, verified programmatically |
| Refusal and hallucination mitigation | Section 6 |
| Evaluation harness with a golden set | Phase 2 |
| Regression detection in CI | Phase 2 |
| Bucketed reporting rather than a single number | Phase 2 |
| Red teaming | Phase 2, the 15 question safety suite |
| Observability and tracing | Phase 3 |
| Agent memory | Phase 3 |
| Query reformulation and retry | Phase 3 |

### What it cannot demonstrate, and should not claim to

Say these plainly rather than dressing them up:

- Index tuning at scale. At 900 chunks HNSW versus IVF is not a real decision.
- Drift detection. That needs months of live traffic.
- Real human in the loop escalation. That needs actual reviewers.
- Multi tool agent routing. One corpus and one tool does not justify it.

Naming a limit accurately is stronger than pretending it away.

---

## 10. Build order for Claude Code

1. Repo skeleton, dependencies, Postgres with pgvector running locally, manifest with hashes.
2. `ingest.py`. Do not move on until the acceptance counts in phase 1 pass.
3. `mapping.py` over NCRB pages 20 to 73. Verify 20 mappings by hand against the PDF.
4. `embed.py` with batching and content hash caching.
5. `retrieve.py`, dense only first.
6. `generate.py` with the grounded prompt, plus `verify.py`.
7. End to end CLI. Run the phase 1 acceptance checks.
8. `eval/golden.yaml` and `run_eval.py`. Get a first pass rate on dense only.
9. Sparse retrieval and RRF fusion. Rerun. Record the difference per category.
10. `route.py`, threshold only, then the safety classifier, then the three prompt templates.
11. Rerun the eval including the safety suite. Drive misroutes to zero.
12. CI workflow.

Stop after step 12 if needed. Everything past it is phase 3.

---

## 11. Standing rules

- The system never presents itself as legal advice. A disclaimer sits in every response path.
- No answer contains a section number that was not retrieved. Enforced in code, not by prompt.
- No evaluation result is reported as a single aggregate figure.
- No claim in the README that the acceptance criteria have not verified.
- Raw PDFs are never edited. Everything is regenerated from source.
