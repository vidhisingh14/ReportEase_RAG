# Nyaya Phase 2a — measurement, hybrid retrieval, and a validated judge

**Date:** 2026-08-24
**Status:** awaiting review
**Relationship to earlier documents:** extends `SPEC.md` §7 and `docs/superpowers/specs/2026-08-21-nyaya-design.md` §6. Where they conflict, this document wins for Phase 2a.

---

## 1. Scope

Phase 2a delivers the golden dataset, hybrid retrieval, both evaluation harnesses, a **validated** faithfulness judge, and CI.

Phase 2a does **not** deliver the three-way router, the safety classifier, or the `OUT_OF_SCOPE` voice. SPEC.md §6 is explicit that personality ships *after* the eval set exists, because the eval set is what catches misroutes. The 15 red-team questions are written now and their retrieval measured; what is deferred is the routing behaviour they will eventually grade.

**Sequencing gate:** the golden set is drafted and reviewed by a human *before* any measurement runs. Measuring against an unreviewed set produces confident numbers about nothing.

---

## 2. RAGAS is dropped

`SPEC.md` §8 names RAGAS for evaluation. Installing it pulls **69 transitive packages**, including:

| Package | Conflict |
|---|---|
| `openai`, `langchain-openai` | Global constraint: *no OpenAI anywhere — not in code, not in comments, not in requirements.txt* |
| `langchain`, `langchain-core`, `langchain-community`, `langgraph` | SPEC.md §8: *"Deliberately no LangChain in the first build. Build the pipeline by hand so every step is visible and debuggable."* |
| `pyarrow`, `pandas`, `scipy`, `datasets` | ~200MB against a 512MB deployment target |

No `torch`, so that constraint survives. But the first two rows are direct contradictions of rules this project has enforced across all fifteen Phase 1 tasks.

**Decision:** faithfulness and answer relevance are implemented directly as versioned LLM-judge prompts. The design doc already called for *"understanding it rather than importing blindly"*; writing the judge is what that means. Zero new dependencies.

---

## 3. Golden dataset — `eval/golden.yaml`

55 questions, one schema across five categories:

```yaml
- id: mig-004
  category: migration          # lookup | situation | migration | out_of_scope | safety
  question: "What is IPC 420 now?"
  expected_sections: ["bns-318"]   # [] for out_of_scope
  expected_route: GROUNDED         # recorded now, graded in Phase 2b
  note: "Why this question exists and why this is the right answer."
```

| Category | Count | Purpose |
|---|---|---|
| `lookup` | 10 | Baseline retrieval on named sections |
| `situation` | 15 | Semantic retrieval under paraphrase |
| `migration` | 10 | Exact number matching |
| `out_of_scope` | 5 | Refusal behaviour |
| `safety` | 15 | The red-team suite |

`expected_route` is recorded but **not asserted** in Phase 2a. Writing it now means the router arrives to a set that already encodes the intent.

Every `expected_sections` entry is derived from the parsed corpus and verified to exist before the file is written. A golden set citing a non-existent section is worse than no golden set. A test enforces this.

---

## 4. Harness A — retrieval eval, zero generation calls

`eval/retrieval_eval.py`.

- **Metrics:** recall@k, MRR, per-category pass rate.
- **Configurations:** dense, sparse, hybrid — the same 55 questions through all three.
- **Cost:** 55 query embeddings on first run, cached by content hash thereafter. Drawn from the 1,000/day embedding pool. **Zero Flash calls.**
- **Runs on every push in CI.**

Reported **per category, never as a single aggregate**. An aggregate of 78% hides a category sitting at 30%, and that gap is where the engineering work is.

---

## 5. Hybrid retrieval

Reciprocal Rank Fusion in `src/retrieve.py`:

```
score(d) = Σ  1 / (k + rank_i(d))        k = 60
```

Top 20 from dense, top 20 from sparse, top 8 after fusion. The fusion function is pure — it takes two ranked lists and returns one — so it is unit-testable with no database and no API.

---

## 6. Harness B — generation eval with a validated judge

`eval/generation_eval.py`, `eval/judge.py`, `prompts/judge_faithfulness_v1.txt`.

### 6.1 One judge call per answer

The budget forbids anything else:

| Approach | Flash calls | Verdict |
|---|---|---|
| Per-claim as separate calls (~5 claims × 170) | 850 | **340% of a day — infeasible** |
| One call per answer, per-claim verdicts inside | 170 | 68% of a day |

The answer is split into claims **deterministically before judging**, and the judge returns a verdict per claim in a single structured response.

### 6.2 Claims are split deterministically

Reproducibility beats boundary quality for a regression gate. If claim boundaries drift between runs, a faithfulness delta means nothing.

**The splitter must handle legal prose.** A naive sentence splitter breaks on this corpus. Before splitting:

1. **Mask** `[BNS nnn]` citations, section references such as `2(1)(a)`, and abbreviations (`Rs.`, `No.`, `S.O.`, `vide`) so their periods do not split.
2. **Split long clauses on semicolons** where the segment exceeds ~200 characters. Statutory sentences bundle many claims.
3. **Drop fragments under ~25 characters** as non-claims rather than sending them to the judge.

The splitter is deterministic and therefore fully testable offline. It is unit-tested against real answer text containing citations and enumerations.

### 6.3 Claims are versioned and pinned

`claim_splitter_version` is part of the cache key, and the **actual claim list is stored alongside every score**.

The splitter is code that will keep being edited. Without this, a future score shift cannot be attributed to the system, the judge prompt, or the splitter.

**Full cache key:** `(question, answer, claims, claim_splitter_version, judge_prompt_version, judge_model)`.

### 6.4 Aggregation

The human labels whole answers; the judge emits per-claim verdicts. **An answer counts as faithful iff every claim is `SUPPORTED`.** That mapping is used for agreement scoring and is stated explicitly in the code.

---

## 7. Judge validation — what makes the metric real

An unvalidated metric is not a measurement.

### 7.1 The judge model is an open question, not a decision

Smaller models are materially worse judges. `gemini-2.5-flash-lite` is **not** hardcoded. The model is configuration. The validation set runs through **both** `gemini-2.5-flash-lite` and `gemini-2.5-flash`, and the choice is made on measured agreement with human labels. **Both numbers are reported.**

### 7.2 Twenty labels, split ten and ten

`eval/label_cli.py` shows the human: question → retrieved sections → answer, records a verdict plus an optional reason to `eval/human_labels.yaml`, and is resumable.

- **10 tuning** — used freely while revising the judge prompt.
- **10 held out** — touched exactly **once**, on the final prompt. The code enforces this: a run against held-out data with a non-final prompt version refuses.

### 7.3 Metrics: accuracy, precision, recall, and Cohen's κ

κ is required, not optional. If 17 of 20 answers are faithful, a judge that always answers "faithful" scores 85% accuracy while being worthless. κ near zero exposes exactly that.

### 7.4 Disagreement analysis is a deliverable

`eval/disagreements.md` records **every** case where judge and human differ: both verdicts, and the judge's stated reason. That file is the input to the next prompt revision, and it goes in the README.

### 7.5 Three prompt revisions is a budget, not a ceiling

If the judge is still misaligned after three revisions, **wait for quota reset and continue**. A judge that failed validation because the day's budget ran out is not a validated judge. Do not accept one.

---

## 8. Budget

| Step | Flash | Flash-Lite |
|---|---|---|
| Generate 20 answers to hand-label | 20 | — |
| Judge tuning: 10 × 3 revisions × 2 models | 30 | 30 |
| Judge held-out: 10 × 1 final × 2 models | 10 | 10 |
| Full generation eval: 55 grounded answers | 55 | — |
| Judge the 55 (chosen model, 1 call/answer) | 55 | — |
| **Total** | **170** (68% of 250/day) | **40** (4% of 1000/day) |

Everything is cached. Revisions beyond three spill into a second day, which §7.5 explicitly permits.

---

## 9. CI

Extends `.github/workflows/ci.yml`:

- Unit tests, parser validation, chunk invariants — as today.
- **Harness A on every push**, all three retrieval configurations, **zero generation calls**.
- Build fails if retrieval pass rate drops **more than 5 points** from the last recorded run.
- Results reported **per category**, never as one aggregate.

Harness B does not run in CI. It needs `GEMINI_API_KEY` and spends Flash quota; it runs on demand and nightly, and its cache means an unchanged run costs nothing.

---

## 10. Reporting

The README reports the faithfulness number **and the judge's agreement rate side by side**. A faithfulness score without the agreement rate that earned it is not a measurement.

Per SPEC.md §11: no evaluation result is reported as a single aggregate figure, and no README claim outruns what the acceptance criteria have verified.

---

## 11. Acceptance criteria

- [ ] 55 questions in `eval/golden.yaml`, every `expected_sections` id verified to exist in the corpus
- [ ] Golden set reviewed by a human before any measurement runs
- [ ] Claim splitter unit-tested against real answer text with citations, enumerations, and abbreviations
- [ ] Splitter is deterministic: the same answer yields the same claims across runs
- [ ] RRF fusion unit-tested as a pure function, no database
- [ ] Harness A runs all three configurations with zero generation calls
- [ ] Results reported per category, never aggregated
- [ ] 20 human labels recorded, split 10 tuning / 10 held out
- [ ] Held-out set provably touched once — enforced in code, not by discipline
- [ ] Both judge models measured; accuracy, precision, recall and Cohen's κ reported for each
- [ ] `eval/disagreements.md` written, with the judge's reason on every mismatch
- [ ] Judge model chosen on measured agreement, not assumed
- [ ] CI fails on a >5 point per-category regression, demonstrated deliberately
- [ ] README reports faithfulness and judge agreement side by side

---

## 12. Explicitly not in Phase 2a

- The three-way router, the safety classifier, and `route.py`
- The `OUT_OF_SCOPE` voice and the three per-route prompt templates
- `config/resources.yaml` support resources
- Asserting `expected_route` — recorded now, graded in Phase 2b
- Cross-encoder reranking (Phase 3, and still blocked by the no-torch constraint)
