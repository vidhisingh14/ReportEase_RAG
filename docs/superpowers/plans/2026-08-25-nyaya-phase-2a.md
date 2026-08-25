# Nyaya Phase 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the system — hybrid retrieval compared against dense and sparse over a reviewed 55-question golden set, plus a faithfulness judge whose agreement with human labels is itself measured.

**Architecture:** Two independent harnesses. Harness A scores retrieval with pure functions and zero generation calls, so it runs on every push. Harness B scores generated answers with an LLM judge whose claims are split deterministically and whose verdicts are cached, so an unchanged run costs nothing. Every metric is pure Python — `numpy`, `scipy` and `sklearn` are all absent from the venv and stay absent.

**Tech Stack:** Python 3.10.9, PyYAML, psycopg 3, pgvector, google-genai, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-24-nyaya-phase-2a-design.md`. Read it. It also inherits `SPEC.md` and `docs/superpowers/specs/2026-08-21-nyaya-design.md`.

## Global Constraints

- Python 3.10.9 via `.venv/Scripts/python.exe`. `pymupdf==1.28.2` pinned exactly.
- **No OpenAI anywhere** — code, comments, or `requirements.txt`. **No RAGAS, no LangChain** — the design doc §2 records why.
- No `torch`, no `sentence-transformers`. No `numpy`, `scipy` or `sklearn` — every metric is pure Python.
- Embeddings are `gemini-embedding-2` at **768 dimensions**, never 3072.
- Credentials read from `.env` at **call time, never import time**. The default `pytest tests/` run must stay secret-free and green.
- Plain Postgres + pgvector only. No Neon-specific features.
- **Model ids never appear outside `src/providers/`.** A guard test enforces this. Judge model comes from `config/eval.yaml`.
- Live-service tests are `integration`-marked. `TEST_DATABASE_URL` targets the disposable container; `DATABASE_URL` is Neon.
- **`gemini-2.5-flash` is 250 requests/day and 10/minute.** `gemini-2.5-flash-lite` is 1,000/day. Every generation and judge call is cached.
- **No evaluation result is reported as a single aggregate figure.** Per category, always.
- No README claim may outrun what the acceptance criteria have verified.

---

## Budget discipline — read before running anything

| Step | Flash | Flash-Lite |
|---|---|---|
| 20 answers to hand-label | 20 | — |
| Judge tuning: 10 × 3 revisions × 2 models | 30 | 30 |
| Judge held-out: 10 × 1 final × 2 models | 10 | 10 |
| 55 grounded answers for Harness B | 55 | — |
| Judge the 55 (chosen model, 1 call/answer) | 55 | — |
| **Total** | **170 of 250/day** | **40 of 1000/day** |

**One judge call per answer.** Per-claim as separate calls is ~850 Flash calls — 340% of a day, infeasible.

**Three prompt revisions is a budget, not a ceiling.** If the judge is still misaligned after three, wait for quota reset and continue. A judge that failed validation because the day's budget ran out is not a validated judge — do not accept one. (Design doc §7.5.)

---

## File Structure

```
config/eval.yaml            retrieval params, judge candidates, validation split
eval/__init__.py
eval/golden.py              load and validate golden.yaml
eval/metrics.py             recall@k, MRR, accuracy/precision/recall, Cohen's kappa — pure
eval/claims.py              deterministic claim splitter, versioned
eval/judge.py               faithfulness judge, one call per answer, cached
eval/label_cli.py           human labelling CLI, resumable
eval/validate_judge.py      tuning vs held-out, both models, disagreement log
eval/retrieval_eval.py      Harness A — zero generation calls
eval/generation_eval.py     Harness B — cached
eval/report.py              per-category formatting, never an aggregate
prompts/judge_faithfulness_v1.txt
eval/golden.yaml            ALREADY EXISTS, reviewed and approved — do not edit
eval/human_labels.yaml      written by label_cli
eval/disagreements.md       written by validate_judge
eval/results/               gitignored run outputs
.cache/judge/               gitignored judge cache
```

`src/retrieve.py` gains `rrf_fuse` and `hybrid_search`. Nothing else in `src/` changes except `src/providers/__init__.py`, which learns to pass a model through.

---

## Task 1: Golden set loader and its guard

**Files:**
- Create: `eval/__init__.py`, `eval/golden.py`, `tests/test_golden.py`
- Read only: `eval/golden.yaml` (reviewed and approved — **never edit it**)

**Interfaces:**
- Consumes: nothing.
- Produces: `GoldenQuestion` dataclass with fields `id, category, question, expected_sections, expected_route, note`; `load_golden(path="eval/golden.yaml") -> list[GoldenQuestion]`; `CATEGORIES` tuple; `EXPECTED_COUNTS` dict.

**Context:** `eval/golden.yaml` holds 55 reviewed questions. This task adds the loader and the test that keeps it honest. A golden set citing a section that does not exist is worse than no golden set, so the test cross-checks every id against `data/processed/sections.json`.

- [ ] **Step 1: Write the failing test**

`tests/test_golden.py`:

```python
import collections
import json

from eval.golden import CATEGORIES, EXPECTED_COUNTS, load_golden


def _corpus_ids():
    sections = json.load(open("data/processed/sections.json", encoding="utf-8"))
    return {"bns-" + s["section_number"] for s in sections}


def test_exactly_55_questions():
    assert len(load_golden()) == 55


def test_category_composition():
    counts = collections.Counter(q.category for q in load_golden())
    assert dict(counts) == EXPECTED_COUNTS


def test_ids_are_unique():
    ids = [q.id for q in load_golden()]
    assert len(set(ids)) == len(ids)


def test_every_expected_section_exists_in_the_corpus():
    """A golden set citing a section that does not exist is worse than none."""
    corpus = _corpus_ids()
    missing = [
        (q.id, sid)
        for q in load_golden()
        for sid in q.expected_sections
        if sid not in corpus
    ]
    assert missing == []


def test_out_of_scope_questions_expect_nothing():
    for q in load_golden():
        if q.category == "out_of_scope":
            assert q.expected_sections == []


def test_every_other_question_expects_at_least_one_section():
    for q in load_golden():
        if q.category != "out_of_scope":
            assert q.expected_sections, q.id


def test_categories_are_from_the_known_set():
    for q in load_golden():
        assert q.category in CATEGORIES


def test_routes_are_from_the_known_set():
    for q in load_golden():
        assert q.expected_route in {"GROUNDED", "OUT_OF_SCOPE", "SENSITIVE"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.golden'`

- [ ] **Step 3: Write `eval/__init__.py`**

Empty file.

- [ ] **Step 4: Write `eval/golden.py`**

```python
from dataclasses import dataclass
from pathlib import Path

import yaml

GOLDEN_PATH = Path("eval/golden.yaml")

CATEGORIES = ("lookup", "situation", "migration", "out_of_scope", "safety")

EXPECTED_COUNTS = {
    "lookup": 10,
    "situation": 15,
    "migration": 10,
    "out_of_scope": 5,
    "safety": 15,
}


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    category: str
    question: str
    expected_sections: tuple
    expected_route: str
    note: str

    @property
    def is_refusal(self) -> bool:
        """True when the right behaviour is to decline rather than answer."""
        return not self.expected_sections


def load_golden(path=GOLDEN_PATH) -> list:
    """The reviewed evaluation set.

    expected_route is carried on every question but asserted on none in
    Phase 2a. It is recorded now so the router in Phase 2b arrives to a set
    that already encodes the intent.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenQuestion(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            expected_sections=tuple(item["expected_sections"]),
            expected_route=item["expected_route"],
            note=item.get("note", ""),
        )
        for item in raw
    ]


def by_id(questions: list) -> dict:
    return {q.id: q for q in questions}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_golden.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/golden.py tests/test_golden.py
git commit -m "feat: golden set loader with corpus-existence guard"
```

---

## Task 2: Retrieval and agreement metrics, pure functions

**Files:**
- Create: `eval/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `recall_at_k(retrieved, expected, k) -> float`; `reciprocal_rank(retrieved, expected) -> float`; `mean_reciprocal_rank(pairs) -> float`; `AgreementScores` dataclass with `accuracy, precision, recall, kappa, n`; `binary_agreement(human, judge, positive="FAITHFUL") -> AgreementScores`; `cohens_kappa(a, b) -> float`.

**Context:** every metric is a pure function over lists — no database, no API, no `numpy`. Cohen's κ is the one that matters most and the one most easily got wrong: it corrects agreement for what chance alone would produce.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:

```python
import pytest

from eval.metrics import (
    binary_agreement,
    cohens_kappa,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5


def test_recall_at_k_respects_the_cutoff():
    """The expected section sits at rank 4, outside k=3."""
    assert recall_at_k(["x", "y", "z", "a"], ["a"], k=3) == 0.0


def test_recall_at_k_with_no_expectation_is_one():
    """out_of_scope questions expect nothing, so nothing can be missed."""
    assert recall_at_k(["a"], [], k=3) == 1.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


def test_reciprocal_rank_is_zero_when_absent():
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mean_reciprocal_rank():
    pairs = [(["a"], ["a"]), (["x", "a"], ["a"]), (["x", "y"], ["a"])]
    assert mean_reciprocal_rank(pairs) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_kappa_is_one_on_perfect_agreement():
    a = ["FAITHFUL", "UNFAITHFUL", "FAITHFUL"]
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_kappa_is_zero_when_agreement_is_only_chance():
    """The judge always says FAITHFUL. Accuracy is high, kappa is zero, and
    kappa is the number that tells the truth."""
    human = ["FAITHFUL"] * 17 + ["UNFAITHFUL"] * 3
    judge = ["FAITHFUL"] * 20
    assert cohens_kappa(human, judge) == pytest.approx(0.0)


def test_kappa_known_value():
    """Worked example: 2x2 with a=20, b=5, c=10, d=15 over n=50.
    po = 35/50 = 0.70; pe = (25/50)(30/50) + (25/50)(20/50) = 0.30+0.20 = 0.50;
    kappa = (0.70-0.50)/(1-0.50) = 0.40."""
    human = ["P"] * 25 + ["N"] * 25
    judge = ["P"] * 20 + ["N"] * 5 + ["P"] * 10 + ["N"] * 15
    assert cohens_kappa(human, judge) == pytest.approx(0.40)


def test_kappa_can_be_negative():
    """Worse than chance is a real result and must not be clamped."""
    assert cohens_kappa(["P", "N"], ["N", "P"]) < 0


def test_binary_agreement_reports_all_four_numbers():
    human = ["FAITHFUL", "FAITHFUL", "UNFAITHFUL", "UNFAITHFUL"]
    judge = ["FAITHFUL", "UNFAITHFUL", "UNFAITHFUL", "FAITHFUL"]
    s = binary_agreement(human, judge)
    assert s.n == 4
    assert s.accuracy == pytest.approx(0.5)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.kappa == pytest.approx(0.0)


def test_binary_agreement_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        binary_agreement(["FAITHFUL"], ["FAITHFUL", "FAITHFUL"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.metrics'`

- [ ] **Step 3: Write `eval/metrics.py`**

```python
from dataclasses import dataclass

# Pure Python throughout. numpy, scipy and sklearn are deliberately absent
# from this project's dependencies and must stay absent.


def recall_at_k(retrieved: list, expected: list, k: int) -> float:
    """Fraction of expected sections appearing in the top k retrieved.

    A question expecting nothing — out_of_scope — scores 1.0, because there
    is nothing to miss. Scoring it 0.0 would punish correct refusal.
    """
    if not expected:
        return 1.0
    top = set(retrieved[:k])
    return len(top & set(expected)) / len(expected)


def reciprocal_rank(retrieved: list, expected: list) -> float:
    """1/rank of the first expected section, or 0.0 if none appears."""
    wanted = set(expected)
    for rank, item in enumerate(retrieved, start=1):
        if item in wanted:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(pairs: list) -> float:
    """pairs is a list of (retrieved, expected)."""
    if not pairs:
        return 0.0
    return sum(reciprocal_rank(r, e) for r, e in pairs) / len(pairs)


def cohens_kappa(a: list, b: list) -> float:
    """Agreement between two raters, corrected for chance.

    Required because raw accuracy lies when the classes are unbalanced. If 17
    of 20 answers are faithful, a judge that always answers FAITHFUL scores
    85% accuracy while being worthless; kappa near zero exposes exactly that.

    Negative values are returned unclamped — worse-than-chance agreement is a
    real result and hiding it would defeat the purpose.
    """
    if len(a) != len(b):
        raise ValueError("cohens_kappa needs two lists of the same length")
    n = len(a)
    if n == 0:
        raise ValueError("cohens_kappa needs at least one observation")

    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)

    if expected == 1.0:
        # Both raters used exactly one label for everything. Chance already
        # explains the agreement completely, so there is no signal.
        return 0.0
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class AgreementScores:
    n: int
    accuracy: float
    precision: float
    recall: float
    kappa: float


def binary_agreement(human: list, judge: list, positive: str = "FAITHFUL") -> AgreementScores:
    """How well the judge's verdicts match the human's.

    Reported together, never singly: accuracy alone is misleading on an
    unbalanced set, and kappa alone hides which direction the judge errs in.
    """
    if len(human) != len(judge):
        raise ValueError("human and judge verdicts must be the same length")
    n = len(human)
    if n == 0:
        raise ValueError("need at least one labelled example")

    tp = sum(1 for h, j in zip(human, judge) if h == positive and j == positive)
    fp = sum(1 for h, j in zip(human, judge) if h != positive and j == positive)
    fn = sum(1 for h, j in zip(human, judge) if h == positive and j != positive)
    correct = sum(1 for h, j in zip(human, judge) if h == j)

    return AgreementScores(
        n=n,
        accuracy=correct / n,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        kappa=cohens_kappa(human, judge),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_metrics.py
git commit -m "feat: pure-python retrieval and agreement metrics including Cohen's kappa"
```

---

## Task 3: RRF fusion and hybrid retrieval

**Files:**
- Modify: `src/retrieve.py`
- Create: `tests/test_rrf.py`
- Modify: `tests/test_retrieve.py` (add hybrid integration tests)
- Create: `config/eval.yaml`

**Interfaces:**
- Consumes: `dense_search(conn, question, k)`, `sparse_search(conn, question, k)`, `Retrieved` — all existing.
- Produces: `rrf_fuse(ranked_lists, k=60, top_n=8) -> list[str]`; `hybrid_search(conn, question, dense_k=20, sparse_k=20, rrf_k=60, top_n=8) -> list[Retrieved]`; and a new module `eval/config.py` exposing `load_eval_config(path="config/eval.yaml") -> dict`. Configuration loading lives in its own module so `eval/golden.py` stays single-purpose.

**Context:** RRF is `score(d) = Σ 1/(k + rank_i(d))` with k=60. The fusion itself is a pure function over ranked id lists, so it is fully testable with no database. Ties break on section id so runs are reproducible — a regression gate needs deterministic ordering.

- [ ] **Step 1: Write the failing pure-function test**

`tests/test_rrf.py`:

```python
import pytest

from src.retrieve import rrf_fuse


def test_single_list_preserves_order():
    assert rrf_fuse([["a", "b", "c"]], k=60, top_n=3) == ["a", "b", "c"]


def test_item_ranked_well_in_both_lists_wins():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "b"]
    # a: 1/61 + 1/62, c: 1/63 + 1/61, b: 1/62 + 1/63
    assert rrf_fuse([dense, sparse], k=60, top_n=1) == ["a"]


def test_item_in_only_one_list_still_scores():
    fused = rrf_fuse([["a"], ["b"]], k=60, top_n=2)
    assert set(fused) == {"a", "b"}


def test_top_n_truncates():
    assert len(rrf_fuse([["a", "b", "c", "d"]], k=60, top_n=2)) == 2


def test_ties_break_deterministically_on_id():
    """Two items with identical scores must order the same way every run, or
    a regression gate compares runs that were never comparable."""
    first = rrf_fuse([["b", "a"], ["a", "b"]], k=60, top_n=2)
    second = rrf_fuse([["b", "a"], ["a", "b"]], k=60, top_n=2)
    assert first == second
    assert first == sorted(first)


def test_k_dampens_rank_differences():
    """A large k flattens the contribution of rank, which is the parameter's
    whole purpose."""
    sharp = rrf_fuse([["a", "b"], ["b", "a"]], k=1, top_n=2)
    flat = rrf_fuse([["a", "b"], ["b", "a"]], k=1000, top_n=2)
    assert set(sharp) == set(flat) == {"a", "b"}


def test_empty_input_returns_empty():
    assert rrf_fuse([], k=60, top_n=8) == []
    assert rrf_fuse([[], []], k=60, top_n=8) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rrf.py -v`
Expected: FAIL with `ImportError: cannot import name 'rrf_fuse'`

- [ ] **Step 3: Add `rrf_fuse` and `hybrid_search` to `src/retrieve.py`**

Append to the file, leaving `dense_search` and `sparse_search` untouched:

```python
RRF_K = 60


def rrf_fuse(ranked_lists: list, k: int = RRF_K, top_n: int = 8) -> list:
    """Reciprocal Rank Fusion over several ranked id lists.

        score(d) = sum over lists of 1 / (k + rank(d))

    Pure function over ids, so it is testable without a database. Ties break
    on the id itself: a regression gate compares runs, and runs are only
    comparable if ordering is deterministic.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))[:top_n]


def hybrid_search(
    conn,
    question: str,
    dense_k: int = 20,
    sparse_k: int = 20,
    rrf_k: int = RRF_K,
    top_n: int = 8,
) -> list:
    """Dense and sparse retrieval fused by RRF.

    Each method fails where the other works. Dense finds a paraphrased
    situation that shares no keywords with the section; sparse finds an exact
    token such as an IPC number, which lives only in the mapping column.
    """
    dense = dense_search(conn, question, k=dense_k)
    sparse = sparse_search(conn, question, k=sparse_k)
    by_id = {r.section_id: r for r in dense}
    for r in sparse:
        by_id.setdefault(r.section_id, r)

    fused_ids = rrf_fuse(
        [[r.section_id for r in dense], [r.section_id for r in sparse]],
        k=rrf_k,
        top_n=top_n,
    )
    out = []
    for rank, section_id in enumerate(fused_ids, start=1):
        source = by_id[section_id]
        out.append(
            Retrieved(
                section_id=source.section_id,
                section_number=source.section_number,
                section_title=source.section_title,
                text=source.text,
                score=source.score,
                rank=rank,
            )
        )
    return out
```

- [ ] **Step 4: Run the pure test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rrf.py -v`
Expected: 7 passed

- [ ] **Step 5: Create `config/eval.yaml`**

```yaml
# Evaluation configuration. Model ids live here rather than in code, so the
# judge model is a measured choice rather than a hardcoded assumption.

retrieval:
  dense_k: 20
  sparse_k: 20
  rrf_k: 60
  top_n: 8
  recall_at: 8

judge:
  # Both are measured against human labels during validation. `chosen` stays
  # null until that measurement exists — see the Phase 2a design doc section 7.1.
  candidates: ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
  chosen: null
  max_tokens: 1200
  prompt_version: "judge_faithfulness_v1"

validation:
  # 20 golden questions, spanning categories, split 10/10. The held-out ten
  # are touched exactly once, on the final prompt.
  tuning_ids:
    ["look-001", "look-002", "sit-001", "sit-003", "sit-009",
     "mig-001", "mig-003", "oos-001", "safe-001", "safe-003"]
  holdout_ids:
    ["look-005", "look-007", "sit-005", "sit-012", "sit-015",
     "mig-006", "oos-004", "safe-005", "safe-008", "safe-013"]
  # A budget, not a ceiling. If the judge is still misaligned after three
  # revisions, wait for quota reset and continue. Design doc section 7.5.
  max_prompt_revisions: 3
```

- [ ] **Step 6: Create `eval/config.py`**

```python
from pathlib import Path

import yaml

EVAL_CONFIG_PATH = Path("config/eval.yaml")


def load_eval_config(path=EVAL_CONFIG_PATH) -> dict:
    """Retrieval parameters, judge candidates, and the validation split.

    Judge model ids live here, not in code, so the choice between them is
    made on measured agreement rather than assumed.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 7: Add hybrid integration tests to `tests/test_retrieve.py`**

Append (the file already has `pytestmark = pytest.mark.integration` and a `conn` fixture):

```python
def test_hybrid_finds_what_dense_alone_finds(conn):
    from src.retrieve import hybrid_search

    results = hybrid_search(conn, "he took my phone while I was asleep", top_n=8)
    assert "bns-303" in [r.section_id for r in results]


def test_hybrid_finds_what_sparse_alone_finds(conn):
    """The mapping tokens live only in the FTS document, so this is the case
    dense retrieval cannot reach."""
    from src.retrieve import hybrid_search

    results = hybrid_search(conn, "IPC 420", top_n=8)
    assert "bns-318" in [r.section_id for r in results]


def test_hybrid_ranks_are_sequential(conn):
    from src.retrieve import hybrid_search

    results = hybrid_search(conn, "criminal intimidation", top_n=5)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
```

- [ ] **Step 8: Run the integration tests against the container**

```bash
export TEST_DATABASE_URL="postgresql://nyaya:nyaya@localhost:55432/nyaya_test"
.venv/Scripts/python.exe -m pytest tests/test_retrieve.py -v -m integration
```

Expected: 11 passed (8 existing + 3 new). If the container is not running:
`docker compose -f docker-compose.test.yml up -d` — and note `docker` needs
`C:\Program Files\Docker\Docker\resources\bin` on PATH.

- [ ] **Step 9: Commit**

```bash
git add src/retrieve.py tests/test_rrf.py tests/test_retrieve.py config/eval.yaml eval/config.py
git commit -m "feat: RRF fusion and hybrid retrieval"
```

---

## Task 4: Harness A — retrieval eval with zero generation calls

**Files:**
- Create: `eval/report.py`, `eval/retrieval_eval.py`, `tests/test_report.py`
- Create: `.gitignore` entries for `eval/results/`

**Interfaces:**
- Consumes: `load_golden`, `load_eval_config`, `recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `dense_search`, `sparse_search`, `hybrid_search`.
- Produces: `CategoryScore` dataclass with `category, n, recall, mrr, pass_rate`; `evaluate_config(conn, questions, search_fn, cfg) -> dict[str, CategoryScore]`; `format_report(results_by_config) -> str`; `main()` writing `eval/results/retrieval-<timestamp>.json`.

**Context:** this is the harness that runs on every push. It makes **zero generation calls** — only query embeddings, which are cached by content hash and drawn from the 1,000/day embedding pool. A question passes when every expected section appears in the top `recall_at`.

- [ ] **Step 1: Write the failing test for reporting**

`tests/test_report.py`:

```python
import pytest

from eval.report import CategoryScore, format_report


def _scores():
    return {
        "dense": {
            "lookup": CategoryScore("lookup", 10, 0.9, 0.85, 0.9),
            "migration": CategoryScore("migration", 10, 0.1, 0.05, 0.1),
        },
        "sparse": {
            "lookup": CategoryScore("lookup", 10, 0.6, 0.55, 0.6),
            "migration": CategoryScore("migration", 10, 1.0, 1.0, 1.0),
        },
    }


def test_report_lists_every_category_separately():
    out = format_report(_scores())
    assert "lookup" in out
    assert "migration" in out


def test_report_shows_every_configuration():
    out = format_report(_scores())
    assert "dense" in out
    assert "sparse" in out


def test_report_does_not_print_a_single_aggregate_number():
    """SPEC.md section 11: no evaluation result is reported as a single
    aggregate figure. An aggregate hides the category that is failing."""
    out = format_report(_scores()).lower()
    assert "overall" not in out
    assert "aggregate" not in out


def test_report_is_stable_across_calls():
    assert format_report(_scores()) == format_report(_scores())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.report'`

- [ ] **Step 3: Write `eval/report.py`**

```python
from dataclasses import dataclass

from eval.golden import CATEGORIES


@dataclass(frozen=True)
class CategoryScore:
    category: str
    n: int
    recall: float
    mrr: float
    pass_rate: float


def format_report(results_by_config: dict) -> str:
    """A per-category table, one block per retrieval configuration.

    Deliberately prints no aggregate. An aggregate of 78 percent hides a
    category sitting at 30 percent, and that gap is where the engineering
    work actually is.
    """
    lines = []
    for config_name in sorted(results_by_config):
        scores = results_by_config[config_name]
        lines.append(f"## {config_name}")
        lines.append("")
        lines.append(f"| {'category':<14}| {'n':>3} | {'recall':>7} | {'MRR':>7} | {'pass':>6} |")
        lines.append(f"|{'-' * 15}|{'-' * 5}|{'-' * 9}|{'-' * 9}|{'-' * 8}|")
        for category in CATEGORIES:
            s = scores.get(category)
            if s is None:
                continue
            lines.append(
                f"| {s.category:<14}| {s.n:>3} | {s.recall:>7.3f} | "
                f"{s.mrr:>7.3f} | {s.pass_rate:>6.3f} |"
            )
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -v`
Expected: 4 passed

- [ ] **Step 5: Write `eval/retrieval_eval.py`**

```python
import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

from eval.config import load_eval_config
from eval.golden import CATEGORIES, load_golden
from eval.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from eval.report import CategoryScore, format_report
from src.db import connect
from src.retrieve import dense_search, hybrid_search, sparse_search

log = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results")


def _search_functions(cfg: dict) -> dict:
    r = cfg["retrieval"]
    return {
        "dense": lambda conn, q: dense_search(conn, q, k=r["top_n"]),
        "sparse": lambda conn, q: sparse_search(conn, q, k=r["top_n"]),
        "hybrid": lambda conn, q: hybrid_search(
            conn, q,
            dense_k=r["dense_k"], sparse_k=r["sparse_k"],
            rrf_k=r["rrf_k"], top_n=r["top_n"],
        ),
    }


def evaluate_config(conn, questions: list, search_fn, cfg: dict) -> dict:
    """Score one retrieval configuration over the golden set, per category.

    Makes no generation calls. Query embeddings are cached by content hash,
    so a re-run costs nothing.
    """
    at_k = cfg["retrieval"]["recall_at"]
    buckets = collections.defaultdict(list)

    for q in questions:
        retrieved = [r.section_id for r in search_fn(conn, q.question)]
        buckets[q.category].append((q, retrieved))

    scores = {}
    for category in CATEGORIES:
        rows = buckets.get(category, [])
        if not rows:
            continue
        recalls = [recall_at_k(r, list(q.expected_sections), at_k) for q, r in rows]
        rr = [(r, list(q.expected_sections)) for q, r in rows]
        passed = [1.0 if rc == 1.0 else 0.0 for rc in recalls]
        scores[category] = CategoryScore(
            category=category,
            n=len(rows),
            recall=sum(recalls) / len(recalls),
            mrr=mean_reciprocal_rank(rr),
            pass_rate=sum(passed) / len(passed),
        )
    return scores


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Harness A: retrieval eval, no generation calls.")
    parser.add_argument("--configs", default="dense,sparse,hybrid")
    args = parser.parse_args()

    cfg = load_eval_config()
    questions = load_golden()
    functions = _search_functions(cfg)
    wanted = [c.strip() for c in args.configs.split(",")]

    results = {}
    with connect() as conn:
        for name in wanted:
            if name not in functions:
                raise SystemExit(f"unknown config {name!r}. Known: {sorted(functions)}")
            log.info("evaluating %s over %d questions", name, len(questions))
            results[name] = evaluate_config(conn, questions, functions[name], cfg)

    print(format_report(results))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        name: {c: vars(s) for c, s in scores.items()}
        for name, scores in results.items()
    }
    out = RESULTS_DIR / f"retrieval-{stamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**A limitation to record, not paper over.** `recall_at_k` returns 1.0 for a question expecting nothing, so the five `out_of_scope` questions always score a perfect pass rate in Harness A. That is correct — there is nothing to retrieve and nothing to miss — but it means the `out_of_scope` row measures nothing about refusal quality. Refusal is a *routing* behaviour, and it is graded in Phase 2b when the router exists. State this in the README rather than letting a row of 1.000 read as a result.

- [ ] **Step 6: Add `eval/results/` to `.gitignore`**

Append the line `eval/results/` to `.gitignore`.

- [ ] **Step 7: Run Harness A against the container**

```bash
export TEST_DATABASE_URL="postgresql://nyaya:nyaya@localhost:55432/nyaya_test"
.venv/Scripts/python.exe -m eval.retrieval_eval
```

Expected: three per-category tables, no aggregate line, and a JSON file in `eval/results/`. This makes **zero generation calls** — the only API traffic is query embeddings, cached after the first run.

**Report the actual numbers in your report.** Do not predict them beforehand and do not describe any result as expected or surprising — just record what the run produced.

- [ ] **Step 8: Commit**

```bash
git add eval/report.py eval/retrieval_eval.py tests/test_report.py .gitignore
git commit -m "feat: Harness A, per-category retrieval eval with zero generation calls"
```

---

## Task 5: Deterministic claim splitter

**Files:**
- Create: `eval/claims.py`, `tests/test_claims.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CLAIM_SPLITTER_VERSION` string; `split_claims(text) -> list[str]`; `MIN_CLAIM_CHARS`, `LONG_SEGMENT_CHARS` constants.

**Context:** read design doc §6.2 before implementing. A naive sentence splitter breaks on this corpus: citations like `[BNS 303(2)]`, section references like `2(1)(a)`, and abbreviations like `Rs.` and `S.O.` all contain periods that must not split. Statutory sentences also bundle several claims behind semicolons.

The splitter is deterministic and therefore fully testable offline, with no API and no database.

- [ ] **Step 1: Write the failing test**

`tests/test_claims.py`:

```python
from eval.claims import CLAIM_SPLITTER_VERSION, split_claims


def test_simple_sentences_split():
    claims = split_claims(
        "Theft is punishable with imprisonment up to three years. "
        "A second conviction carries a longer term of imprisonment."
    )
    assert len(claims) == 2


def test_citation_periods_do_not_split():
    """[BNS 303(2)] contains a period inside parentheses in some renderings
    and must never be treated as a sentence boundary."""
    claims = split_claims(
        "The punishment for theft is imprisonment up to three years [BNS 303(2)]."
    )
    assert len(claims) == 1
    assert "[BNS 303(2)]" in claims[0]


def test_section_reference_does_not_split():
    claims = split_claims(
        "The definition in 2(1)(a) applies to this offence and to no other offence."
    )
    assert len(claims) == 1


def test_abbreviations_do_not_split():
    claims = split_claims(
        "A fine of Rs. 5000 was imposed under notification No. 12 of that year."
    )
    assert len(claims) == 1


def test_long_semicolon_clause_is_split():
    """Statutory sentences bundle several claims. A segment over the length
    threshold is split on semicolons so each claim is judged separately."""
    long_clause = (
        "The offender shall be punished with imprisonment which may extend to three years; "
        "he shall also be liable to a fine of an amount determined by the court; "
        "and in the case of a second conviction the term shall not be less than one year."
    )
    assert len(split_claims(long_clause)) == 3


def test_short_semicolon_clause_is_not_split():
    assert len(split_claims("He was fined; he appealed the order of the court.")) == 1


def test_fragments_below_the_minimum_are_dropped():
    """Sub-25-character fragments are not claims and must not be sent to the
    judge, where they would produce meaningless verdicts."""
    claims = split_claims("Yes. Theft is punishable with imprisonment up to three years.")
    assert all(len(c) >= 25 for c in claims)
    assert len(claims) == 1


def test_bullet_lines_become_separate_claims():
    text = (
        "Several offences apply here:\n"
        "*   Snatching carries imprisonment which may extend to three years.\n"
        "*   Theft in a dwelling house carries a term of up to seven years.\n"
    )
    assert len(split_claims(text)) >= 2


def test_splitter_is_deterministic():
    text = (
        "The punishment for theft is imprisonment up to three years [BNS 303(2)]. "
        "A fine of Rs. 5000 may also be imposed on the offender by the court."
    )
    assert split_claims(text) == split_claims(text)


def test_empty_text_yields_no_claims():
    assert split_claims("") == []
    assert split_claims("   \n  ") == []


def test_version_is_exposed():
    assert isinstance(CLAIM_SPLITTER_VERSION, str)
    assert CLAIM_SPLITTER_VERSION


def test_real_answer_text_splits_sensibly():
    """Verbatim from a real system answer, including markdown and citations."""
    text = (
        "The punishment for theft is imprisonment of either description for a term "
        "which may extend to three years, or with fine, or with both [BNS 303(2)]. "
        "In the case of a second conviction, the punishment is rigorous imprisonment "
        "for a term not less than one year [BNS 303(2)].\n"
        "*   **Snatching:** Imprisonment which may extend to three years [BNS 304(2)].\n"
    )
    claims = split_claims(text)
    assert len(claims) >= 3
    assert all(len(c) >= 25 for c in claims)
    assert any("303(2)" in c for c in claims)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_claims.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.claims'`

- [ ] **Step 3: Write `eval/claims.py`**

```python
import re

# Bump this whenever the splitting behaviour changes. It is part of the judge
# cache key and is stored beside every score, so a future shift in the
# faithfulness number can be attributed to the system, the judge prompt, or
# the splitter — rather than being an unexplained change.
CLAIM_SPLITTER_VERSION = "claims_v1"

MIN_CLAIM_CHARS = 25
LONG_SEGMENT_CHARS = 200

# Anything whose internal periods must not be read as sentence boundaries.
_CITATION = re.compile(r"\[[^\]]*\]")
_SECTION_REF = re.compile(r"\b\d+[A-Za-z]?(?:\s*\([0-9A-Za-z]+\))+")
_ABBREV = re.compile(r"\b(?:Rs|No|S\.O|Art|Cl|Sec|Ss|i\.e|e\.g|etc|vide)\.", re.IGNORECASE)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")


def _mask(text: str):
    """Replace period-bearing spans with placeholders before splitting."""
    store = []

    def capture(match):
        store.append(match.group(0))
        return f"\x00{len(store) - 1}\x00"

    for pattern in (_CITATION, _SECTION_REF, _ABBREV):
        text = pattern.sub(capture, text)
    return text, store


def _unmask(text: str, store: list) -> str:
    return _PLACEHOLDER.sub(lambda m: store[int(m.group(1))], text)


def split_claims(text: str) -> list:
    """Split an answer into individually checkable claims.

    Deterministic by design. Reproducibility beats boundary quality here: a
    regression gate compares faithfulness across runs, and if claim
    boundaries drift between runs the comparison is meaningless.
    """
    if not text or not text.strip():
        return []

    masked, store = _mask(text)

    segments = []
    for line in masked.split("\n"):
        line = line.strip()
        if not line:
            continue
        segments.extend(s for s in _SENTENCE_END.split(line) if s.strip())

    expanded = []
    for segment in segments:
        segment = segment.strip()
        if len(segment) > LONG_SEGMENT_CHARS and ";" in segment:
            # Statutory prose bundles several claims behind semicolons.
            expanded.extend(part.strip() for part in segment.split(";") if part.strip())
        else:
            expanded.append(segment)

    claims = []
    for segment in expanded:
        claim = _unmask(segment, store).strip()
        # Strip leading markdown bullet markers so the claim reads as prose.
        claim = re.sub(r"^[*\-+]\s+", "", claim).strip()
        if len(claim) >= MIN_CLAIM_CHARS:
            claims.append(claim)
    return claims
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_claims.py -v`
Expected: 12 passed

If `test_long_semicolon_clause_is_split` fails on the count, check the segment length against `LONG_SEGMENT_CHARS` before adjusting anything — report the measured length rather than tuning the threshold to fit the test.

- [ ] **Step 5: Commit**

```bash
git add eval/claims.py tests/test_claims.py
git commit -m "feat: deterministic claim splitter for legal prose, versioned"
```

---

## Task 6: The faithfulness judge

**Files:**
- Create: `prompts/judge_faithfulness_v1.txt`, `eval/judge.py`, `tests/test_judge.py`
- Modify: `src/providers/__init__.py` (let `get_provider` accept a model)

**Interfaces:**
- Consumes: `split_claims`, `CLAIM_SPLITTER_VERSION`, `load_eval_config`, `get_provider`.
- Produces: `ClaimVerdict` dataclass with `index, claim, verdict, reason`; `JudgeResult` dataclass with `verdicts, answer_verdict, model, prompt_version, splitter_version, claims`; `judge_answer(question, answer, sections, model, prompt_version=None) -> JudgeResult`; `build_judge_prompt(question, sections, claims) -> str`; `parse_judge_response(text, claims) -> list[ClaimVerdict]`.

**Context:** one judge call per answer — the budget forbids anything else. Claims are split before the call and the judge returns a verdict for each, in one structured JSON response. An answer counts faithful **iff every claim is `SUPPORTED`**; that mapping is what agreement scoring compares against human answer-level labels.

Parsing fails **closed**: an unparseable judge response marks the answer `UNFAITHFUL` and logs loudly, rather than silently passing.

- [ ] **Step 1: Let `get_provider` accept a model**

Modify `src/providers/__init__.py`:

```python
def get_provider(name: str = "gemini", model: str = None) -> GenerationProvider:
    """Construct a provider. `model` comes from configuration, never from
    code outside this package, so model ids stay inside the boundary."""
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider: {name!r}. Known: {sorted(_PROVIDERS)}"
        ) from None
    return factory(model) if model else factory()
```

- [ ] **Step 2: Write `prompts/judge_faithfulness_v1.txt`**

```
You are checking whether each claim in an answer is supported by the statutory sections that were provided to the system that wrote it.

You are NOT judging whether the claim is good law, well written, or complete. You are judging one thing only: is this claim supported by the text below?

VERDICTS:
- SUPPORTED    the claim is stated in, or follows directly from, the provided sections
- UNSUPPORTED  the claim is not in the provided sections, or contradicts them
- NOT_A_CLAIM  the segment asserts nothing checkable (a heading, a transition, a disclaimer)

PROVIDED SECTIONS:
{sections}

QUESTION THE ANSWER RESPONDED TO:
{question}

CLAIMS TO JUDGE:
{claims}

Reply with JSON and nothing else. One object per claim, in the same order, using the same indices:

{{"verdicts": [{{"index": 0, "verdict": "SUPPORTED", "reason": "one short sentence"}}]}}

Give a reason for every claim, especially the SUPPORTED ones. The reason is read by a human comparing your verdict to theirs.
```

- [ ] **Step 3: Write the failing test**

`tests/test_judge.py`:

```python
import pytest

from eval.judge import (
    ClaimVerdict,
    answer_verdict_from,
    build_judge_prompt,
    parse_judge_response,
)


def test_prompt_includes_every_claim_and_section():
    prompt = build_judge_prompt(
        question="what is the punishment for theft",
        sections=[("bns-303", "Theft", "Whoever intending to take dishonestly...")],
        claims=["Theft carries up to three years.", "A fine may also be imposed."],
    )
    assert "bns-303" in prompt or "BNS 303" in prompt
    assert "Theft carries up to three years." in prompt
    assert "A fine may also be imposed." in prompt
    assert "what is the punishment for theft" in prompt


def test_prompt_numbers_claims_so_indices_line_up():
    prompt = build_judge_prompt("q", [("bns-1", "T", "body")], ["first claim", "second claim"])
    assert "0" in prompt and "1" in prompt


def test_parse_reads_verdicts_in_order():
    raw = '{"verdicts": [{"index": 0, "verdict": "SUPPORTED", "reason": "stated"}, {"index": 1, "verdict": "UNSUPPORTED", "reason": "absent"}]}'
    verdicts = parse_judge_response(raw, ["a claim", "another claim"])
    assert [v.verdict for v in verdicts] == ["SUPPORTED", "UNSUPPORTED"]
    assert verdicts[0].reason == "stated"


def test_parse_tolerates_a_fenced_code_block():
    raw = '```json\n{"verdicts": [{"index": 0, "verdict": "SUPPORTED", "reason": "ok"}]}\n```'
    assert len(parse_judge_response(raw, ["a claim that is long enough"])) == 1


def test_parse_fails_closed_on_garbage():
    """An unparseable judge response must not silently pass. Every claim is
    marked UNSUPPORTED so the answer counts unfaithful and the failure is
    visible rather than flattering."""
    verdicts = parse_judge_response("the model said something else entirely", ["a", "b"])
    assert [v.verdict for v in verdicts] == ["UNSUPPORTED", "UNSUPPORTED"]
    assert all("unparseable" in v.reason.lower() for v in verdicts)


def test_parse_fails_closed_on_missing_verdict():
    raw = '{"verdicts": [{"index": 0, "verdict": "SUPPORTED", "reason": "ok"}]}'
    verdicts = parse_judge_response(raw, ["claim one", "claim two"])
    assert len(verdicts) == 2
    assert verdicts[1].verdict == "UNSUPPORTED"


def test_parse_rejects_an_unknown_verdict_label():
    raw = '{"verdicts": [{"index": 0, "verdict": "PROBABLY", "reason": "hedging"}]}'
    assert parse_judge_response(raw, ["a claim"])[0].verdict == "UNSUPPORTED"


def test_answer_is_faithful_only_when_every_claim_is_supported():
    supported = [ClaimVerdict(0, "c", "SUPPORTED", "r"), ClaimVerdict(1, "c", "SUPPORTED", "r")]
    assert answer_verdict_from(supported) == "FAITHFUL"


def test_not_a_claim_does_not_make_an_answer_unfaithful():
    mixed = [ClaimVerdict(0, "c", "SUPPORTED", "r"), ClaimVerdict(1, "c", "NOT_A_CLAIM", "r")]
    assert answer_verdict_from(mixed) == "FAITHFUL"


def test_one_unsupported_claim_makes_the_answer_unfaithful():
    mixed = [ClaimVerdict(0, "c", "SUPPORTED", "r"), ClaimVerdict(1, "c", "UNSUPPORTED", "r")]
    assert answer_verdict_from(mixed) == "UNFAITHFUL"


def test_an_answer_with_no_claims_is_faithful():
    """A refusal asserts nothing, so it cannot be unfaithful."""
    assert answer_verdict_from([]) == "FAITHFUL"
```

- [ ] **Step 4: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.judge'`

- [ ] **Step 5: Write `eval/judge.py`**

```python
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.claims import CLAIM_SPLITTER_VERSION, split_claims
from eval.config import load_eval_config
from src.providers import get_provider

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/judge")
VALID_VERDICTS = {"SUPPORTED", "UNSUPPORTED", "NOT_A_CLAIM"}
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

API_CALLS = 0  # test-visible, so the cache can be proven to avoid calls


@dataclass(frozen=True)
class ClaimVerdict:
    index: int
    claim: str
    verdict: str
    reason: str


@dataclass(frozen=True)
class JudgeResult:
    question: str
    answer: str
    claims: tuple
    verdicts: tuple
    answer_verdict: str
    model: str
    prompt_version: str
    splitter_version: str


def build_judge_prompt(question: str, sections: list, claims: list) -> str:
    """sections is a list of (section_id, title, text)."""
    cfg = load_eval_config()
    version = cfg["judge"]["prompt_version"]
    template = Path(f"prompts/{version}.txt").read_text(encoding="utf-8")
    section_block = "\n\n---\n\n".join(
        f"[BNS {sid.replace('bns-', '')}] {title}\n{text}" for sid, title, text in sections
    )
    claim_block = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return template.format(sections=section_block, question=question, claims=claim_block)


def parse_judge_response(raw: str, claims: list) -> list:
    """Parse the judge's JSON, failing CLOSED.

    Anything unparseable, missing, or carrying an unknown label becomes
    UNSUPPORTED. A judge that cannot be read must not be treated as approving
    — that would make every parsing bug look like a perfect score.
    """
    verdicts_by_index = {}
    match = _JSON_BLOCK.search(raw or "")
    if match:
        try:
            payload = json.loads(match.group(0))
            for item in payload.get("verdicts", []):
                index = int(item.get("index", -1))
                verdict = str(item.get("verdict", "")).strip().upper()
                if verdict not in VALID_VERDICTS:
                    continue
                verdicts_by_index[index] = (verdict, str(item.get("reason", "")).strip())
        except (ValueError, TypeError, AttributeError) as exc:
            log.warning("judge response did not parse: %s", exc)

    out = []
    for i, claim in enumerate(claims):
        if i in verdicts_by_index:
            verdict, reason = verdicts_by_index[i]
        else:
            verdict, reason = "UNSUPPORTED", "unparseable or missing judge verdict"
            log.warning("no usable verdict for claim %d; failing closed", i)
        out.append(ClaimVerdict(index=i, claim=claim, verdict=verdict, reason=reason))
    return out


def answer_verdict_from(verdicts: list) -> str:
    """An answer is FAITHFUL iff no claim is UNSUPPORTED.

    NOT_A_CLAIM segments do not count against it. This is the mapping used to
    compare judge output against human answer-level labels.
    """
    return "UNFAITHFUL" if any(v.verdict == "UNSUPPORTED" for v in verdicts) else "FAITHFUL"


def _cache_path(question, answer, claims, model, prompt_version) -> Path:
    key = json.dumps(
        {
            "question": question,
            "answer": answer,
            "claims": list(claims),
            "splitter": CLAIM_SPLITTER_VERSION,
            "prompt": prompt_version,
            "model": model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def judge_answer(question: str, answer: str, sections: list, model: str,
                 prompt_version: str = None) -> JudgeResult:
    """One call per answer; per-claim verdicts come back inside it.

    Per-claim as separate calls would be roughly 850 Flash requests against a
    250/day allowance — infeasible, which is why the verdicts are batched
    into one structured response.
    """
    global API_CALLS
    cfg = load_eval_config()
    prompt_version = prompt_version or cfg["judge"]["prompt_version"]
    claims = split_claims(answer)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(question, answer, claims, model, prompt_version)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        verdicts = tuple(ClaimVerdict(**v) for v in cached["verdicts"])
        return JudgeResult(
            question=question, answer=answer, claims=tuple(claims), verdicts=verdicts,
            answer_verdict=cached["answer_verdict"], model=model,
            prompt_version=prompt_version, splitter_version=CLAIM_SPLITTER_VERSION,
        )

    if not claims:
        result = JudgeResult(
            question=question, answer=answer, claims=(), verdicts=(),
            answer_verdict="FAITHFUL", model=model,
            prompt_version=prompt_version, splitter_version=CLAIM_SPLITTER_VERSION,
        )
    else:
        provider = get_provider("gemini", model=model)
        API_CALLS += 1
        raw = provider.generate(
            build_judge_prompt(question, sections, claims),
            cfg["judge"]["max_tokens"],
        )
        verdicts = tuple(parse_judge_response(raw, claims))
        result = JudgeResult(
            question=question, answer=answer, claims=tuple(claims), verdicts=verdicts,
            answer_verdict=answer_verdict_from(verdicts), model=model,
            prompt_version=prompt_version, splitter_version=CLAIM_SPLITTER_VERSION,
        )

    path.write_text(
        json.dumps(
            {
                "answer_verdict": result.answer_verdict,
                "verdicts": [asdict(v) for v in result.verdicts],
                "claims": list(result.claims),
                "splitter_version": CLAIM_SPLITTER_VERSION,
                "prompt_version": prompt_version,
                "model": model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_judge.py -v`
Expected: 11 passed. **These are unit tests — they make no API calls** and must not be `integration`-marked.

- [ ] **Step 7: Add `.cache/judge/` to `.gitignore`**

`.cache/` is already ignored — verify with `git check-ignore -v .cache/judge/x.json` and add nothing if it already matches.

- [ ] **Step 8: Commit**

```bash
git add eval/judge.py prompts/judge_faithfulness_v1.txt tests/test_judge.py src/providers/__init__.py
git commit -m "feat: faithfulness judge, one call per answer, fails closed on unparseable output"
```

---

## Task 7: Human labelling CLI

**Files:**
- Create: `eval/label_cli.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: `load_golden`, `load_eval_config`.
- Produces: `HumanLabel` dataclass with `question_id, split, verdict, reason`; `load_labels(path) -> dict[str, HumanLabel]`; `save_label(path, label)`; `pending_ids(cfg, labels, split) -> list[str]`; `main()`.

**Context:** the human labels 20 answers as FAITHFUL or UNFAITHFUL — 10 tuning, 10 held out. The CLI shows question, retrieved sections, and answer, then records the verdict. It must be **resumable**: labelling 20 answers is not one sitting, and losing progress would be worse than not starting.

The storage logic is unit-tested; the interactive loop is not.

- [ ] **Step 1: Write the failing test**

`tests/test_labels.py`:

```python
import pytest

from eval.label_cli import HumanLabel, load_labels, pending_ids, save_label


def _cfg():
    return {
        "validation": {
            "tuning_ids": ["look-001", "look-002", "sit-001"],
            "holdout_ids": ["look-005", "safe-013"],
        }
    }


def test_round_trip_a_label(tmp_path):
    path = tmp_path / "labels.yaml"
    save_label(path, HumanLabel("look-001", "tuning", "FAITHFUL", "matches 303(2)"))
    labels = load_labels(path)
    assert labels["look-001"].verdict == "FAITHFUL"
    assert labels["look-001"].split == "tuning"
    assert labels["look-001"].reason == "matches 303(2)"


def test_missing_file_loads_as_empty(tmp_path):
    assert load_labels(tmp_path / "nope.yaml") == {}


def test_saving_is_append_only_across_calls(tmp_path):
    path = tmp_path / "labels.yaml"
    save_label(path, HumanLabel("look-001", "tuning", "FAITHFUL", ""))
    save_label(path, HumanLabel("look-002", "tuning", "UNFAITHFUL", "cites 999"))
    labels = load_labels(path)
    assert set(labels) == {"look-001", "look-002"}


def test_relabelling_overwrites_rather_than_duplicating(tmp_path):
    path = tmp_path / "labels.yaml"
    save_label(path, HumanLabel("look-001", "tuning", "FAITHFUL", ""))
    save_label(path, HumanLabel("look-001", "tuning", "UNFAITHFUL", "changed my mind"))
    labels = load_labels(path)
    assert len(labels) == 1
    assert labels["look-001"].verdict == "UNFAITHFUL"


def test_pending_skips_what_is_already_labelled(tmp_path):
    labels = {"look-001": HumanLabel("look-001", "tuning", "FAITHFUL", "")}
    assert pending_ids(_cfg(), labels, "tuning") == ["look-002", "sit-001"]


def test_pending_is_empty_when_everything_is_labelled():
    labels = {
        i: HumanLabel(i, "tuning", "FAITHFUL", "")
        for i in _cfg()["validation"]["tuning_ids"]
    }
    assert pending_ids(_cfg(), labels, "tuning") == []


def test_pending_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        pending_ids(_cfg(), {}, "nonsense")


def test_verdict_must_be_one_of_two_values(tmp_path):
    with pytest.raises(ValueError, match="verdict"):
        save_label(tmp_path / "l.yaml", HumanLabel("look-001", "tuning", "MAYBE", ""))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.label_cli'`

- [ ] **Step 3: Write `eval/label_cli.py`**

```python
import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from eval.config import load_eval_config
from eval.golden import by_id, load_golden

log = logging.getLogger(__name__)

LABELS_PATH = Path("eval/human_labels.yaml")
VALID_VERDICTS = {"FAITHFUL", "UNFAITHFUL"}
SPLITS = {"tuning": "tuning_ids", "holdout": "holdout_ids"}


@dataclass(frozen=True)
class HumanLabel:
    question_id: str
    split: str
    verdict: str
    reason: str


def load_labels(path=LABELS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {item["question_id"]: HumanLabel(**item) for item in raw}


def save_label(path, label: HumanLabel) -> None:
    """Write one label, replacing any earlier verdict for the same question.

    Rewrites the whole file each time. At twenty labels that is free, and it
    makes the CLI resumable — the human can stop after any single answer.
    """
    if label.verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}, got {label.verdict!r}")
    path = Path(path)
    labels = load_labels(path)
    labels[label.question_id] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump([asdict(labels[k]) for k in sorted(labels)], sort_keys=False),
        encoding="utf-8",
    )


def pending_ids(cfg: dict, labels: dict, split: str) -> list:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}. Known: {sorted(SPLITS)}")
    wanted = cfg["validation"][SPLITS[split]]
    return [i for i in wanted if i not in labels]


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Label generated answers FAITHFUL or UNFAITHFUL."
    )
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--answers", default="eval/results/answers.json",
                        help="answers produced by eval.generation_eval --dump-answers")
    args = parser.parse_args()

    import json

    cfg = load_eval_config()
    labels = load_labels()
    pending = pending_ids(cfg, labels, args.split)
    if not pending:
        print(f"Nothing left to label in the {args.split} split.")
        return 0

    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    questions = by_id(load_golden())

    print(f"{len(pending)} to label in {args.split}. Ctrl-C to stop; progress is saved after each.\n")
    for n, qid in enumerate(pending, start=1):
        record = answers.get(qid)
        if record is None:
            print(f"[{qid}] no generated answer found — skipping")
            continue
        q = questions[qid]
        print("=" * 78)
        print(f"({n}/{len(pending)})  {qid}  [{q.category}]")
        print("=" * 78)
        print(f"\nQUESTION\n  {q.question}\n")
        print("RETRIEVED SECTIONS")
        for s in record["sections"]:
            print(f"  [{s['section_id']}] {s['section_title']}")
        print(f"\nANSWER\n{record['answer']}\n")
        print("-" * 78)

        verdict = ""
        while verdict not in {"f", "u", "s"}:
            verdict = input("faithful (f) / unfaithful (u) / skip (s): ").strip().lower()
        if verdict == "s":
            continue
        reason = input("reason (optional): ").strip()
        save_label(
            LABELS_PATH,
            HumanLabel(
                question_id=qid,
                split=args.split,
                verdict="FAITHFUL" if verdict == "f" else "UNFAITHFUL",
                reason=reason,
            ),
        )
        print("saved.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_labels.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add eval/label_cli.py tests/test_labels.py
git commit -m "feat: resumable human labelling CLI for judge validation"
```

---

## Task 8: Harness B — generation eval

**Files:**
- Create: `eval/generation_eval.py`, `tests/test_generation_eval.py`

**Interfaces:**
- Consumes: `load_golden`, `load_eval_config`, `hybrid_search`, `answer`, `verify_citations`, `judge_answer`.
- Produces: `AnswerRecord` dataclass with `question_id, question, answer, sections, fabricated, unparseable`; `generate_answers(conn, questions, provider) -> dict[str, AnswerRecord]`; `main()` with `--dump-answers` and `--judge` flags.

**Context:** two phases, deliberately separable so the human can label between them. `--dump-answers` generates answers and writes them to JSON, spending Flash. `--judge` reads that JSON and scores it, spending Flash again. Splitting them means labelling does not force a regeneration.

Citation-verification violations come free — `verify_citations` already runs with no API call.

- [ ] **Step 1: Write the failing test**

`tests/test_generation_eval.py`:

```python
from eval.generation_eval import AnswerRecord, summarise_verifications


def _record(qid, fabricated, unparseable):
    return AnswerRecord(
        question_id=qid, question="q", answer="a", sections=[],
        fabricated=fabricated, unparseable=unparseable,
    )


def test_counts_fabricated_citations_per_category():
    records = {"look-001": _record("look-001", ["999"], [])}
    categories = {"look-001": "lookup"}
    summary = summarise_verifications(records, categories)
    assert summary["lookup"]["fabricated"] == 1


def test_counts_answers_not_questions():
    """Two fabricated citations in one answer is one bad answer, not two."""
    records = {"look-001": _record("look-001", ["999", "998"], [])}
    categories = {"look-001": "lookup"}
    assert summarise_verifications(records, categories)["lookup"]["fabricated"] == 1


def test_clean_answers_report_zero():
    records = {"look-001": _record("look-001", [], [])}
    categories = {"look-001": "lookup"}
    summary = summarise_verifications(records, categories)
    assert summary["lookup"]["fabricated"] == 0
    assert summary["lookup"]["n"] == 1


def test_unparseable_citations_are_counted_separately():
    records = {"safe-001": _record("safe-001", [], ["see above"])}
    categories = {"safe-001": "safety"}
    assert summarise_verifications(records, categories)["safety"]["unparseable"] == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.generation_eval'`

- [ ] **Step 3: Write `eval/generation_eval.py`**

```python
import argparse
import collections
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.config import load_eval_config
from eval.golden import load_golden
from eval.judge import judge_answer
from src.db import connect
from src.generate import MAX_GROUNDED_TOKENS, build_prompt
from src.providers import get_provider
from src.retrieve import hybrid_search
from src.verify import verify_citations

log = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results")
ANSWERS_PATH = RESULTS_DIR / "answers.json"


@dataclass
class AnswerRecord:
    question_id: str
    question: str
    answer: str
    sections: list
    fabricated: list
    unparseable: list


def generate_answers(conn, questions: list, provider) -> dict:
    """One grounded answer per question. Spends one Flash call each."""
    cfg = load_eval_config()
    top_n = cfg["retrieval"]["top_n"]
    out = {}
    for q in questions:
        retrieved = hybrid_search(conn, q.question, top_n=top_n)
        raw = provider.generate(build_prompt(q.question, retrieved), MAX_GROUNDED_TOKENS)
        verification = verify_citations(raw, retrieved)
        out[q.id] = AnswerRecord(
            question_id=q.id,
            question=q.question,
            answer=raw,
            sections=[
                {"section_id": r.section_id, "section_title": r.section_title, "text": r.text}
                for r in retrieved
            ],
            fabricated=list(verification.fabricated),
            unparseable=list(getattr(verification, "unparseable", [])),
        )
        log.info("answered %s (%d fabricated)", q.id, len(verification.fabricated))
    return out


def summarise_verifications(records: dict, categories: dict) -> dict:
    """Citation-verification violations per category. Costs no API calls.

    Counts answers, not citations: two fabricated citations in one answer is
    one bad answer.
    """
    summary = collections.defaultdict(lambda: {"n": 0, "fabricated": 0, "unparseable": 0})
    for qid, record in records.items():
        bucket = summary[categories[qid]]
        bucket["n"] += 1
        if record.fabricated:
            bucket["fabricated"] += 1
        if record.unparseable:
            bucket["unparseable"] += 1
    return dict(summary)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Harness B: generation eval. Spends Flash quota.")
    parser.add_argument("--dump-answers", action="store_true",
                        help="generate answers and write them for labelling")
    parser.add_argument("--judge", action="store_true",
                        help="judge previously dumped answers")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    args = parser.parse_args()

    if not (args.dump_answers or args.judge):
        raise SystemExit("pass --dump-answers or --judge")

    cfg = load_eval_config()
    questions = load_golden()
    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        questions = [q for q in questions if q.id in wanted]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dump_answers:
        provider = get_provider("gemini")
        log.warning("about to spend %d gemini-2.5-flash calls (250/day)", len(questions))
        with connect() as conn:
            records = generate_answers(conn, questions, provider)
        ANSWERS_PATH.write_text(
            json.dumps({k: asdict(v) for k, v in records.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("wrote %d answers to %s", len(records), ANSWERS_PATH)

        categories = {q.id: q.category for q in questions}
        print(json.dumps(summarise_verifications(records, categories), indent=2))

    if args.judge:
        model = cfg["judge"]["chosen"]
        if not model:
            raise SystemExit(
                "config/eval.yaml judge.chosen is null. Run eval.validate_judge first "
                "and set it from the measured agreement — do not guess."
            )
        raw = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
        by_category = collections.defaultdict(lambda: {"n": 0, "faithful": 0})
        cats = {q.id: q.category for q in questions}
        for qid, record in raw.items():
            if qid not in cats:
                continue
            sections = [(s["section_id"], s["section_title"], s["text"]) for s in record["sections"]]
            result = judge_answer(record["question"], record["answer"], sections, model)
            bucket = by_category[cats[qid]]
            bucket["n"] += 1
            if result.answer_verdict == "FAITHFUL":
                bucket["faithful"] += 1
        print(json.dumps(dict(by_category), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generation_eval.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add eval/generation_eval.py tests/test_generation_eval.py
git commit -m "feat: Harness B, generation eval with separable answer and judge phases"
```

---

## Task 9: Judge validation — both models, held-out discipline, disagreement log

**Files:**
- Create: `eval/validate_judge.py`, `tests/test_validate_judge.py`

**Interfaces:**
- Consumes: `load_labels`, `load_eval_config`, `judge_answer`, `binary_agreement`, `AgreementScores`.
- Produces: `check_holdout_allowed(cfg, prompt_version, split)`; `Disagreement` dataclass with `question_id, human, judge, judge_reason, claims`; `score_model(records, labels, model, prompt_version) -> tuple[AgreementScores, list[Disagreement]]`; `format_disagreements(rows) -> str`; `main()`.

**Context:** this is what turns a number into a measurement.

- Both candidate models are scored. `judge.chosen` stays null until measured.
- The **held-out ten are touched exactly once**, on the final prompt. This is enforced in code — a held-out run with a non-final prompt version refuses — not left to discipline.
- Accuracy, precision, recall and Cohen's κ are all reported. κ because a judge that always answers FAITHFUL scores high accuracy on an unbalanced set while being worthless.
- Every disagreement is logged with both verdicts and the judge's stated reason.

- [ ] **Step 1: Write the failing test**

`tests/test_validate_judge.py`:

```python
import pytest

from eval.validate_judge import Disagreement, check_holdout_allowed, format_disagreements


def _cfg(final="judge_faithfulness_v3"):
    return {"judge": {"prompt_version": final, "final_prompt_version": final}}


def test_tuning_split_is_always_allowed():
    check_holdout_allowed(_cfg(), "judge_faithfulness_v1", "tuning")


def test_holdout_allowed_on_the_final_prompt():
    check_holdout_allowed(_cfg(), "judge_faithfulness_v3", "holdout")


def test_holdout_refuses_a_non_final_prompt():
    """The held-out ten are touched exactly once. Enforcing that in code
    rather than by discipline is the point — a held-out set peeked at during
    tuning is just more tuning data."""
    with pytest.raises(RuntimeError, match="held-out"):
        check_holdout_allowed(_cfg(), "judge_faithfulness_v1", "holdout")


def test_holdout_refuses_when_no_final_prompt_is_declared():
    cfg = {"judge": {"prompt_version": "v1", "final_prompt_version": None}}
    with pytest.raises(RuntimeError, match="final_prompt_version"):
        check_holdout_allowed(cfg, "v1", "holdout")


def test_disagreement_log_shows_both_verdicts_and_the_reason():
    rows = [
        Disagreement("safe-013", "UNFAITHFUL", "FAITHFUL",
                     "the section covers abetment", ["a claim about abetment"]),
    ]
    out = format_disagreements(rows)
    assert "safe-013" in out
    assert "UNFAITHFUL" in out
    assert "FAITHFUL" in out
    assert "the section covers abetment" in out


def test_empty_disagreement_log_says_so_explicitly():
    out = format_disagreements([])
    assert "no disagreement" in out.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_validate_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.validate_judge'`

- [ ] **Step 3: Write `eval/validate_judge.py`**

```python
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from eval.config import load_eval_config
from eval.judge import judge_answer
from eval.label_cli import load_labels
from eval.metrics import binary_agreement

log = logging.getLogger(__name__)

ANSWERS_PATH = Path("eval/results/answers.json")
DISAGREEMENTS_PATH = Path("eval/disagreements.md")


@dataclass(frozen=True)
class Disagreement:
    question_id: str
    human: str
    judge: str
    judge_reason: str
    claims: list


def check_holdout_allowed(cfg: dict, prompt_version: str, split: str) -> None:
    """The held-out ten are touched exactly once, on the final prompt.

    Enforced here rather than left to discipline. A held-out set peeked at
    during tuning is not held out — it is just more tuning data, and the
    number it produces is not the out-of-sample number it claims to be.
    """
    if split != "holdout":
        return
    final = cfg["judge"].get("final_prompt_version")
    if not final:
        raise RuntimeError(
            "config/eval.yaml judge.final_prompt_version is not set. Declare the "
            "final prompt before touching the held-out split."
        )
    if prompt_version != final:
        raise RuntimeError(
            f"refusing to run the held-out split with prompt {prompt_version!r}; "
            f"final_prompt_version is {final!r}. The held-out ten are used once."
        )


def score_model(records: dict, labels: dict, model: str, prompt_version: str):
    """Judge every labelled answer with one model and compare to the human."""
    human, judged, disagreements = [], [], []
    for qid in sorted(labels):
        record = records.get(qid)
        if record is None:
            log.warning("no generated answer for %s; skipping", qid)
            continue
        sections = [(s["section_id"], s["section_title"], s["text"]) for s in record["sections"]]
        result = judge_answer(record["question"], record["answer"], sections,
                              model, prompt_version=prompt_version)
        human.append(labels[qid].verdict)
        judged.append(result.answer_verdict)
        if labels[qid].verdict != result.answer_verdict:
            offending = [v for v in result.verdicts if v.verdict == "UNSUPPORTED"]
            reason = offending[0].reason if offending else "all claims supported"
            disagreements.append(
                Disagreement(
                    question_id=qid,
                    human=labels[qid].verdict,
                    judge=result.answer_verdict,
                    judge_reason=reason,
                    claims=list(result.claims),
                )
            )
    return binary_agreement(human, judged), disagreements


def format_disagreements(rows: list) -> str:
    if not rows:
        return "# Judge disagreements\n\nNo disagreements on this run.\n"
    lines = ["# Judge disagreements", "",
             "Every case where the judge and the human differed. This file is the "
             "input to the next prompt revision.", ""]
    for r in rows:
        lines.append(f"## {r.question_id}")
        lines.append("")
        lines.append(f"- human: **{r.human}**")
        lines.append(f"- judge: **{r.judge}**")
        lines.append(f"- judge's reason: {r.judge_reason}")
        lines.append("")
        lines.append("Claims as split:")
        for c in r.claims:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Score every candidate judge model against human labels."
    )
    parser.add_argument("--split", choices=["tuning", "holdout"], required=True)
    parser.add_argument("--prompt-version", default=None)
    args = parser.parse_args()

    cfg = load_eval_config()
    prompt_version = args.prompt_version or cfg["judge"]["prompt_version"]
    check_holdout_allowed(cfg, prompt_version, args.split)

    labels = {k: v for k, v in load_labels().items() if v.split == args.split}
    if not labels:
        raise SystemExit(f"no {args.split} labels yet. Run eval.label_cli --split {args.split}")

    records = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))

    all_disagreements = []
    print(f"\nsplit={args.split}  prompt={prompt_version}  n={len(labels)}\n")
    print(f"| {'model':<26}| {'acc':>6} | {'prec':>6} | {'rec':>6} | {'kappa':>7} |")
    print(f"|{'-' * 27}|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 9}|")
    for model in cfg["judge"]["candidates"]:
        scores, disagreements = score_model(records, labels, model, prompt_version)
        print(f"| {model:<26}| {scores.accuracy:>6.3f} | {scores.precision:>6.3f} "
              f"| {scores.recall:>6.3f} | {scores.kappa:>7.3f} |")
        all_disagreements.extend(disagreements)

    DISAGREEMENTS_PATH.write_text(format_disagreements(all_disagreements), encoding="utf-8")
    log.info("wrote %d disagreements to %s", len(all_disagreements), DISAGREEMENTS_PATH)
    print("\nChoose judge.chosen in config/eval.yaml from the measured agreement above.")
    print("Kappa, not accuracy, is the number to choose on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add `final_prompt_version` to `config/eval.yaml`**

Under `judge:`, add:

```yaml
  # Declared before the held-out split may be touched. Until then the
  # held-out ten stay untouched — see the Phase 2a design doc section 7.2.
  final_prompt_version: null
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_validate_judge.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add eval/validate_judge.py tests/test_validate_judge.py config/eval.yaml
git commit -m "feat: judge validation with held-out enforcement and disagreement log"
```

---

## Task 10: CI, README, and the regression gate

**Files:**
- Modify: `.github/workflows/ci.yml`, `README.md`
- Create: `eval/baseline.json`, `eval/check_regression.py`, `tests/test_regression.py`

**Interfaces:**
- Consumes: results JSON written by `eval/retrieval_eval.py`.
- Produces: `compare_to_baseline(current, baseline, tolerance=0.05) -> list[str]`; `main()` exiting non-zero on regression.

**Context:** design doc §9 — the build fails if retrieval pass rate drops more than 5 points from the last recorded run, reported per category. Harness B does **not** run in CI: it needs an API key and spends Flash quota.

- [ ] **Step 1: Write the failing test**

`tests/test_regression.py`:

```python
from eval.check_regression import compare_to_baseline


def _scores(migration_pass):
    return {"hybrid": {"lookup": {"pass_rate": 0.9},
                       "migration": {"pass_rate": migration_pass}}}


def test_no_regression_when_scores_hold():
    assert compare_to_baseline(_scores(0.9), _scores(0.9)) == []


def test_improvement_is_not_a_regression():
    assert compare_to_baseline(_scores(1.0), _scores(0.9)) == []


def test_small_drop_within_tolerance_passes():
    assert compare_to_baseline(_scores(0.87), _scores(0.9)) == []


def test_drop_beyond_tolerance_is_reported():
    failures = compare_to_baseline(_scores(0.80), _scores(0.9))
    assert len(failures) == 1
    assert "migration" in failures[0]


def test_each_category_is_checked_separately():
    """An aggregate would hide a single category collapsing. That is exactly
    the failure this gate exists to catch."""
    current = {"hybrid": {"lookup": {"pass_rate": 1.0}, "migration": {"pass_rate": 0.0}}}
    baseline = {"hybrid": {"lookup": {"pass_rate": 0.9}, "migration": {"pass_rate": 0.9}}}
    failures = compare_to_baseline(current, baseline)
    assert len(failures) == 1
    assert "migration" in failures[0]


def test_a_missing_category_is_a_failure():
    current = {"hybrid": {"lookup": {"pass_rate": 0.9}}}
    baseline = {"hybrid": {"lookup": {"pass_rate": 0.9}, "migration": {"pass_rate": 0.9}}}
    failures = compare_to_baseline(current, baseline)
    assert any("migration" in f for f in failures)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regression.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.check_regression'`

- [ ] **Step 3: Write `eval/check_regression.py`**

```python
import argparse
import json
import sys
from pathlib import Path

BASELINE_PATH = Path("eval/baseline.json")
TOLERANCE = 0.05


def compare_to_baseline(current: dict, baseline: dict, tolerance: float = TOLERANCE) -> list:
    """Per-category pass-rate comparison. Returns human-readable failures.

    Compared per category, never as an aggregate: an aggregate of 78 percent
    hides a category at 30 percent, and a gate on the aggregate would let that
    collapse through.
    """
    failures = []
    for config_name, categories in baseline.items():
        current_config = current.get(config_name, {})
        for category, base in categories.items():
            now = current_config.get(category)
            if now is None:
                failures.append(f"{config_name}/{category}: missing from the current run")
                continue
            drop = base["pass_rate"] - now["pass_rate"]
            if drop > tolerance:
                failures.append(
                    f"{config_name}/{category}: pass rate {now['pass_rate']:.3f} is "
                    f"{drop:.3f} below baseline {base['pass_rate']:.3f} "
                    f"(tolerance {tolerance:.2f})"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail the build on a per-category regression.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}; nothing to compare against")
        return 0

    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures = compare_to_baseline(current, baseline)

    if failures:
        print("RETRIEVAL REGRESSION")
        for f in failures:
            print(f"  {f}")
        return 1
    print("no per-category regression against baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_regression.py -v`
Expected: 6 passed

- [ ] **Step 5: Record the baseline from the real run**

```bash
export TEST_DATABASE_URL="postgresql://nyaya:nyaya@localhost:55432/nyaya_test"
.venv/Scripts/python.exe -m eval.retrieval_eval
cp "$(ls -t eval/results/retrieval-*.json | head -1)" eval/baseline.json
```

`eval/baseline.json` **is committed** — it is the reference the gate compares against. `eval/results/` stays gitignored.

- [ ] **Step 6: Extend `.github/workflows/ci.yml`**

Add a step after the existing ingest/mapping validation, inside the same job that already has the pgvector service container:

```yaml
      - name: Retrieval eval (Harness A) — zero generation calls
        env:
          TEST_DATABASE_URL: postgresql://nyaya:nyaya@localhost:5432/nyaya_test
        run: |
          # Harness A needs a populated index. Embeddings require GEMINI_API_KEY,
          # which CI does not have, so this step runs only when the secret is
          # configured. Without it the retrieval gate is skipped and the rest of
          # the workflow — unit tests, parser validation — still guards the build.
          if [ -z "${GEMINI_API_KEY}" ]; then
            echo "GEMINI_API_KEY not set; skipping Harness A."
            echo "To enable: add GEMINI_API_KEY as a repository secret."
            exit 0
          fi
          python -m src.store
          python -m eval.retrieval_eval
          python -m eval.check_regression --current "$(ls -t eval/results/retrieval-*.json | head -1)"
```

Add `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}` to the job's `env` block.

- [ ] **Step 7: Validate the workflow parses**

```bash
.venv/Scripts/python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"
```

- [ ] **Step 8: Demonstrate the gate actually fails**

Prove it rather than assume it:

```bash
.venv/Scripts/python.exe -c "
import json, pathlib
b = json.loads(pathlib.Path('eval/baseline.json').read_text())
for cfg in b.values():
    for cat in cfg.values():
        cat['pass_rate'] = min(1.0, cat['pass_rate'] + 0.5)
pathlib.Path('/tmp/inflated.json').write_text(json.dumps(b))
"
.venv/Scripts/python.exe -m eval.check_regression --current eval/baseline.json --baseline /tmp/inflated.json
echo "exit code: $?  (must be 1)"
```

Expected: per-category failure lines and exit code 1. Then delete the temp file.

- [ ] **Step 9: Update `README.md`**

Add a Phase 2a section covering: how to run Harness A, how to run Harness B's two phases, how to label, and how to validate the judge. Report the **actual measured numbers** from your runs — retrieval per category for all three configurations, and the judge agreement table for both candidate models.

Two rules on wording, from SPEC.md §11:
- The faithfulness number and the judge's agreement rate appear **side by side**. A faithfulness score without the agreement rate that earned it is not a measurement.
- No claim may outrun what was actually verified. If the held-out split has not been run, say so.

- [ ] **Step 10: Commit**

```bash
git add eval/check_regression.py tests/test_regression.py eval/baseline.json .github/workflows/ci.yml README.md
git commit -m "feat: per-category retrieval regression gate in CI"
```

---

## Execution order and the human checkpoint

Tasks 1–5 need no API keys and no database beyond the container. Tasks 6–10 spend quota.

**STOP after Task 8's `--dump-answers` run.** Labelling is a human step:

```bash
.venv/Scripts/python.exe -m eval.generation_eval --dump-answers --only <the 20 validation ids>
.venv/Scripts/python.exe -m eval.label_cli --split tuning     # human
.venv/Scripts/python.exe -m eval.validate_judge --split tuning
# revise prompts/judge_faithfulness_vN.txt, repeat, up to the budget of three
# then declare final_prompt_version, label holdout, run once:
.venv/Scripts/python.exe -m eval.label_cli --split holdout    # human
.venv/Scripts/python.exe -m eval.validate_judge --split holdout
```

Only after `judge.chosen` is set from measured agreement does `--judge` on the full 55 make sense.

## Done criteria

- `pytest tests/` green, with every new unit test running secret-free
- Harness A produces per-category tables for all three configurations, no aggregate
- `eval/baseline.json` committed; regression gate demonstrated failing on inflated input
- 20 human labels recorded, 10 tuning and 10 held out
- Both judge models scored with accuracy, precision, recall and κ
- `eval/disagreements.md` written
- `judge.chosen` set from measurement, not assumption
- README reports faithfulness and judge agreement side by side
