# Nyaya Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the grounded-RAG spine over the Bharatiya Nyaya Sanhita 2023 — ingest 358 sections from the gazette PDF, map them to their IPC ancestors, embed and store them in Neon Postgres, and answer a question from the CLI with verified citations.

**Architecture:** A hand-built pipeline with no framework. PDF → bold-span section parser validated against the Act's own index → `sections.json` → Gemini embeddings → Neon Postgres with pgvector and a generated tsvector → dense retrieval → Gemini generation behind a provider interface → programmatic citation verification. Every stage is a separate module with one responsibility, so each is independently testable and the whole thing is debuggable without a framework in the way.

**Tech Stack:** Python 3.10.9, PyMuPDF 1.28.2 (pinned), google-genai, psycopg 3, pgvector, Neon Postgres, pytest.

**Spec:** `SPEC.md` and `docs/superpowers/specs/2026-08-21-nyaya-design.md`. Read both. Where they disagree, the design doc wins.

## Global Constraints

- **Python 3.10.9.** The existing `.venv/` at the repo root. Do not migrate to 3.11.
- **`pymupdf==1.28.2` is pinned exactly.** Span-level font name and size extraction is version sensitive and the parser depends on that behaviour.
- **No OpenAI anywhere in the codebase.** Not in code, not in comments, not in `requirements.txt`.
- **No `torch`, no `sentence-transformers`.** Deployment target is a 512MB free tier.
- **Embedding model is `gemini-embedding-2` at `output_dimensionality=768`.** Never 3072 — pgvector's HNSW index caps at 2000 dimensions.
- **`DATABASE_URL` and `GEMINI_API_KEY` come from `.env` only.** Never hardcoded, never committed. `.env` is in `.gitignore` before any other file is written.
- **Storage code uses plain Postgres + pgvector only.** No Neon-specific features.
- **Generation sits behind a provider interface.** Swapping the provider must not touch anything outside `src/providers/`.
- **Raw PDFs in `data/raw/` are never edited.** Everything downstream is regenerated.
- **No answer may contain a section number that was not retrieved.** Enforced in code, not by prompt.
- **Every response path carries the not-legal-advice disclaimer.**

---

## Decisions this plan makes, and why

Three points where the specs are silent or self-contradictory. Each is settled here so no task has to improvise.

**1. Embedding calls are one section per request, not batched.**

The Gemini docs state plainly: *"Gemini Embedding 2 produces a single aggregated embedding"* when multiple inputs are provided directly in `contents`. Whether wrapping each section in its own `types.Content` avoids this is undocumented, and getting it wrong produces no error — just one averaged vector where 358 should be, silently destroying the index.

At 358 sections against a 1,000 RPD / 100 RPM budget, one call per section costs ~4 minutes and 36% of a day's embedding quota, once, and content-hash caching makes every re-run free. The correctness guarantee is worth more than the round trips. The count assertion from the design doc §1.2 stays in place regardless, as the tripwire. If a future SDK version documents per-`Content` batching that returns N vectors, it drops in behind the same `embed_sections()` interface.

**2. The 6000-character chunk ceiling is raised to 13,000.**

SPEC.md §7 requires "zero chunks over 6000 characters". Measured against the real corpus, **four sections exceed it**: §2 Definitions (12,904), §356 Defamation (8,825), §335 Making a false document (7,387), §101 Murder (6,523).

This is a direct contradiction with SPEC.md §4, which calls one-chunk-per-section *"the most important decision in the project"* and says sections with subsections stay whole. The 6000 figure was an estimate; the whole-section rule is a design principle. **The principle wins.** The ceiling becomes 13,000, which holds all 358 sections whole with headroom.

**3. A separate, stricter guard exists for silent embedding truncation.**

`gemini-embedding-2` accepts 8,192 tokens and **silently truncates** anything longer. The longest section is 12,904 characters ≈ 3,500 tokens, comfortably safe — but "comfortably safe today on BNS" is not a guarantee for BNSS. So a `max_embed_chars: 20000` guard hard-fails the run before any request is sent. 20,000 characters is below 8,192 tokens even at a pessimistic 2.5 characters per token, so passing this guard means truncation is impossible rather than unlikely.

---

## File Structure

```
.gitignore                     env, venv, cache, processed data
.env.example                   every variable, all values blank
requirements.txt               pinned deps
config/acts.yaml               per-act parser config — fonts, pages, expected counts
data/raw/                      the three PDFs + manifest.json (never edited)
data/processed/                sections.json, mappings.json (generated, gitignored)

src/models.py                  Section dataclass — the one shared type
src/config.py                  load acts.yaml and env
src/manifest.py                SHA256 verification, enacted-status gate
src/pdf_text.py                page furniture stripping, body-page joining
src/dehyphen.py                keep-list construction, line-break joining, join log
src/parse_index.py             the Arrangement of Sections oracle
src/parse_sections.py          bold-span heading detection
src/parse_chapters.py          chapter headings
src/ingest.py                  assembles sections.json, runs validation
src/mapping.py                 NCRB table → mappings.json
src/db.py                      connection, schema DDL
src/embed_format.py            BOTH formatters — doc and query, one module
src/embed.py                   Gemini embeddings, content-hash cache, assertions
src/store.py                   load sections + embeddings into Postgres
src/retrieve.py                dense + full-text search
src/providers/base.py          GenerationProvider protocol
src/providers/gemini.py        Gemini implementation
src/generate.py                prompt assembly, provider dispatch
src/verify.py                  citation verification
src/cli.py                     end-to-end entry point
prompts/grounded_v1.txt
scripts/acceptance.py          the Phase 1 checklist as runnable code
tests/                         one test module per src module
```

Files that change together live together: everything that reads the PDF is a separate small module, because each is a distinct failure mode with a distinct test. `embed_format.py` is deliberately its own module so index-time and query-time formatting cannot drift apart — that is the whole reason it exists.

---

## Task 1: Repo skeleton, config, and manifest verification

**Files:**
- Create: `.gitignore`, `.env.example`, `requirements.txt`, `config/acts.yaml`
- Create: `src/__init__.py`, `src/models.py`, `src/config.py`, `src/manifest.py`
- Create: `data/raw/manifest.json`
- Test: `tests/test_manifest.py`, `tests/test_config.py`
- Move: `a202345.pdf`, `A202346.pdf`, `BNS2023.pdf` → `data/raw/`

**Interfaces:**
- Consumes: nothing.
- Produces: `Section` dataclass; `load_act_config(key) -> dict`; `verify_source(act_key) -> dict`; `ManifestMismatch` exception.

- [ ] **Step 1: Initialise the repo with the ignore file first**

`.gitignore` must exist before anything else, so `.env` can never be committed even by accident.

```bash
cd /d/ReportEase_RAG
git init
```

Create `.gitignore`:

```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.cache/
data/processed/
logs/
.superpowers/
```

- [ ] **Step 2: Create `.env.example` with every variable, all blank**

```
# Neon Postgres. Use the nyaya-dev project connection string locally.
DATABASE_URL=

# Google Gemini. The only key required to run.
GEMINI_API_KEY=

# Optional. For the later Claude-vs-Gemini generation comparison.
ANTHROPIC_API_KEY=
```

- [ ] **Step 3: Create `requirements.txt`**

```
pymupdf==1.28.2
google-genai>=1.51.0
psycopg[binary]>=3.1
pgvector>=0.3.0
python-dotenv>=1.0
PyYAML>=6.0
pytest>=8.0
```

Install:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

- [ ] **Step 4: Register the `integration` marker and split the suite**

Design doc §6.3 requires the test suite to run on every push with **zero generation calls**. A suite that needs `DATABASE_URL` and `GEMINI_API_KEY` to collect cannot do that, and would spend Flash quota on every run — across a 15-task acceptance loop that alone would exhaust the 250 RPD budget.

So: tests that need a live database or a live API key are marked `integration` and are excluded by default.

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -m "not integration"
markers =
    integration: requires a live database or API key. Run with: pytest -m integration
```

`addopts` is what makes the default run unit-only. To run the integration tests, the marker expression on the command line overrides it:

```bash
.venv/Scripts/python.exe -m pytest tests/                     # unit only, no secrets needed
.venv/Scripts/python.exe -m pytest tests/ -m integration      # needs .env
.venv/Scripts/python.exe -m pytest tests/ -m ""               # everything
```

Modules that are entirely integration declare it once at the top:

```python
import pytest

pytestmark = pytest.mark.integration
```

That applies to `tests/test_db.py`, `tests/test_store.py`, `tests/test_embed.py`, and `tests/test_retrieve.py` in full. In `tests/test_generate.py` only the live-call test is marked; the rest are pure unit tests.

- [ ] **Step 5: Write `README.md`**

```markdown
# Nyaya

Grounded question answering over Indian criminal law. Answers come only from
the text of the Bharatiya Nyaya Sanhita 2023, and every answer cites the
sections it came from.

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
Phase 2. No claim is made here that the acceptance criteria have not verified.
```

- [ ] **Step 6: Move the PDFs into `data/raw/` and write the manifest**

```bash
mkdir -p data/raw data/processed logs
mv a202345.pdf A202346.pdf BNS2023.pdf data/raw/
```

Create `data/raw/manifest.json`. These hashes were computed on 2026-08-21 and are authoritative:

```json
{
  "bns": {
    "path": "data/raw/a202345.pdf",
    "sha256": "ff92dcc72778944011807644b6033b1140ddbe6d7e9f82ac32fd419dae03aa86",
    "act_number": "45 of 2023",
    "status": "enacted",
    "as_of_date": "2025-10-06",
    "pages": 112,
    "retrieved": "2026-08-21",
    "source": "indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf"
  },
  "bnss": {
    "path": "data/raw/A202346.pdf",
    "sha256": "54b27a4f2786dc5867c2cc23391e8359b3b29125684119acbc652d1630a716d6",
    "act_number": "46 of 2023",
    "status": "enacted",
    "as_of_date": "2025-10-06",
    "pages": 282,
    "retrieved": "2026-08-21",
    "source": "indiacode.nic.in/bitstream/123456789/20099/1/A202346.pdf"
  },
  "bns_ipc_mapping": {
    "path": "data/raw/BNS2023.pdf",
    "sha256": "a83f12a93e32c9e0b39a85f75850a448dc4682b1b44b2d6bd680165bb9931549",
    "act_number": "NCRB Sankalan mapping table",
    "status": "enacted",
    "as_of_date": "2025-10-06",
    "pages": 237,
    "retrieved": "2026-08-21",
    "source": "ncrb.gov.in/uploads/SankalanPortal/DownloadPDF/BNS2023.pdf"
  }
}
```

- [ ] **Step 7: Create `config/acts.yaml`**

Every value the parser needs, so BNSS becomes a config entry rather than a rewrite.

```yaml
bns:
  manifest_key: bns
  act: BNS
  index_pages: [3, 15]
  body_start_marker: "BE it enacted"
  heading_font_contains: "Bold"
  heading_min_size: 10.0
  footnote_max_size: 9.5
  expected_section_count: 358
  expected_chapter_count: 20
  max_title_diffs: 8
  min_chunk_chars: 50
  max_chunk_chars: 13000
  max_embed_chars: 20000
  mapping:
    manifest_key: bns_ipc_mapping
    pages: [20, 73]
    target_act: IPC
    expected_min_rows: 330
```

- [ ] **Step 8: Write the failing tests**

`tests/test_manifest.py`:

```python
import json
import pytest
from src.manifest import sha256_file, verify_source, ManifestMismatch


def test_sha256_matches_recorded_hash():
    entry = verify_source("bns")
    assert entry["act_number"] == "45 of 2023"
    assert entry["pages"] == 112


def test_tampered_hash_raises(tmp_path):
    manifest = json.loads(open("data/raw/manifest.json").read())
    manifest["bns"]["sha256"] = "0" * 64
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatch, match="sha256 mismatch"):
        verify_source("bns", manifest_path=str(p))


def test_non_enacted_status_refused(tmp_path):
    manifest = json.loads(open("data/raw/manifest.json").read())
    manifest["bns"]["status"] = "bill"
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatch, match="not enacted"):
        verify_source("bns", manifest_path=str(p))


def test_sha256_of_known_file_is_stable():
    assert sha256_file("data/raw/a202345.pdf").startswith("ff92dcc7")
```

`tests/test_config.py`:

```python
from src.config import load_act_config


def test_bns_config_has_expected_counts():
    cfg = load_act_config("bns")
    assert cfg["expected_section_count"] == 358
    assert cfg["expected_chapter_count"] == 20
    assert cfg["index_pages"] == [3, 15]
    assert cfg["heading_min_size"] == 10.0
```

- [ ] **Step 9: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.manifest'`

- [ ] **Step 10: Write `src/models.py`**

```python
from dataclasses import dataclass, field, asdict


@dataclass
class Section:
    """One chunk. One chunk is always exactly one section — never split."""

    id: str
    act: str
    act_number: str
    status: str
    as_of_date: str
    section_number: str
    section_title: str
    chapter_number: str
    chapter_title: str
    text: str
    illustrations: list = field(default_factory=list)
    maps_to: dict = field(default_factory=dict)
    maps_to_text: str = ""
    illustrations_text: str = ""
    source_page: int = 0
    char_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Section":
        return Section(**d)
```

- [ ] **Step 11: Write `src/config.py`**

```python
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_ACTS_PATH = Path("config/acts.yaml")


def load_act_config(key: str, path: Path = _ACTS_PATH) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if key not in data:
        raise KeyError(f"no act config named {key!r} in {path}")
    return data[key]


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
```

- [ ] **Step 12: Write `src/manifest.py`**

```python
import hashlib
import json
from pathlib import Path

DEFAULT_MANIFEST = "data/raw/manifest.json"


class ManifestMismatch(Exception):
    """Raised when a source PDF does not match its recorded hash or status."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(key: str, manifest_path: str = DEFAULT_MANIFEST) -> dict:
    """Verify a source PDF before any ingest reads it.

    Guards two distinct failure modes: a swapped or corrupted file, and a
    withdrawn bill masquerading as the enacted act. Bills read almost
    identically to the acts they became but carry different section numbers,
    so indexing one would produce confident citations to sections that do not
    legally exist.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if key not in manifest:
        raise ManifestMismatch(f"no manifest entry for {key!r}")
    entry = manifest[key]

    if entry["status"] != "enacted":
        raise ManifestMismatch(
            f"{key}: source status is {entry['status']!r}, not enacted. Refusing to ingest."
        )

    actual = sha256_file(entry["path"])
    if actual != entry["sha256"]:
        raise ManifestMismatch(
            f"{key}: sha256 mismatch. expected {entry['sha256']}, got {actual}"
        )
    return entry
```

- [ ] **Step 13: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: 5 passed (4 in test_manifest.py, 1 in test_config.py)

- [ ] **Step 14: Commit**

```bash
git add .gitignore .env.example requirements.txt config/ data/raw/ src/ tests/ SPEC.md docs/
git commit -m "feat: repo skeleton, act config, and source manifest verification"
```

---

## Task 2: PDF page text with furniture stripping

**Files:**
- Create: `src/pdf_text.py`
- Test: `tests/test_pdf_text.py`

**Interfaces:**
- Consumes: `load_act_config` from Task 1.
- Produces: `page_text(doc, pno, footnote_max_size) -> str`; `body_start_page(doc, marker) -> int`; `joined_body(doc, cfg) -> tuple[str, list[tuple[int, int]], int]` returning `(text, [(char_offset, page_index)], body_start_page_index)`.

**Context:** Exactly two stripping rules, per design doc §3.1. The line-frequency heuristic from the original SPEC.md is deleted because its top hits are `Illustrations.` (42 pages) and `Illustration.` (36 pages) — the delimiters for the highest-value text in the corpus.

- [ ] **Step 1: Write the failing test**

`tests/test_pdf_text.py`:

```python
import pymupdf
import pytest

from src.config import load_act_config
from src.pdf_text import body_start_page, joined_body, page_text


@pytest.fixture(scope="module")
def doc():
    return pymupdf.open("data/raw/a202345.pdf")


@pytest.fixture(scope="module")
def cfg():
    return load_act_config("bns")


def test_body_starts_at_page_16(doc, cfg):
    # 0-indexed 15 == printed page 16, where the enacting formula appears
    assert body_start_page(doc, cfg["body_start_marker"]) == 15


def test_page_number_line_is_stripped(doc, cfg):
    text = page_text(doc, 15, cfg["footnote_max_size"])
    assert not text.lstrip().startswith("16")
    assert "THE BHARATIYA NYAYA SANHITA, 2023" in text


def test_footnote_is_stripped_by_font_size(doc, cfg):
    """The one footnote in the Act sits mid-section, between Explanation 1
    and Explanation 2 of section 2, because it is page-16 furniture and
    section 2 spans that page. A positional rule cannot remove it."""
    text = page_text(doc, 15, cfg["footnote_max_size"])
    assert "1st day of July, 2024" not in text
    assert "S.O. 850(E)" not in text


def test_superscript_footnote_marker_is_stripped(doc, cfg):
    text = page_text(doc, 15, cfg["footnote_max_size"])
    assert "such date as the Central Government" in " ".join(text.split())


def test_illustrations_delimiter_survives(doc, cfg):
    """Guards against reinstating the deleted line-frequency heuristic."""
    text, _, _ = joined_body(doc, cfg)
    assert text.count("Illustrations.") > 30
    assert text.count("Illustration.") > 20


def test_joined_body_offsets_align_to_pages(doc, cfg):
    text, offsets, start = joined_body(doc, cfg)
    assert start == 15
    assert offsets[0] == (0, 15)
    for offset, pno in offsets:
        assert 0 <= offset <= len(text)
    assert [p for _, p in offsets] == list(range(15, doc.page_count))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pdf_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pdf_text'`

- [ ] **Step 3: Write `src/pdf_text.py`**

```python
import re

BARE_NUMBER = re.compile(r"^\s*\d+\s*$")


def page_text(doc, pno: int, footnote_max_size: float = 9.5) -> str:
    """Text of one page with page furniture removed.

    Two rules, and only two:
      1. Drop spans at or below `footnote_max_size`. This removes footnotes
         and superscript reference markers. It must be a span-level font-size
         rule rather than a positional one, because the Act's single footnote
         lands mid-section rather than at the foot of its section.
      2. Drop the first line when it is a bare number. That is the page
         number, the only genuinely repeating furniture in this document.
    """
    lines = []
    for block in doc[pno].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            kept = "".join(
                span["text"]
                for span in line["spans"]
                if span["size"] > footnote_max_size
            )
            if kept.strip():
                lines.append(kept)

    if lines and BARE_NUMBER.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines)


def body_start_page(doc, marker: str) -> int:
    """0-indexed page where the operative text begins.

    The marker separates the Arrangement of Sections index from the body.
    Index lines look exactly like section headings, so parsing the index as
    body would roughly double the section count.
    """
    for pno in range(doc.page_count):
        if marker in doc[pno].get_text():
            return pno
    raise ValueError(f"body start marker {marker!r} not found in document")


def joined_body(doc, cfg: dict):
    """Whole body as one string, plus a char-offset index per page.

    Returns (text, [(char_offset, page_index)], body_start_page_index).
    The offsets let a heading found on a known page be located in the joined
    text without scanning from the top.
    """
    start = body_start_page(doc, cfg["body_start_marker"])
    offsets = []
    parts = []
    pos = 0
    for pno in range(start, doc.page_count):
        text = page_text(doc, pno, cfg["footnote_max_size"])
        offsets.append((pos, pno))
        parts.append(text)
        pos += len(text) + 1  # +1 for the joining newline
    return "\n".join(parts), offsets, start
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pdf_text.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/pdf_text.py tests/test_pdf_text.py
git commit -m "feat: page furniture stripping by font size and bare-number line"
```

---

## Task 3: De-hyphenation with an audited keep-list

**Files:**
- Create: `src/dehyphen.py`
- Test: `tests/test_dehyphen.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_keep_list(text) -> set[str]`; `dehyphenate(text, keep_list) -> tuple[str, list[tuple[str, str]]]` returning `(text, joins)` where each join is `(before, after)`.

**Context:** 6 line-break hyphens in BNS, 184 in BNSS. Genuine legal hyphenates — `house-breaking`, `currency-notes`, `bank-notes`, `power-of-attorney` — must survive. The keep-list is harvested from the corpus itself: any term appearing hyphenated mid-line, where no line break could have caused it, is genuine.

- [ ] **Step 1: Write the failing test**

`tests/test_dehyphen.py`:

```python
from src.dehyphen import build_keep_list, dehyphenate


def test_keep_list_harvests_midline_hyphenates():
    text = "The offence of house-breaking is defined.\nSee currency-notes above."
    keep = build_keep_list(text)
    assert "house-breaking" in keep
    assert "currency-notes" in keep


def test_keep_list_ignores_linebreak_hyphens():
    """A hyphen at a line break is evidence of splitting, not of a real
    hyphenate, so it must not seed the keep-list."""
    text = "counter-\nfeit coin"
    assert build_keep_list(text) == set()


def test_split_word_is_joined():
    text = "counter-\nfeit coin"
    out, joins = dehyphenate(text, build_keep_list(text))
    assert "counterfeit coin" in out
    assert joins == [("counter- feit", "counterfeit")]


def test_genuine_hyphenate_split_across_lines_is_preserved():
    """Section 179 renders 'bank-notes' as 'bank-\\nnotes'. Because
    'bank-notes' appears hyphenated mid-line elsewhere in the Act, the
    hyphen must be restored rather than removed."""
    text = "Possession of bank-notes here.\nUsing forged bank-\nnotes elsewhere."
    out, joins = dehyphenate(text, build_keep_list(text))
    assert "bank-notes elsewhere" in " ".join(out.split())
    assert ("bank-notes", "bank-notes") in joins


def test_three_part_hyphenate_survives_a_split_at_either_hyphen():
    """'power-of-attorney' must survive being broken at EITHER hyphen. A
    non-overlapping keep-list harvest sees only 'power-of' and would silently
    join the second break into 'power-ofattorney'."""
    text = (
        "A power-of-attorney is a document.
"
        "First break: power-
of-attorney here.
"
        "Second break: power-of-
attorney there.
"
    )
    keep = build_keep_list(text)
    assert "power-of" in keep
    assert "of-attorney" in keep
    out, _ = dehyphenate(text, keep)
    assert "power-ofattorney" not in out
    assert "powerof-attorney" not in out


def test_every_join_is_logged():
    text = "counter-\nfeit and inter-\nnational"
    _, joins = dehyphenate(text, build_keep_list(text))
    assert len(joins) == 2


def test_real_corpus_join_count_is_small_and_reviewable():
    import pymupdf
    from src.config import load_act_config
    from src.pdf_text import joined_body

    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    body, _, _ = joined_body(doc, cfg)
    _, joins = dehyphenate(body, build_keep_list(body))
    assert len(joins) < 20, f"unexpected join volume: {joins}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dehyphen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dehyphen'`

- [ ] **Step 3: Write `src/dehyphen.py`**

```python
import re

# A hyphen with letters on both sides and no newline between them. Because
# the pattern contains no newline, it can only match within a single line,
# which is exactly what makes it evidence of a genuine hyphenate.
#
# Wrapped in a zero-width lookahead so matches can OVERLAP. Without it,
# findall consumes 'power-of' and never sees 'of-attorney', so a three-part
# hyphenate split at its second hyphen would be silently joined into
# 'power-ofattorney' -- altering statutory text with no error.
MIDLINE_HYPHEN = re.compile(r"(?=\b([A-Za-z]{2,})-([A-Za-z]{2,})\b)")

# A hyphen immediately before a line break.
LINEBREAK_HYPHEN = re.compile(r"\b([A-Za-z]{2,})-[ \t]*\n[ \t]*([A-Za-z]{2,})\b")


def build_keep_list(text: str) -> set:
    """Terms that are genuinely hyphenated, harvested from the corpus itself.

    Built from mid-line occurrences only. A term that appears hyphenated in
    running text cannot owe its hyphen to a line break, so it is real.
    """
    return {f"{a}-{b}".lower() for a, b in MIDLINE_HYPHEN.findall(text)}


def dehyphenate(text: str, keep_list: set):
    """Join words split across a line break, preserving genuine hyphenates.

    Returns (text, joins). Every decision is recorded in `joins` as
    (before, after) so the keep-list can be audited rather than trusted.
    """
    joins = []

    def replace(match):
        left, right = match.group(1), match.group(2)
        hyphenated = f"{left}-{right}"
        if hyphenated.lower() in keep_list:
            joins.append((hyphenated, hyphenated))
            return hyphenated
        joins.append((f"{left}- {right}", left + right))
        return left + right

    return LINEBREAK_HYPHEN.sub(replace, text), joins
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dehyphen.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/dehyphen.py tests/test_dehyphen.py
git commit -m "feat: de-hyphenation with corpus-harvested keep-list and join log"
```

---

## Task 4: The index oracle

**Files:**
- Create: `src/parse_index.py`
- Test: `tests/test_parse_index.py`

**Interfaces:**
- Consumes: `load_act_config` from Task 1.
- Produces: `parse_index(doc, cfg) -> dict[int, str]` mapping section number to title.

**Context:** The Arrangement of Sections on pages 3–15 yields exactly 358 entries, contiguous and unique. It is the validation oracle for the body parser. Note that the index is not infallible — §330 reads `hous-ebreaking` where the body correctly reads `house-breaking`. Body text wins; the oracle checks the count and flags diffs for review.

- [ ] **Step 1: Write the failing test**

`tests/test_parse_index.py`:

```python
import pymupdf
import pytest

from src.config import load_act_config
from src.parse_index import parse_index


@pytest.fixture(scope="module")
def index():
    doc = pymupdf.open("data/raw/a202345.pdf")
    return parse_index(doc, load_act_config("bns"))


def test_index_has_exactly_358_entries(index):
    assert len(index) == 358


def test_index_is_contiguous_and_unique(index):
    assert set(index.keys()) == set(range(1, 359))


def test_known_titles(index):
    assert index[1] == "Short title, commencement and application"
    assert index[303] == "Theft"
    assert index[318] == "Cheating"
    assert index[358] == "Repeal and savings"


def test_wrapped_title_is_reassembled(index):
    """Section 10's title wraps across two lines in the index."""
    assert index[10].startswith("Punishment of person guilty of one of several offences")
    assert index[10].endswith("doubtful of which")


def test_index_typo_is_preserved_not_corrected(index):
    """The PDF's own index misspells section 330. The oracle reports what
    the index says; reconciliation against the body happens in ingest."""
    assert index[330] == "House-trespass and hous-ebreaking"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parse_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parse_index'`

- [ ] **Step 3: Write `src/parse_index.py`**

```python
import re

ENTRY = re.compile(r"^(\d+)\.\s+(.*)$")


def _is_continuation(line: str) -> bool:
    """True when a line continues the previous index entry's title.

    Index pages interleave entries with structural furniture: chapter
    headings in caps, the word SECTIONS, sub-headings like 'Of defamation',
    and bare page numbers. None of those continue a title.
    """
    if line.isupper():
        return False
    if re.fullmatch(r"\d+", line):
        return False
    if line.startswith("Of "):
        return False
    return True


def parse_index(doc, cfg: dict) -> dict:
    """Parse the Arrangement of Sections into {section_number: title}.

    This is the validation oracle for the body parser. It is authoritative
    for the *set* of section numbers, and advisory for titles.
    """
    lo, hi = cfg["index_pages"]
    entries = []
    for pno in range(lo - 1, hi):
        for raw in doc[pno].get_text().split("\n"):
            line = raw.strip()
            if not line:
                continue
            match = ENTRY.match(line)
            if match:
                entries.append([int(match.group(1)), match.group(2)])
            elif entries and _is_continuation(line):
                entries[-1][1] += " " + line

    return {num: " ".join(title.split()).rstrip(".") for num, title in entries}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parse_index.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/parse_index.py tests/test_parse_index.py
git commit -m "feat: Arrangement of Sections index parser as validation oracle"
```

---

## Task 5: Bold-span section heading parser

**Files:**
- Create: `src/parse_sections.py`
- Test: `tests/test_parse_sections.py`

**Interfaces:**
- Consumes: `body_start_page` from Task 2, `load_act_config` from Task 1.
- Produces: `parse_headings(doc, cfg, body_start) -> list[tuple[int, str, int]]` as `(section_number, title, printed_page)`.

**Context:** Read design doc §2.3 before writing this. The regex proposed in the original SPEC.md matched 2 of 358 sections. Headings are set in a bold font and body text is not; that is the reliable signal. The regex here parses an *already-identified heading run*, never raw page text.

- [ ] **Step 1: Write the failing test**

`tests/test_parse_sections.py`:

```python
import pymupdf
import pytest

from src.config import load_act_config
from src.parse_sections import merge_runs, parse_headings
from src.pdf_text import body_start_page


@pytest.fixture(scope="module")
def headings():
    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    return parse_headings(doc, cfg, body_start_page(doc, cfg["body_start_marker"]))


def test_finds_exactly_358_headings(headings):
    assert len(headings) == 358


def test_headings_are_unique_and_monotonic(headings):
    numbers = [n for n, _, _ in headings]
    assert len(set(numbers)) == 358
    assert numbers == sorted(numbers)
    assert numbers == list(range(1, 359))


def test_known_titles(headings):
    by_number = {n: t for n, t, _ in headings}
    assert by_number[303] == "Theft"
    assert by_number[318] == "Cheating"


def test_wrapped_title_is_reassembled(headings):
    """Section 10 places its em-dash on the following line."""
    by_number = {n: t for n, t, _ in headings}
    assert by_number[10].startswith("Punishment of person guilty")


def test_dash_before_title_heading_form(headings):
    """Section 255 is '255.-Title', with the dash before the title rather
    than after it. No title-then-dash pattern can match it."""
    by_number = {n: t for n, t, _ in headings}
    assert by_number[255].startswith("Public servant disobeying direction of law")


def test_page_numbers_are_plausible(headings):
    by_number = {n: p for n, _, p in headings}
    assert by_number[303] == 88
    assert by_number[1] == 16


def test_merge_runs_does_not_swallow_the_next_section():
    """A run that starts with 'N.' always begins a new heading, even when
    the previous run did not end with a dash."""
    runs = ["5. Commutation of sentence.", "6. Fractions of terms of punishment.—"]
    assert merge_runs(runs) == runs


def test_merge_runs_joins_a_wrapped_title():
    runs = ["253. Harbouring offender who has escaped from custody", "or whose apprehension has been ordered.—"]
    merged = merge_runs(runs)
    assert len(merged) == 1
    assert "apprehension has been ordered" in merged[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parse_sections.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parse_sections'`

- [ ] **Step 3: Write `src/parse_sections.py`**

```python
import re

DASHES = "–—"  # en dash, em dash
NUM_START = re.compile(r"^\s*\d+\.")
HEADING = re.compile(r"^(\d+)\.\s*[%s]?\s*(.*)$" % DASHES)


def bold_runs(doc, pno: int, cfg: dict) -> list:
    """Consecutive bold spans on a page, merged into runs.

    Headings are the only bold text in the body at this size. The font name
    and minimum size come from config so a differently-rendered act is a
    config change rather than a parser rewrite.
    """
    runs = []
    current = ""
    for block in doc[pno].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                is_heading_font = (
                    cfg["heading_font_contains"] in span["font"]
                    and span["size"] >= cfg["heading_min_size"]
                )
                if is_heading_font:
                    current += span["text"]
                elif current.strip():
                    runs.append(current)
                    current = ""
            if current.strip():
                runs.append(current)
                current = ""
    if current.strip():
        runs.append(current)
    return [" ".join(run.split()) for run in runs if run.strip()]


def merge_runs(runs: list) -> list:
    """Reassemble headings whose titles wrap across lines.

    A run beginning with 'N.' always starts a new heading. Any other run
    continues the previous one, unless the previous one already ended with a
    dash — the dash marks the end of a title.
    """
    merged = []
    for run in runs:
        if NUM_START.match(run) or not merged:
            merged.append(run)
        elif not merged[-1].rstrip().endswith(tuple(DASHES)):
            merged[-1] += " " + run
        else:
            merged.append(run)
    return merged


def parse_headings(doc, cfg: dict, body_start: int) -> list:
    """All section headings as (section_number, title, printed_page)."""
    headings = []
    for pno in range(body_start, doc.page_count):
        for run in merge_runs(bold_runs(doc, pno, cfg)):
            match = HEADING.match(run)
            if not match:
                continue
            title = match.group(2).strip().rstrip(DASHES).strip().rstrip(".")
            headings.append(
                (int(match.group(1)), " ".join(title.split()), pno + 1)
            )
    return headings
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parse_sections.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/parse_sections.py tests/test_parse_sections.py
git commit -m "feat: bold-span section heading parser, 358/358 on BNS"
```

---

## Task 6: Chapter parser

**Files:**
- Create: `src/parse_chapters.py`
- Test: `tests/test_parse_chapters.py`

**Interfaces:**
- Consumes: `body_start_page` from Task 2.
- Produces: `parse_chapters(doc, body_start) -> list[tuple[str, str, int]]` as `(roman_numeral, title, printed_page)`; `chapter_for_page(chapters, page) -> tuple[str, str]`.

- [ ] **Step 1: Write the failing test**

`tests/test_parse_chapters.py`:

```python
import pymupdf
import pytest

from src.config import load_act_config
from src.parse_chapters import chapter_for_page, parse_chapters
from src.pdf_text import body_start_page


@pytest.fixture(scope="module")
def chapters():
    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    return parse_chapters(doc, body_start_page(doc, cfg["body_start_marker"]))


def test_finds_20_chapters(chapters):
    assert len(chapters) == 20


def test_first_and_last_chapters(chapters):
    assert chapters[0] == ("I", "PRELIMINARY", 16)
    assert chapters[-1] == ("XX", "REPEAL AND SAVINGS", 110)


def test_property_offences_chapter(chapters):
    numerals = {c[0]: c for c in chapters}
    assert numerals["XVII"][1] == "OF OFFENCES AGAINST PROPERTY"
    assert numerals["XVII"][2] == 88


def test_chapter_for_page_assigns_theft_correctly(chapters):
    """Section 303 Theft is on page 88, in Chapter XVII."""
    number, title = chapter_for_page(chapters, 88)
    assert number == "XVII"
    assert title == "OF OFFENCES AGAINST PROPERTY"


def test_chapter_for_page_uses_the_most_recent_chapter(chapters):
    number, _ = chapter_for_page(chapters, 95)
    assert number == "XVII"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parse_chapters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parse_chapters'`

- [ ] **Step 3: Write `src/parse_chapters.py`**

```python
import re

CHAPTER = re.compile(r"^CHAPTER\s+([IVXL]+)$")


def parse_chapters(doc, body_start: int) -> list:
    """All chapter headings as (roman_numeral, title, printed_page).

    The title is the next all-caps line after the CHAPTER line. Chapter
    titles may wrap, so consecutive caps lines are joined.
    """
    chapters = []
    for pno in range(body_start, doc.page_count):
        lines = [line.strip() for line in doc[pno].get_text().split("\n")]
        for i, line in enumerate(lines):
            match = CHAPTER.match(line)
            if not match:
                continue
            title_parts = []
            for candidate in lines[i + 1 : i + 5]:
                if not candidate:
                    continue
                if not candidate.isupper():
                    break
                title_parts.append(candidate)
            chapters.append(
                (match.group(1), " ".join(title_parts), pno + 1)
            )
    return chapters


def chapter_for_page(chapters: list, page: int):
    """The chapter in force on a given printed page.

    Returns (roman_numeral, title). Chapters are in document order, so the
    answer is the last chapter whose heading appears at or before `page`.
    """
    current = ("", "")
    for numeral, title, chapter_page in chapters:
        if chapter_page <= page:
            current = (numeral, title)
        else:
            break
    return current
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_parse_chapters.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/parse_chapters.py tests/test_parse_chapters.py
git commit -m "feat: chapter heading parser, 20/20 on BNS"
```

---

## Task 7: Section assembly and `sections.json`

**Files:**
- Create: `src/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `locate_headings(joined, offsets, headings) -> list[int]`; `extract_illustrations(text) -> list[str]`; `build_sections(act_key) -> list[Section]`; `validate(sections, index, cfg) -> tuple[list[str], list[tuple]]`; `main()` writing `data/processed/sections.json`.

**Context:** This is where the chunk invariants are enforced. Read the "Decisions this plan makes" section above on the 13,000-character ceiling before implementing `validate`.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:

```python
import json

import pytest

from src.ingest import build_sections, extract_illustrations, validate
from src.config import load_act_config
from src.parse_index import parse_index


@pytest.fixture(scope="module")
def sections():
    return build_sections("bns")


def test_exactly_358_sections(sections):
    assert len(sections) == 358


def test_no_chunk_below_minimum(sections):
    cfg = load_act_config("bns")
    short = [s.section_number for s in sections if s.char_count < cfg["min_chunk_chars"]]
    assert short == []


def test_no_chunk_above_maximum(sections):
    cfg = load_act_config("bns")
    long = [s.section_number for s in sections if s.char_count > cfg["max_chunk_chars"]]
    assert long == []


def test_definitions_section_is_kept_whole(sections):
    """Section 2 is 12,904 characters. SPEC.md's original 6000 ceiling would
    have forced a split, violating the more important rule that a section is
    never split. The ceiling moved; the section stays whole."""
    section_2 = next(s for s in sections if s.section_number == "2")
    assert section_2.char_count > 12000
    assert "wrongful loss" in section_2.text
    assert "words and expressions used but not defined" in section_2.text


def test_theft_section_shape(sections):
    theft = next(s for s in sections if s.section_number == "303")
    assert theft.id == "bns-303"
    assert theft.section_title == "Theft"
    assert theft.chapter_number == "XVII"
    assert theft.chapter_title == "OF OFFENCES AGAINST PROPERTY"
    assert theft.source_page == 88
    assert theft.act_number == "45 of 2023"
    assert theft.status == "enacted"
    assert theft.as_of_date == "2025-10-06"


def test_theft_illustrations_are_extracted(sections):
    theft = next(s for s in sections if s.section_number == "303")
    assert len(theft.illustrations) >= 7
    joined = " ".join(theft.illustrations)
    assert "cuts down a tree" in joined
    assert "finds a ring belonging to Z" in joined


def test_illustrations_stay_in_body_text_too(sections):
    """Illustrations are the highest-value text for semantic matching, so
    they are duplicated into the array rather than moved out of the body."""
    theft = next(s for s in sections if s.section_number == "303")
    assert "finds a ring belonging to Z" in theft.text


def test_section_text_does_not_leak_into_the_next_section(sections):
    theft = next(s for s in sections if s.section_number == "303")
    assert "304." not in theft.text
    assert "Snatching" not in theft.text


def test_footnote_absent_from_section_2(sections):
    section_2 = next(s for s in sections if s.section_number == "2")
    assert "1st day of July, 2024" not in section_2.text


def test_extract_illustrations_splits_lettered_items():
    text = (
        "Some body text.\n"
        "Illustrations.\n"
        "(a) A does a thing.\n"
        "(b) B does another thing.\n"
    )
    items = extract_illustrations(text)
    assert len(items) == 2
    assert items[0].startswith("(a) A does a thing")


def test_extract_illustrations_stops_at_explanation():
    text = (
        "Illustration.\n"
        "(a) A does a thing.\n"
        "Explanation.-Something else entirely.\n"
    )
    items = extract_illustrations(text)
    assert len(items) == 1
    assert "Something else entirely" not in items[0]


def test_validate_reports_index_diffs_not_errors(sections):
    import pymupdf

    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    index = parse_index(doc, cfg)
    errors, diffs = validate(sections, index, cfg)
    assert errors == []
    assert len(diffs) <= cfg["max_title_diffs"]


def test_body_wins_over_index_typo(sections):
    """Index says 'hous-ebreaking'; the body says 'house-breaking'."""
    section = next(s for s in sections if s.section_number == "330")
    assert "house-breaking" in section.section_title
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingest'`

- [ ] **Step 3: Write `src/ingest.py`**

```python
import json
import logging
import re
from pathlib import Path

import pymupdf

from src.config import load_act_config
from src.dehyphen import build_keep_list, dehyphenate
from src.manifest import verify_source
from src.models import Section
from src.parse_chapters import chapter_for_page, parse_chapters
from src.parse_index import parse_index
from src.parse_sections import parse_headings
from src.pdf_text import body_start_page, joined_body

log = logging.getLogger(__name__)

ILLUSTRATION_HEADING = re.compile(r"(?m)^\s*Illustrations?\.\s*$")
ILLUSTRATION_STOP = re.compile(r"(?m)^\s*(Explanation|Exception)\b")
ILLUSTRATION_ITEM = re.compile(r"(?m)^(?=\([a-z]\)\s)")

OUTPUT = Path("data/processed/sections.json")


def _find_heading(joined: str, number: int, title: str, start: int) -> int:
    """Char offset of one section heading in the joined body.

    Searching for the bare number is not enough — cross-references such as
    'punishable under section 303' would match. The candidate is accepted
    only when the text that follows it looks like the heading's own title.
    """
    pattern = re.compile(r"(?<![\d.])%d\." % number)
    probe = " ".join(title.split()).lower()[:12]
    for match in pattern.finditer(joined, start):
        tail = " ".join(joined[match.end() : match.end() + 80].split())
        tail = tail.lower().lstrip("–— ")
        if tail.startswith(probe):
            return match.start()
    raise ValueError(f"could not locate heading for section {number} ({title!r})")


def locate_headings(joined: str, offsets: list, headings: list) -> list:
    """Char offset of every heading, in document order."""
    page_start = {pno: offset for offset, pno in offsets}
    positions = []
    cursor = 0
    for number, title, printed_page in headings:
        # printed_page is 1-indexed; page_start is keyed by 0-indexed page
        floor = max(cursor, page_start.get(printed_page - 1, 0))
        position = _find_heading(joined, number, title, floor)
        positions.append(position)
        cursor = position + 1
    return positions


def extract_illustrations(text: str) -> list:
    """The 'A does X to B' examples in the statute.

    These read like real-world situations, which is exactly how ordinary
    users phrase questions, making them the highest-value text in the corpus
    for semantic matching. They are extracted into their own field AND left
    in the body text.
    """
    items = []
    for heading in ILLUSTRATION_HEADING.finditer(text):
        segment = text[heading.end() :]
        stop = ILLUSTRATION_STOP.search(segment)
        if stop:
            segment = segment[: stop.start()]
        segment = segment.strip()
        if not segment:
            continue
        for part in ILLUSTRATION_ITEM.split(segment):
            normalised = " ".join(part.split())
            if normalised:
                items.append(normalised)
    return items


def build_sections(act_key: str) -> list:
    entry = verify_source(act_key)
    cfg = load_act_config(act_key)
    doc = pymupdf.open(entry["path"])

    body_start = body_start_page(doc, cfg["body_start_marker"])
    joined, offsets, _ = joined_body(doc, cfg)

    keep_list = build_keep_list(joined)
    joined, joins = dehyphenate(joined, keep_list)
    for before, after in joins:
        log.info("dehyphenate: %r -> %r", before, after)

    headings = parse_headings(doc, cfg, body_start)
    chapters = parse_chapters(doc, body_start)
    positions = locate_headings(joined, offsets, headings)

    sections = []
    for i, (number, title, printed_page) in enumerate(headings):
        end = positions[i + 1] if i + 1 < len(positions) else len(joined)
        text = joined[positions[i] : end].strip()
        illustrations = extract_illustrations(text)
        chapter_number, chapter_title = chapter_for_page(chapters, printed_page)
        sections.append(
            Section(
                id=f"{cfg['act'].lower()}-{number}",
                act=cfg["act"],
                act_number=entry["act_number"],
                status=entry["status"],
                as_of_date=entry["as_of_date"],
                section_number=str(number),
                section_title=title,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                text=text,
                illustrations=illustrations,
                illustrations_text=" ".join(illustrations),
                source_page=printed_page,
                char_count=len(text),
            )
        )
    return sections


def _normalise(value: str) -> str:
    value = value.replace("’", "'").replace("‘", "'")
    return " ".join(value.split()).rstrip(".").lower()


def validate(sections: list, index: dict, cfg: dict):
    """Hard invariants and advisory diffs.

    Returns (errors, diffs). Errors are fatal. Diffs are title disagreements
    between body and index, which are expected in small numbers and are
    logged individually so a real extraction fault cannot hide among the
    cosmetic ones.
    """
    errors = []
    diffs = []

    expected = cfg["expected_section_count"]
    if len(sections) != expected:
        errors.append(f"expected {expected} sections, got {len(sections)}")

    numbers = [int(s.section_number) for s in sections]
    if numbers != sorted(numbers):
        errors.append("section numbers are not monotonic in document order")
    if len(set(numbers)) != len(numbers):
        errors.append("duplicate section numbers")
    if set(numbers) != set(range(1, expected + 1)):
        missing = sorted(set(range(1, expected + 1)) - set(numbers))
        errors.append(f"section numbers not contiguous; missing {missing[:20]}")

    for section in sections:
        if section.char_count < cfg["min_chunk_chars"]:
            errors.append(
                f"section {section.section_number} is {section.char_count} chars, "
                f"below minimum {cfg['min_chunk_chars']}"
            )
        if section.char_count > cfg["max_chunk_chars"]:
            errors.append(
                f"section {section.section_number} is {section.char_count} chars, "
                f"above maximum {cfg['max_chunk_chars']}"
            )
        if section.char_count > cfg["max_embed_chars"]:
            errors.append(
                f"section {section.section_number} is {section.char_count} chars, "
                f"which risks silent truncation by the embedding model"
            )
        index_title = index.get(int(section.section_number))
        if index_title and _normalise(index_title) != _normalise(section.section_title):
            diffs.append((section.section_number, index_title, section.section_title))

    if len(diffs) > cfg["max_title_diffs"]:
        errors.append(
            f"{len(diffs)} title diffs against the index, above the "
            f"{cfg['max_title_diffs']} allowed. The parser is probably wrong."
        )
    return errors, diffs


def main(act_key: str = "bns") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_act_config(act_key)
    entry = verify_source(act_key)

    sections = build_sections(act_key)
    index = parse_index(pymupdf.open(entry["path"]), cfg)
    errors, diffs = validate(sections, index, cfg)

    for number, index_title, body_title in diffs:
        log.warning(
            "title diff section %s | index: %r | body: %r (body wins)",
            number, index_title, body_title,
        )

    if errors:
        for error in errors:
            log.error("%s", error)
        raise SystemExit(f"ingest failed with {len(errors)} error(s)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps([s.to_dict() for s in sections], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("wrote %d sections to %s (%d title diffs)", len(sections), OUTPUT, len(diffs))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ingest.py -v`
Expected: 13 passed

- [ ] **Step 5: Run ingest and review the diff log by hand**

Run: `.venv/Scripts/python.exe -m src.ingest`
Expected: `wrote 358 sections to data/processed/sections.json (6 title diffs)`

Read every logged diff. All six should be cosmetic — apostrophes on §26 and §89, trailing punctuation on §44, hyphenation on §179 and §180, and the index's own typo on §330. **If any diff is not cosmetic, stop and fix the parser.** This review is an acceptance criterion, not a formality.

Also review the de-hyphenation joins:

```bash
.venv/Scripts/python.exe -m src.ingest 2>&1 | grep dehyphenate
```

Expected: fewer than 20 lines, each an obvious word split.

- [ ] **Step 6: Print five random sections for the human reviewer — DO NOT MARK COMPLETE**

> **Reserved for the human.** Run the command and leave the output for them. The implementing agent must not tick this box and must not claim the verification happened. It is a Phase 1 acceptance criterion from SPEC.md §7, and only a person comparing against the PDF can satisfy it.

```bash
.venv/Scripts/python.exe -c "
import json, random
s = json.load(open('data/processed/sections.json', encoding='utf-8'))
random.seed()
for sec in random.sample(s, 5):
    print('='*70)
    print(sec['section_number'], '|', sec['section_title'], '| page', sec['source_page'])
    print(sec['text'][:700])
"
```

Open `data/raw/a202345.pdf` at each reported page and compare word for word. This is a Phase 1 acceptance criterion from SPEC.md §7.

- [ ] **Step 7: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: section assembly with chunk invariants and index reconciliation"
```

---

## Task 8: NCRB mapping table parser

**Files:**
- Create: `src/mapping.py`
- Test: `tests/test_mapping.py`

**Interfaces:**
- Consumes: `verify_source`, `load_act_config` from Task 1.
- Produces: `parse_mappings(act_key) -> dict[str, list[str]]` mapping BNS section number to IPC section numbers; `mapping_text(ipc_sections, target_act) -> str`; `main()` writing `data/processed/mappings.json`.

**Context:** Pages 20–73 of `BNS2023.pdf`. `page.find_tables()` succeeds on 53 of the 54 pages and returns a consistent two-column layout: BNS on the left, IPC on the right. Two traps found while probing:

1. **Continuation rows carry no section number.** BNS 318's mapping to IPC 420 lives in a row whose left cell reads `318 (4)`, not `318.`. A parser requiring `^\d+\.` silently drops it — and IPC 420 is a Phase 1 acceptance criterion. The parser must carry the current section forward across rows.
2. **The right cell is narrowly wrapped.** IPC 420 renders as `420. \nCheating \nand \ndishonestly \ninducing delivery of property.` Number extraction must tolerate that.

- [ ] **Step 1: Write the failing test**

`tests/test_mapping.py`:

```python
import pytest

from src.mapping import mapping_text, parse_mappings


@pytest.fixture(scope="module")
def mappings():
    return parse_mappings("bns")


def test_covers_most_sections(mappings):
    assert len(mappings) >= 330


def test_theft_maps_to_ipc_378_and_379(mappings):
    """SPEC.md section 4 uses exactly this as its worked example."""
    assert mappings["303"] == ["378", "379"]


def test_cheating_includes_ipc_420(mappings):
    """IPC 420 lives in a continuation row whose left cell reads '318 (4)',
    with no section number of its own. A parser that requires a leading
    'NNN.' drops it, and 'what is IPC 420 now' is an acceptance criterion."""
    assert "420" in mappings["318"]


def test_ipc_numbers_are_strings_without_duplicates(mappings):
    for bns, ipc in mappings.items():
        assert all(isinstance(n, str) for n in ipc)
        assert len(ipc) == len(set(ipc))


def test_no_placeholder_text_leaks_into_numbers(mappings):
    flat = [n for ipc in mappings.values() for n in ipc]
    assert "New" not in flat
    assert "Deleted" not in flat


def test_mapping_text_is_a_flat_searchable_string():
    assert mapping_text(["378", "379"], "IPC") == "IPC 378 IPC 379"


def test_mapping_text_is_empty_for_new_sections():
    assert mapping_text([], "IPC") == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mapping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.mapping'`

- [ ] **Step 3: Write `src/mapping.py`**

```python
import json
import logging
import re
from pathlib import Path

import pymupdf

from src.config import load_act_config
from src.manifest import verify_source

log = logging.getLogger(__name__)

OUTPUT = Path("data/processed/mappings.json")

# A left cell that declares a new BNS section: '303. Theft.'
SECTION_DECLARATION = re.compile(r"^\s*(\d+)\s*\.")
# A left cell that continues one: '318 (4)'
SECTION_CONTINUATION = re.compile(r"^\s*(\d+)\s*\(")
# An IPC section number in the right cell. The cell is narrowly wrapped, so
# the number may be followed by a newline rather than a space.
IPC_NUMBER = re.compile(r"(?:^|\n)\s*(\d+[A-Z]?)\s*\.")


def parse_mappings(act_key: str) -> dict:
    """BNS section number -> list of IPC section numbers.

    Sourced from the NCRB correspondence table, not from the gazette. The
    gazette never mentions the IPC at all, which is why migration queries
    need this table to be searchable.
    """
    cfg = load_act_config(act_key)
    mapping_cfg = cfg["mapping"]
    entry = verify_source(mapping_cfg["manifest_key"])
    doc = pymupdf.open(entry["path"])

    lo, hi = mapping_cfg["pages"]
    mappings = {}
    current = None

    for pno in range(lo - 1, hi):
        tables = doc[pno].find_tables()
        if not tables.tables:
            continue
        for row in tables.tables[0].extract():
            if len(row) < 2:
                continue
            left = (row[0] or "").strip()
            right = (row[1] or "").strip()

            declaration = SECTION_DECLARATION.match(left)
            if declaration:
                current = declaration.group(1)
                mappings.setdefault(current, [])
            else:
                continuation = SECTION_CONTINUATION.match(left)
                if continuation:
                    current = continuation.group(1)
                    mappings.setdefault(current, [])

            if current is None or not right:
                continue
            for number in IPC_NUMBER.findall(right):
                if number not in mappings[current]:
                    mappings[current].append(number)

    return mappings


def mapping_text(ipc_sections: list, target_act: str) -> str:
    """Flatten a mapping into a searchable string: 'IPC 378 IPC 379'.

    Stored as its own plain column so it can be inspected with a SELECT and
    diffed against the source table by eye. Building it inline inside a
    generated-column expression would populate without erroring even when
    subtly wrong, and the only symptom would be migration queries quietly
    retrieving nothing.
    """
    return " ".join(f"{target_act} {number}" for number in ipc_sections)


def main(act_key: str = "bns") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_act_config(act_key)
    mappings = parse_mappings(act_key)

    minimum = cfg["mapping"]["expected_min_rows"]
    if len(mappings) < minimum:
        raise SystemExit(
            f"parsed only {len(mappings)} mapping rows, expected at least {minimum}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("wrote %d mappings to %s", len(mappings), OUTPUT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mapping.py -v`
Expected: 7 passed

- [ ] **Step 5: Print twenty mappings for the human reviewer — DO NOT MARK COMPLETE**

> **Reserved for the human.** Run the commands and leave the output for them. The implementing agent must not tick this box. Verifying against NCRB pages 20–73 is a SPEC.md §10 acceptance criterion and requires a person reading the printed table.

Run: `.venv/Scripts/python.exe -m src.mapping`

```bash
.venv/Scripts/python.exe -c "
import json
m = json.load(open('data/processed/mappings.json', encoding='utf-8'))
for k in ['1','2','101','103','111','115','118','303','304','305','306','309','316','318','319','324','351','352','356','358']:
    print(f'BNS {k:>4} -> IPC {m.get(k)}')
"
```

Open `data/raw/BNS2023.pdf` at pages 20–73 and check each of these twenty against the printed table. This is a SPEC.md §10 acceptance criterion.

- [ ] **Step 6: Commit**

```bash
git add src/mapping.py tests/test_mapping.py
git commit -m "feat: NCRB BNS-to-IPC mapping parser with continuation-row handling"
```

---

## STOP — pause here for the human

**Do not begin Task 9.** Tasks 1–8 run with no secrets, no database, and no API calls. Nothing past this point can start until:

1. The `nyaya-dev` Neon project exists and its connection string is in `.env` as `DATABASE_URL`.
2. `GEMINI_API_KEY` is in `.env`.
3. The human has completed the two reserved by-hand reviews — five sections against the gazette PDF, twenty mappings against NCRB pages 20–73.

The third condition is the one that matters. Task 11 embeds all 358 sections and writes them to the index. Embedding a corpus whose parser no person has eyeballed means indexing text nobody has checked, and the by-hand review is the only check that catches a parser which is confidently wrong rather than obviously broken. Automated checks confirm the *shape* is right; only a human comparing against the PDF confirms the *content* is.

Report status and wait.

---

## Task 9: Database schema

**Files:**
- Create: `src/db.py`, `sql/schema.sql`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `require_env` from Task 1.
- Produces: `connect()` context manager yielding a psycopg connection with pgvector registered; `create_schema(conn)`; `drop_schema(conn)`.

**Context:** Plain Postgres and pgvector only — no Neon-specific features, so a later move to self-hosted is not blocked. The `fts` generated column reads from plain text columns rather than transforming `maps_to` jsonb inline; see design doc §5 for why.

- [ ] **Step 1: Set up the Neon dev database**

Create a Neon project named `nyaya-dev`. Copy its connection string into `.env` as `DATABASE_URL`. It must include `?sslmode=require`.

Enable the extension once:

```bash
.venv/Scripts/python.exe -c "
import psycopg, os
from dotenv import load_dotenv
load_dotenv()
with psycopg.connect(os.environ['DATABASE_URL']) as c:
    c.execute('CREATE EXTENSION IF NOT EXISTS vector')
    c.commit()
    print(c.execute('SELECT extversion FROM pg_extension WHERE extname=%s', ('vector',)).fetchone())
"
```

Expected: a version string such as `('0.8.0',)`

- [ ] **Step 2: Write `sql/schema.sql`**

```sql
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
```

- [ ] **Step 3: Write the failing test**

`tests/test_db.py`:

```python
import pytest

from src.db import connect, create_schema

# Every test here needs a live database.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        create_schema(connection)
        yield connection


def test_pgvector_is_available(conn):
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None


def test_all_tables_exist(conn):
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {
        "sections", "embeddings", "queries", "retrievals",
        "answers", "eval_runs", "eval_results",
    } <= names


def test_embedding_dimension_is_768(conn):
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'embeddings'::regclass AND attname = 'vector'"
    ).fetchone()
    assert row[0] == 768


def test_non_enacted_status_is_rejected_by_the_database(conn):
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO sections (id, act, act_number, status, as_of_date,"
            " section_number, section_title, chapter_number, chapter_title,"
            " text, source_page, char_count)"
            " VALUES ('x-1','X','1 of 2023','bill','2025-10-06','1','T','I','C','body',1,4)"
        )
    conn.rollback()


def test_fts_includes_mapping_tokens(conn):
    conn.execute(
        "INSERT INTO sections (id, act, act_number, status, as_of_date,"
        " section_number, section_title, chapter_number, chapter_title,"
        " text, maps_to_text, source_page, char_count)"
        " VALUES ('test-1','TEST','1 of 2023','enacted','2025-10-06','1',"
        " 'Test','I','Chapter','some body text','IPC 420',1,14)"
        " ON CONFLICT (id) DO NOTHING"
    )
    row = conn.execute(
        "SELECT id FROM sections WHERE fts @@ plainto_tsquery('english', 'IPC 420')"
    ).fetchone()
    assert row is not None
    conn.execute("DELETE FROM sections WHERE id = 'test-1'")
    conn.commit()


def test_queries_text_is_nullable_for_sensitive_routes(conn):
    import uuid

    qid = uuid.uuid4()
    conn.execute(
        "INSERT INTO queries (query_id, text, route) VALUES (%s, NULL, 'SENSITIVE')",
        (qid,),
    )
    row = conn.execute(
        "SELECT text, route FROM queries WHERE query_id = %s", (qid,)
    ).fetchone()
    assert row[0] is None
    assert row[1] == "SENSITIVE"
    conn.execute("DELETE FROM queries WHERE query_id = %s", (qid,))
    conn.commit()
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.db'`

- [ ] **Step 5: Write `src/db.py`**

```python
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from src.config import require_env

SCHEMA_PATH = Path("sql/schema.sql")

TABLES = [
    "eval_results", "eval_runs", "answers", "retrievals",
    "queries", "embeddings", "sections",
]


@contextmanager
def connect():
    """A connection to the database named by DATABASE_URL.

    Plain Postgres and pgvector only. Nothing here may depend on a Neon
    feature, so moving to self-hosted Postgres stays a configuration change.
    """
    conn = psycopg.connect(require_env("DATABASE_URL"), autocommit=False)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def create_schema(conn) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def drop_schema(conn) -> None:
    for table in TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    conn.commit()
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db.py -v -m integration`
Expected: 6 passed

Confirm the default run skips them rather than failing: `.venv/Scripts/python.exe -m pytest tests/test_db.py -v` should report 6 deselected, 0 errors.

- [ ] **Step 7: Commit**

```bash
git add src/db.py sql/schema.sql tests/test_db.py
git commit -m "feat: Postgres schema with pgvector, FTS over plain mapping column"
```

---

## Task 10: Embedding format and generation

**Files:**
- Create: `src/embed_format.py`, `src/embed.py`
- Test: `tests/test_embed_format.py`, `tests/test_embed.py`

**Interfaces:**
- Consumes: `Section` from Task 1, `require_env` from Task 1.
- Produces: `format_document(section) -> str`; `format_query(question) -> str`; `MODEL_NAME`, `DIMENSION` constants; `embed_texts(texts) -> list[list[float]]`; `embed_sections(sections) -> dict[str, list[float]]`; `embed_query(question) -> list[float]`.

**Context:** Read the "Decisions this plan makes" section on one-call-per-section. `embed_format.py` exists as its own module for exactly one reason: index-time and query-time formatting must not drift apart. Never inline either formatter anywhere else.

- [ ] **Step 1: Write the failing tests**

`tests/test_embed_format.py`:

```python
from src.embed_format import DIMENSION, MODEL_NAME, format_document, format_query
from src.models import Section


def _section(**kwargs):
    defaults = dict(
        id="bns-303", act="BNS", act_number="45 of 2023", status="enacted",
        as_of_date="2025-10-06", section_number="303", section_title="Theft",
        chapter_number="XVII", chapter_title="OF OFFENCES AGAINST PROPERTY",
        text="Whoever, intending to take dishonestly...", source_page=88, char_count=40,
    )
    defaults.update(kwargs)
    return Section(**defaults)


def test_document_format_is_exact():
    out = format_document(_section())
    assert out == "title: Theft | text: Whoever, intending to take dishonestly..."


def test_query_format_is_exact():
    assert format_query("what is theft") == "task: question answering | query: what is theft"


def test_model_constants():
    assert MODEL_NAME == "gemini-embedding-2"
    assert DIMENSION == 768


def test_document_format_does_not_include_section_numbers():
    """Section numbers embed to nothing useful. They are matched by BM25 via
    the FTS column instead."""
    assert "303" not in format_document(_section())
```

`tests/test_embed.py`:

```python
import pytest

import pytest

from src.embed import embed_texts
from src.embed_format import DIMENSION

# Every test here spends embedding quota against a live API key.
pytestmark = pytest.mark.integration


def test_embed_texts_returns_one_vector_per_input():
    vectors = embed_texts(["theft of movable property", "criminal intimidation"])
    assert len(vectors) == 2
    assert all(len(v) == DIMENSION for v in vectors)


def test_distinct_inputs_produce_distinct_vectors():
    """The tripwire for silent aggregation. If the model returned one
    averaged vector for a batch, these would be identical."""
    a, b = embed_texts(["theft of movable property", "abetment of suicide"])
    assert a != b


def test_vectors_are_normalised():
    """gemini-embedding-2 normalises truncated outputs, which is what makes
    cosine distance valid without renormalising."""
    (vector,) = embed_texts(["theft"])
    magnitude = sum(x * x for x in vector) ** 0.5
    assert abs(magnitude - 1.0) < 0.01


def test_cache_makes_a_repeat_call_free(tmp_path):
    from src import embed

    embed.CACHE_DIR = tmp_path
    first = embed_texts(["a test sentence about theft"])
    calls_before = embed.API_CALLS
    second = embed_texts(["a test sentence about theft"])
    assert first == second
    assert embed.API_CALLS == calls_before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embed_format.py tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.embed_format'`

- [ ] **Step 3: Write `src/embed_format.py`**

```python
from src.models import Section

MODEL_NAME = "gemini-embedding-2"

# 768, never the 3072 default: pgvector's HNSW index caps at 2000 dimensions.
# gemini-embedding-2 normalises truncated outputs automatically, so cosine
# distance is valid without renormalising.
DIMENSION = 768


def format_document(section: Section) -> str:
    """Index-time text for one section.

    The title is prepended because it is a strong semantic signal that the
    body text often does not contain in plain form — the word 'Theft' rarely
    appears in the section that defines theft.
    """
    return f"title: {section.section_title} | text: {section.text}"


def format_query(question: str) -> str:
    """Query-time text.

    This function and format_document live in the same module deliberately.
    If index-time and query-time formatting drift apart, retrieval degrades
    in a way that is invisible from the outside.
    """
    return f"task: question answering | query: {question}"
```

- [ ] **Step 4: Write `src/embed.py`**

```python
import hashlib
import json
import logging
import random
import time
from pathlib import Path

from google import genai
from google.genai import types

from src.config import require_env
from src.embed_format import DIMENSION, MODEL_NAME, format_document, format_query

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/embeddings")
MAX_RETRIES = 6

# Test-visible counter so the cache can be proven to avoid API calls.
API_CALLS = 0

_client = None


class EmbeddingCountMismatch(Exception):
    """Raised when the API returns a different number of vectors than sent."""


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    return _client


def _cache_path(text: str) -> Path:
    key = hashlib.sha256(f"{MODEL_NAME}:{DIMENSION}:{text}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _embed_one(text: str) -> list:
    """One text, one request.

    Deliberately not batched. The Gemini documentation states that
    gemini-embedding-2 "produces a single aggregated embedding" when multiple
    inputs are passed directly in `contents`, and that failure is silent — it
    yields one averaged vector where N were expected, with no error. At 358
    sections against a 1,000/day quota the round trips are affordable and the
    correctness guarantee is not negotiable.
    """
    global API_CALLS
    client = _get_client()
    for attempt in range(MAX_RETRIES):
        try:
            API_CALLS += 1
            result = client.models.embed_content(
                model=MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSION),
            )
            if len(result.embeddings) != 1:
                raise EmbeddingCountMismatch(
                    f"sent 1 input, received {len(result.embeddings)} embeddings"
                )
            values = list(result.embeddings[0].values)
            if len(values) != DIMENSION:
                raise EmbeddingCountMismatch(
                    f"expected {DIMENSION} dimensions, received {len(values)}"
                )
            return values
        except EmbeddingCountMismatch:
            raise
        except Exception as exc:  # rate limits and transient transport errors
            if attempt == MAX_RETRIES - 1:
                raise
            delay = (2 ** attempt) + random.random()
            log.warning("embed retry %d after %.1fs: %s", attempt + 1, delay, exc)
            time.sleep(delay)


def embed_texts(texts: list) -> list:
    """Embed a list of texts, one request each, cached by content hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors = []
    for text in texts:
        path = _cache_path(text)
        if path.exists():
            vectors.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        vector = _embed_one(text)
        path.write_text(json.dumps(vector), encoding="utf-8")
        vectors.append(vector)

    if len(vectors) != len(texts):
        raise EmbeddingCountMismatch(
            f"sent {len(texts)} texts, produced {len(vectors)} vectors"
        )
    return vectors


def embed_sections(sections: list) -> dict:
    """Embed every section. Returns {section_id: vector}."""
    texts = [format_document(s) for s in sections]
    vectors = embed_texts(texts)
    if len(vectors) != len(sections):
        raise EmbeddingCountMismatch(
            f"sent {len(sections)} sections, produced {len(vectors)} vectors"
        )
    return {section.id: vector for section, vector in zip(sections, vectors)}


def embed_query(question: str) -> list:
    (vector,) = embed_texts([format_query(question)])
    return vector
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_embed_format.py tests/test_embed.py -v -m ""`
Expected: 8 passed (4 unit from test_embed_format, 4 integration from test_embed)

Default run: `.venv/Scripts/python.exe -m pytest tests/test_embed_format.py tests/test_embed.py -v` reports 4 passed, 4 deselected.

- [ ] **Step 6: Commit**

```bash
git add src/embed_format.py src/embed.py tests/test_embed_format.py tests/test_embed.py
git commit -m "feat: Gemini embeddings with content-hash cache and aggregation tripwire"
```

---

## Task 11: Load sections and embeddings into Postgres

**Files:**
- Create: `src/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `connect`, `create_schema` from Task 9; `embed_sections` from Task 10; `mapping_text` from Task 8; `Section` from Task 1.
- Produces: `load_sections(conn, sections, mappings) -> int`; `load_embeddings(conn, vectors) -> int`; `verify_embedding_model(conn) -> None`; `main()`.

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
import json

import pytest

from src.db import connect, create_schema
from src.embed_format import DIMENSION, MODEL_NAME
from src.models import Section
from src.store import load_sections, verify_embedding_model, EmbeddingModelMismatch

# Every test here needs a live database and the processed data files.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    with connect() as connection:
        create_schema(connection)
        yield connection


@pytest.fixture(scope="module")
def loaded(conn):
    sections = [
        Section.from_dict(d)
        for d in json.load(open("data/processed/sections.json", encoding="utf-8"))
    ]
    mappings = json.load(open("data/processed/mappings.json", encoding="utf-8"))
    return load_sections(conn, sections, mappings)


def test_all_358_sections_load(conn, loaded):
    assert loaded == 358
    count = conn.execute("SELECT count(*) FROM sections WHERE act = 'BNS'").fetchone()[0]
    assert count == 358


def test_maps_to_text_is_populated_and_inspectable(conn, loaded):
    row = conn.execute(
        "SELECT maps_to_text FROM sections WHERE id = 'bns-303'"
    ).fetchone()
    assert row[0] == "IPC 378 IPC 379"


def test_maps_to_jsonb_matches_maps_to_text(conn, loaded):
    row = conn.execute(
        "SELECT maps_to, maps_to_text FROM sections WHERE id = 'bns-318'"
    ).fetchone()
    assert "420" in row[0]["sections"]
    assert "IPC 420" in row[1]


def test_reload_is_idempotent(conn, loaded):
    sections = [
        Section.from_dict(d)
        for d in json.load(open("data/processed/sections.json", encoding="utf-8"))
    ]
    mappings = json.load(open("data/processed/mappings.json", encoding="utf-8"))
    again = load_sections(conn, sections, mappings)
    assert again == 358
    count = conn.execute("SELECT count(*) FROM sections").fetchone()[0]
    assert count == 358


def test_verify_embedding_model_passes_on_matching_rows(conn):
    conn.execute("DELETE FROM embeddings")
    conn.execute(
        "INSERT INTO embeddings (section_id, vector, model_name, dimension)"
        " VALUES ('bns-303', %s, %s, %s)",
        ([0.0] * DIMENSION, MODEL_NAME, DIMENSION),
    )
    conn.commit()
    verify_embedding_model(conn)


def test_verify_embedding_model_fails_on_stale_rows(conn):
    conn.execute("DELETE FROM embeddings")
    conn.execute(
        "INSERT INTO embeddings (section_id, vector, model_name, dimension)"
        " VALUES ('bns-303', %s, %s, %s)",
        ([0.0] * DIMENSION, "gemini-embedding-001", DIMENSION),
    )
    conn.commit()
    with pytest.raises(EmbeddingModelMismatch, match="gemini-embedding-001"):
        verify_embedding_model(conn)
    conn.execute("DELETE FROM embeddings")
    conn.commit()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.store'`

- [ ] **Step 3: Write `src/store.py`**

```python
import json
import logging
from pathlib import Path

from src.config import load_act_config
from src.db import connect, create_schema
from src.embed import embed_sections
from src.embed_format import DIMENSION, MODEL_NAME
from src.mapping import mapping_text
from src.models import Section

log = logging.getLogger(__name__)

SECTIONS_PATH = Path("data/processed/sections.json")
MAPPINGS_PATH = Path("data/processed/mappings.json")

INSERT_SECTION = """
INSERT INTO sections (
    id, act, act_number, status, as_of_date, section_number, section_title,
    chapter_number, chapter_title, text, illustrations, illustrations_text,
    maps_to, maps_to_text, source_page, char_count
) VALUES (
    %(id)s, %(act)s, %(act_number)s, %(status)s, %(as_of_date)s,
    %(section_number)s, %(section_title)s, %(chapter_number)s,
    %(chapter_title)s, %(text)s, %(illustrations)s, %(illustrations_text)s,
    %(maps_to)s, %(maps_to_text)s, %(source_page)s, %(char_count)s
)
ON CONFLICT (id) DO UPDATE SET
    text = EXCLUDED.text,
    section_title = EXCLUDED.section_title,
    chapter_number = EXCLUDED.chapter_number,
    chapter_title = EXCLUDED.chapter_title,
    illustrations = EXCLUDED.illustrations,
    illustrations_text = EXCLUDED.illustrations_text,
    maps_to = EXCLUDED.maps_to,
    maps_to_text = EXCLUDED.maps_to_text,
    source_page = EXCLUDED.source_page,
    char_count = EXCLUDED.char_count
"""

INSERT_EMBEDDING = """
INSERT INTO embeddings (section_id, vector, model_name, dimension)
VALUES (%s, %s, %s, %s)
ON CONFLICT (section_id) DO UPDATE SET
    vector = EXCLUDED.vector,
    model_name = EXCLUDED.model_name,
    dimension = EXCLUDED.dimension,
    created_at = now()
"""


class EmbeddingModelMismatch(Exception):
    """Raised when stored vectors came from a different embedding model."""


def load_sections(conn, sections: list, mappings: dict, target_act: str = "IPC") -> int:
    """Insert or update every section, joining in its mapping."""
    for section in sections:
        ipc = mappings.get(section.section_number, [])
        params = section.to_dict()
        params["illustrations"] = json.dumps(section.illustrations, ensure_ascii=False)
        params["maps_to"] = json.dumps(
            {"act": target_act, "sections": ipc} if ipc else {}, ensure_ascii=False
        )
        params["maps_to_text"] = mapping_text(ipc, target_act)
        conn.execute(INSERT_SECTION, params)
    conn.commit()
    return len(sections)


def load_embeddings(conn, vectors: dict) -> int:
    for section_id, vector in vectors.items():
        conn.execute(INSERT_EMBEDDING, (section_id, vector, MODEL_NAME, DIMENSION))
    conn.commit()
    return len(vectors)


def verify_embedding_model(conn) -> None:
    """Fail loudly if stored vectors came from a different model.

    The embedding-001 and embedding-2 vector spaces are incompatible. Mixing
    them degrades retrieval invisibly, with no error and no obvious symptom,
    so this runs at startup rather than being left to chance.
    """
    rows = conn.execute(
        "SELECT DISTINCT model_name, dimension FROM embeddings"
    ).fetchall()
    for model_name, dimension in rows:
        if model_name != MODEL_NAME or dimension != DIMENSION:
            raise EmbeddingModelMismatch(
                f"stored embeddings are {model_name} at {dimension} dimensions, "
                f"but query time uses {MODEL_NAME} at {DIMENSION}. Re-run embedding."
            )


def main(act_key: str = "bns") -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_act_config(act_key)  # fail early if config is missing

    sections = [
        Section.from_dict(d)
        for d in json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    ]
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))

    with connect() as conn:
        create_schema(conn)
        count = load_sections(conn, sections, mappings)
        log.info("loaded %d sections", count)

        vectors = embed_sections(sections)
        if len(vectors) != len(sections):
            raise SystemExit(
                f"embedded {len(vectors)} vectors for {len(sections)} sections"
            )
        log.info("loaded %d embeddings", load_embeddings(conn, vectors))
        verify_embedding_model(conn)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_store.py -v -m integration`
Expected: 6 passed

- [ ] **Step 5: Run the real load**

Run: `.venv/Scripts/python.exe -m src.store`
Expected: `loaded 358 sections`, then `loaded 358 embeddings`. Takes roughly 4 minutes at 100 requests per minute on first run; near-instant afterwards from cache.

- [ ] **Step 6: Commit**

```bash
git add src/store.py tests/test_store.py
git commit -m "feat: load sections, mappings, and embeddings into Postgres"
```

---

## Task 12: Retrieval

**Files:**
- Create: `src/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `connect` from Task 9; `embed_query` from Task 10.
- Produces: `dense_search(conn, question, k) -> list[Retrieved]`; `sparse_search(conn, question, k) -> list[Retrieved]`; `Retrieved` dataclass with `section_id, section_number, section_title, text, score, rank`.

**Context:** Phase 1 is dense-only per SPEC.md §7, but `sparse_search` ships now because the design doc §5 fix — mapping tokens in the FTS document — needs its own direct test, with no LLM involved. RRF fusion is Phase 2.

- [ ] **Step 1: Write the failing test**

`tests/test_retrieve.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.retrieve'`

- [ ] **Step 3: Write `src/retrieve.py`**

```python
from dataclasses import dataclass

from src.embed import embed_query

DENSE_SQL = """
SELECT s.id, s.section_number, s.section_title, s.text,
       1 - (e.vector <=> %s::vector) AS score
FROM embeddings e
JOIN sections s ON s.id = e.section_id
ORDER BY e.vector <=> %s::vector
LIMIT %s
"""

SPARSE_SQL = """
SELECT s.id, s.section_number, s.section_title, s.text,
       ts_rank(s.fts, plainto_tsquery('english', %s)) AS score
FROM sections s
WHERE s.fts @@ plainto_tsquery('english', %s)
ORDER BY score DESC
LIMIT %s
"""


@dataclass
class Retrieved:
    section_id: str
    section_number: str
    section_title: str
    text: str
    score: float
    rank: int


def _rows_to_results(rows: list) -> list:
    return [
        Retrieved(
            section_id=row[0],
            section_number=row[1],
            section_title=row[2],
            text=row[3],
            score=float(row[4]),
            rank=i + 1,
        )
        for i, row in enumerate(rows)
    ]


def dense_search(conn, question: str, k: int = 20) -> list:
    """Cosine similarity over section embeddings."""
    vector = embed_query(question)
    rows = conn.execute(DENSE_SQL, (vector, vector, k)).fetchall()
    return _rows_to_results(rows)


def sparse_search(conn, question: str, k: int = 20) -> list:
    """Postgres full-text search over title, body, illustrations, and mappings.

    The mapping tokens are what make migration queries work. 'IPC 420' does
    not appear anywhere in the gazette, so without maps_to_text in the fts
    column this returns nothing for the single most common migration query.
    """
    rows = conn.execute(SPARSE_SQL, (question, question, k)).fetchall()
    return _rows_to_results(rows)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieve.py -v -m integration`
Expected: 6 passed

If `test_dense_search_cannot_find_ipc_420` fails because dense retrieval *did* find it, that is a genuinely interesting result, not a bug. Change the test to record what actually happened and note it — the design doc's reasoning about section numbers would need revisiting.

- [ ] **Step 5: Commit**

```bash
git add src/retrieve.py tests/test_retrieve.py
git commit -m "feat: dense and sparse retrieval, with direct mapping-lookup test"
```

---

## Task 13: Generation behind a provider interface

**Files:**
- Create: `src/providers/__init__.py`, `src/providers/base.py`, `src/providers/gemini.py`, `src/generate.py`, `prompts/grounded_v1.txt`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `Retrieved` from Task 12; `require_env` from Task 1.
- Produces: `GenerationProvider` protocol with `generate(prompt, max_tokens) -> str`; `GeminiProvider`; `get_provider(name) -> GenerationProvider`; `build_prompt(question, results, template) -> str`; `answer(conn, question) -> Answer` with fields `text, prompt_version, retrieved`.

**Context:** The provider boundary is the point. Adding Claude later must touch nothing outside `src/providers/`.

- [ ] **Step 1: Write `prompts/grounded_v1.txt`**

```
You are answering a question about Indian criminal law using only the statutory sections provided below.

RULES:
1. Answer using ONLY the sections provided. Do not use any other knowledge of Indian law.
2. Cite every substantive claim inline using the format [BNS <section number>], for example [BNS 303].
3. Only cite section numbers that appear in the PROVIDED SECTIONS below. Never cite a section that is not there.
4. If the provided sections do not contain the answer, say so plainly. Do not reason from general knowledge to fill the gap.
5. Be precise and dry. Do not speculate about the questioner's situation.
6. Do not present your answer as legal advice.

PROVIDED SECTIONS:
{sections}

QUESTION:
{question}

ANSWER:
```

- [ ] **Step 2: Write the failing test**

`tests/test_generate.py`:

```python
import pytest

from src.generate import PROMPT_VERSION, build_prompt, get_provider
from src.retrieve import Retrieved


def _result(number, title, text):
    return Retrieved(
        section_id=f"bns-{number}", section_number=number,
        section_title=title, text=text, score=0.9, rank=1,
    )


def test_prompt_includes_every_retrieved_section():
    results = [
        _result("303", "Theft", "Whoever intending to take dishonestly..."),
        _result("304", "Snatching", "Theft is snatching if..."),
    ]
    prompt = build_prompt("what is theft", results)
    assert "BNS 303" in prompt
    assert "BNS 304" in prompt
    assert "Whoever intending to take dishonestly" in prompt
    assert "what is theft" in prompt


def test_prompt_forbids_uncited_sections():
    prompt = build_prompt("q", [_result("303", "Theft", "body")])
    assert "Never cite a section that is not there" in prompt


def test_prompt_version_is_recorded():
    assert PROMPT_VERSION == "grounded_v1"


def test_gemini_provider_is_the_default():
    provider = get_provider()
    assert provider.name == "gemini-2.5-flash"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("not-a-real-provider")


def test_provider_conforms_to_the_interface():
    """The provider boundary is what makes Claude a drop-in later. Verifying
    the contract does not require a real API call, so this stays a unit test."""

    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.calls = []

        def generate(self, prompt: str, max_tokens: int) -> str:
            self.calls.append((prompt, max_tokens))
            return "Theft is [BNS 303]."

    provider = FakeProvider()
    out = provider.generate("a prompt", max_tokens=800)
    assert isinstance(out, str)
    assert provider.calls == [("a prompt", 800)]


@pytest.mark.integration
def test_gemini_provider_returns_text_live():
    provider = get_provider()
    out = provider.generate("Reply with exactly the word: acknowledged", max_tokens=20)
    assert isinstance(out, str)
    assert out.strip()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.generate'`

- [ ] **Step 4: Write `src/providers/base.py`**

```python
from typing import Protocol


class GenerationProvider(Protocol):
    """The only surface generation code may depend on.

    Swapping providers must not require re-embedding and must not touch
    anything outside this package.
    """

    name: str

    def generate(self, prompt: str, max_tokens: int) -> str:
        ...
```

- [ ] **Step 5: Write `src/providers/gemini.py`**

```python
import logging
import random
import time

from google import genai
from google.genai import types

from src.config import require_env

log = logging.getLogger(__name__)

MAX_RETRIES = 5


class GeminiProvider:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.name = model
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            self._client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
        return self._client

    def generate(self, prompt: str, max_tokens: int) -> str:
        client = self._client_or_create()
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = (2 ** attempt) + random.random()
                log.warning("generate retry %d after %.1fs: %s", attempt + 1, delay, exc)
                time.sleep(delay)
```

- [ ] **Step 6: Write `src/providers/__init__.py`**

```python
from src.providers.base import GenerationProvider
from src.providers.gemini import GeminiProvider

__all__ = ["GenerationProvider", "GeminiProvider"]
```

- [ ] **Step 7: Write `src/generate.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path

from src.providers import GeminiProvider, GenerationProvider
from src.retrieve import Retrieved, dense_search

PROMPT_VERSION = "grounded_v1"
PROMPT_PATH = Path(f"prompts/{PROMPT_VERSION}.txt")

MAX_GROUNDED_TOKENS = 800
TOP_K = 8

DISCLAIMER = (
    "This is general information about the text of Indian criminal statutes, "
    "not legal advice. Consult a qualified lawyer about your situation."
)


@dataclass
class Answer:
    text: str
    prompt_version: str
    retrieved: list = field(default_factory=list)


def get_provider(name: str = "gemini") -> GenerationProvider:
    if name == "gemini":
        return GeminiProvider("gemini-2.5-flash")
    raise ValueError(f"unknown provider: {name!r}")


def build_prompt(question: str, results: list) -> str:
    blocks = []
    for result in results:
        blocks.append(
            f"[BNS {result.section_number}] {result.section_title}\n{result.text}"
        )
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(sections="\n\n---\n\n".join(blocks), question=question)


def answer(conn, question: str, provider: GenerationProvider = None, k: int = TOP_K) -> Answer:
    provider = provider or get_provider()
    results = dense_search(conn, question, k=k)
    text = provider.generate(build_prompt(question, results), MAX_GROUNDED_TOKENS)
    return Answer(text=text, prompt_version=PROMPT_VERSION, retrieved=results)
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_generate.py -v`
Expected: 6 passed, 1 deselected

Then the live one: `.venv/Scripts/python.exe -m pytest tests/test_generate.py -v -m integration` -> 1 passed. That is the only Flash call in the whole Phase 1 suite.

- [ ] **Step 9: Commit**

```bash
git add src/providers/ src/generate.py prompts/ tests/test_generate.py
git commit -m "feat: grounded generation behind a swappable provider interface"
```

---

## Task 14: Citation verification

**Files:**
- Create: `src/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Consumes: `Retrieved` from Task 12; `Answer` from Task 13.
- Produces: `extract_citations(text) -> list[str]`; `verify_citations(answer_text, results) -> VerificationResult` with fields `cited, valid, fabricated, cleaned_text`.

**Context:** SPEC.md §11 — no answer may contain a section number that was not retrieved, enforced in code rather than by prompt. This catches the failure mode that matters most, and it is cheap.

- [ ] **Step 1: Write the failing test**

`tests/test_verify.py`:

```python
from src.retrieve import Retrieved
from src.verify import extract_citations, verify_citations


def _results(*numbers):
    return [
        Retrieved(
            section_id=f"bns-{n}", section_number=n, section_title="T",
            text="body", score=0.9, rank=i + 1,
        )
        for i, n in enumerate(numbers)
    ]


def test_extract_citations_finds_bracketed_sections():
    assert extract_citations("Theft is [BNS 303] and snatching is [BNS 304].") == ["303", "304"]


def test_extract_citations_deduplicates_preserving_order():
    assert extract_citations("[BNS 303] then [BNS 303] again and [BNS 101]") == ["303", "101"]


def test_extract_citations_ignores_bare_numbers():
    assert extract_citations("Section 303 says a thing about 420.") == []


def test_all_cited_sections_retrieved_is_valid():
    result = verify_citations("Theft is [BNS 303].", _results("303", "304"))
    assert result.valid == ["303"]
    assert result.fabricated == []


def test_fabricated_citation_is_detected():
    result = verify_citations("Cheating is [BNS 999].", _results("303"))
    assert result.fabricated == ["999"]
    assert result.valid == []


def test_fabricated_citation_is_stripped_from_the_text():
    result = verify_citations(
        "Theft is [BNS 303] and fraud is [BNS 999].", _results("303")
    )
    assert "[BNS 999]" not in result.cleaned_text
    assert "[BNS 303]" in result.cleaned_text


def test_answer_with_no_citations_reports_none():
    result = verify_citations("I cannot answer that.", _results("303"))
    assert result.cited == []
    assert result.fabricated == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.verify'`

- [ ] **Step 3: Write `src/verify.py`**

```python
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[BNS\s+(\d+[A-Z]?)\]")


@dataclass
class VerificationResult:
    cited: list = field(default_factory=list)
    valid: list = field(default_factory=list)
    fabricated: list = field(default_factory=list)
    cleaned_text: str = ""


def extract_citations(text: str) -> list:
    """Section numbers cited inline, in order of first appearance.

    Only bracketed citations count. A bare number in running text is not a
    citation and must not be treated as one.
    """
    seen = []
    for number in CITATION.findall(text):
        if number not in seen:
            seen.append(number)
    return seen


def verify_citations(answer_text: str, results: list) -> VerificationResult:
    """Confirm every cited section was actually retrieved.

    A citation to a section that was never retrieved is a fabrication. This
    is enforced here, in code, rather than trusted to the prompt, because
    prompts are advice and this is a guarantee.
    """
    retrieved = {r.section_number for r in results}
    cited = extract_citations(answer_text)
    valid = [n for n in cited if n in retrieved]
    fabricated = [n for n in cited if n not in retrieved]

    cleaned = answer_text
    for number in fabricated:
        log.warning("fabricated citation stripped: BNS %s not in retrieved set", number)
        cleaned = cleaned.replace(f"[BNS {number}]", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return VerificationResult(
        cited=cited, valid=valid, fabricated=fabricated, cleaned_text=cleaned
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_verify.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/verify.py tests/test_verify.py
git commit -m "feat: programmatic citation verification with fabrication stripping"
```

---

## Task 15: CLI and the Phase 1 acceptance script

**Files:**
- Create: `src/cli.py`, `scripts/acceptance.py`
- Test: `tests/test_cli.py`

`src/cli.py` is the deliverable SPEC.md §7 calls `query.py` — "taking a question, returning an answer with citations". Renamed only because it is the CLI entry point; the role is identical.

**Interfaces:**
- Consumes: everything.
- Produces: `python -m src.cli "<question>"`; `python -m scripts.acceptance`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
from src.cli import format_output
from src.generate import Answer
from src.retrieve import Retrieved
from src.verify import verify_citations


def _answer(text, numbers):
    results = [
        Retrieved(
            section_id=f"bns-{n}", section_number=n, section_title="Theft",
            text="body", score=0.9, rank=i + 1,
        )
        for i, n in enumerate(numbers)
    ]
    return Answer(text=text, prompt_version="grounded_v1", retrieved=results)


def test_output_includes_the_disclaimer():
    answer = _answer("Theft is [BNS 303].", ["303"])
    out = format_output(answer, verify_citations(answer.text, answer.retrieved))
    assert "not legal advice" in out.lower()


def test_output_lists_the_retrieved_sources():
    answer = _answer("Theft is [BNS 303].", ["303", "304"])
    out = format_output(answer, verify_citations(answer.text, answer.retrieved))
    assert "BNS 303" in out
    assert "BNS 304" in out


def test_output_flags_fabricated_citations():
    answer = _answer("Fraud is [BNS 999].", ["303"])
    out = format_output(answer, verify_citations(answer.text, answer.retrieved))
    assert "999" in out
    assert "not retrieved" in out.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cli'`

- [ ] **Step 3: Write `src/cli.py`**

`src/generate.py` needs no change. `answer()` deliberately returns the model's raw text, and `cli.py` applies `verify_citations` to it. Keeping verification outside generation means the unverified text stays available for logging — you cannot measure how often the model fabricates a citation if generation has already scrubbed the evidence.

```python
import argparse
import logging
import sys

from src.db import connect
from src.generate import DISCLAIMER, answer
from src.store import verify_embedding_model
from src.verify import verify_citations


def format_output(result, verification) -> str:
    lines = [verification.cleaned_text, ""]

    if verification.fabricated:
        lines.append(
            "WARNING: the model cited sections that were not retrieved and they "
            f"have been removed: {', '.join(verification.fabricated)}"
        )
        lines.append("")

    lines.append("Sources retrieved:")
    for item in result.retrieved:
        lines.append(
            f"  [BNS {item.section_number}] {item.section_title} "
            f"(score {item.score:.3f})"
        )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ask a question about the BNS.")
    parser.add_argument("question", help="your question, in plain language")
    parser.add_argument("-k", type=int, default=8, help="sections to retrieve")
    args = parser.parse_args()

    with connect() as conn:
        verify_embedding_model(conn)
        result = answer(conn, args.question, k=args.k)
        verification = verify_citations(result.text, result.retrieved)
        print(format_output(result, verification))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: Write `scripts/acceptance.py`**

The Phase 1 checklist from design doc §9, as runnable code.

```python
"""Phase 1 acceptance checks. Run after a full ingest and load.

Prints one line per criterion. Exits non-zero if any automated check fails.
The two by-hand criteria are printed as reminders, not asserted.
"""
import json
import sys

import pymupdf

from src.config import load_act_config
from src.db import connect
from src.manifest import verify_source
from src.parse_index import parse_index
from src.retrieve import dense_search, sparse_search
from src.store import verify_embedding_model

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((PASS if condition else FAIL, name, detail))


def main() -> int:
    cfg = load_act_config("bns")
    entry = verify_source("bns")
    sections = json.load(open("data/processed/sections.json", encoding="utf-8"))
    mappings = json.load(open("data/processed/mappings.json", encoding="utf-8"))

    check("source hash verified", entry["sha256"].startswith("ff92dcc7"))
    check("section count is exactly 358", len(sections) == 358, f"got {len(sections)}")

    numbers = [int(s["section_number"]) for s in sections]
    check("numbers contiguous 1-358", set(numbers) == set(range(1, 359)))
    check("numbers monotonic", numbers == sorted(numbers))

    shortest = min(s["char_count"] for s in sections)
    longest = max(s["char_count"] for s in sections)
    check("no chunk under 50 chars", shortest >= 50, f"min {shortest}")
    check(
        f"no chunk over {cfg['max_chunk_chars']} chars",
        longest <= cfg["max_chunk_chars"],
        f"max {longest}",
    )
    check(
        "no chunk risks embedding truncation",
        longest <= cfg["max_embed_chars"],
        f"max {longest}",
    )

    index = parse_index(pymupdf.open(entry["path"]), cfg)
    check("index oracle has 358 entries", len(index) == 358, f"got {len(index)}")

    check("mapping rows parsed", len(mappings) >= 330, f"got {len(mappings)}")
    check("BNS 303 maps to IPC 378, 379", mappings.get("303") == ["378", "379"])
    check("BNS 318 mapping includes IPC 420", "420" in mappings.get("318", []))

    with connect() as conn:
        verify_embedding_model(conn)
        loaded = conn.execute("SELECT count(*) FROM sections").fetchone()[0]
        embedded = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        check("358 sections in database", loaded == 358, f"got {loaded}")
        check("embedding count equals section count", embedded == loaded,
              f"{embedded} vs {loaded}")

        theft = dense_search(conn, "what is the punishment for theft", k=3)
        check("theft query returns BNS 303 in top 3",
              "bns-303" in [r.section_id for r in theft])

        # The design doc's headline retrieval fix, isolated from generation.
        migration = sparse_search(conn, "IPC 420", k=5)
        check("FTS finds BNS 318 for 'IPC 420', no LLM involved",
              "bns-318" in [r.section_id for r in migration])

    for status, name, detail in results:
        suffix = f"  ({detail})" if detail else ""
        print(f"{status}  {name}{suffix}")

    print("\nBy hand, not asserted here:")
    print("  [ ] 5 random sections verified word for word against the PDF")
    print("  [ ] 20 BNS-to-IPC mappings verified against NCRB pages 20-73")
    print("  [ ] de-hyphenation join log reviewed")
    print("  [ ] index/body title diffs reviewed, all cosmetic")

    failures = sum(1 for status, _, _ in results if status == FAIL)
    print(f"\n{len(results) - failures}/{len(results)} automated checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the full pipeline end to end**

```bash
.venv/Scripts/python.exe -m src.ingest
.venv/Scripts/python.exe -m src.mapping
.venv/Scripts/python.exe -m src.store
.venv/Scripts/python.exe -m scripts.acceptance
```

Expected: `15/15 automated checks passed`

- [ ] **Step 7: Run the acceptance queries through the CLI**

```bash
.venv/Scripts/python.exe -m src.cli "what is the punishment for theft"
.venv/Scripts/python.exe -m src.cli "what is IPC 420 now"
.venv/Scripts/python.exe -m src.cli "someone broke into a house at night and took jewellery, what applies"
```

Each must print an answer containing at least one `[BNS <n>]` citation, the retrieved source list, and the disclaimer. Note that the second query is expected to be weak in Phase 1 — retrieval is dense-only, and the mapping fix only reaches it through sparse search, which arrives in Phase 2's hybrid fusion. The direct FTS check in `acceptance.py` is what proves the underlying fix works.

- [ ] **Step 8: Run the whole test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all unit tests pass, integration tests deselected, zero API calls.

Then, once, with secrets present: `.venv/Scripts/python.exe -m pytest tests/ -v -m ""`
Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add src/cli.py scripts/acceptance.py tests/test_cli.py
git commit -m "feat: end-to-end CLI and Phase 1 acceptance script"
```

---

## Done criteria for Phase 1

All of these, with evidence:

- `python -m scripts.acceptance` reports 15/15
- `python -m pytest tests/` is green
- The four by-hand reviews are done and their findings recorded
- `python -m src.cli "what is the punishment for theft"` returns a cited answer

After this the honest claim is "I built a RAG pipeline." Nothing beyond that yet — measurement is Phase 2, and no README claim may outrun the acceptance criteria that have actually been verified.

## What Phase 1 deliberately does not include

Per SPEC.md §7 and the design doc, these are Phase 2 and must not be built early:

- The three-way router, the safety classifier, and the OUT_OF_SCOPE voice. SPEC.md §6 is explicit that the personality ships **after** the eval set exists, because the eval set is what catches misroutes.
- RRF fusion of dense and sparse. `sparse_search` exists in Phase 1 only so the mapping fix has a direct test.
- The golden dataset and both eval harnesses.
- CI.
