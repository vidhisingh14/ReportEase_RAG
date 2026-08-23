"""Phase 1 acceptance checks. Run after a full ingest and load.

Prints one line per criterion. Exits non-zero if any automated check fails.
The two by-hand criteria are printed as reminders, not asserted.
"""
import json
import sys

import pymupdf

from src.config import load_act_config
from src.db import connect
from src.manifest import sha256_file, verify_source
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

    # Recompute the hash from the PDF on disk and compare to the manifest's
    # recorded value, rather than re-reading the manifest's own string back
    # at itself (which cannot fail regardless of what the file contains).
    check("source hash verified", sha256_file(entry["path"]) == entry["sha256"])
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

        distinct = conn.execute("SELECT count(DISTINCT vector) FROM embeddings").fetchone()[0]
        check("embeddings are distinct, not one aggregated vector", distinct == embedded,
              f"{distinct} distinct of {embedded}")

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
