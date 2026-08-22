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
# Searched with MULTILINE against the whole (possibly multi-line) cell,
# not just anchored to the first character: some cells open with a plain
# offence label ('Cheating by personation\n319(1)\n319(2)') and only carry
# the section number on a later line. Matching cell-start only leaves
# `current` stuck on the previous section and cross-contaminates its list.
SECTION_DECLARATION = re.compile(r"^\s*(\d+)\s*\.", re.MULTILINE)
# A left cell that continues one: '318 (4)'
SECTION_CONTINUATION = re.compile(r"^\s*(\d+)\s*\(", re.MULTILINE)
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
        # The default "lines" strategy requires a complete ruling on both
        # sides of a row to recognize it. Rows that straddle a page break
        # (no ruling above the first row of a continued table) are silently
        # dropped rather than misparsed -- this is exactly where BNS 318's
        # continuation row to IPC 420 lives. "lines_strict" recovers them.
        tables = doc[pno].find_tables(strategy="lines_strict")
        if not tables.tables:
            continue
        for row in tables.tables[0].extract():
            if len(row) < 2:
                continue
            left = (row[0] or "").strip()
            right = (row[1] or "").strip()

            declaration = SECTION_DECLARATION.search(left)
            if declaration:
                current = declaration.group(1)
                mappings.setdefault(current, [])
            else:
                continuation = SECTION_CONTINUATION.search(left)
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
