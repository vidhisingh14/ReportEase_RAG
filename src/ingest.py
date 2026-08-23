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
    joined, raw_offsets, _ = joined_body(doc, cfg)

    keep_list = build_keep_list(joined)

    # joined_body's offsets are char positions in the PRE-dehyphenation text.
    # Dehyphenating the whole string in one pass (as opposed to per page)
    # would leave those offsets stale by however many characters each join
    # removed upstream of a given page, which is usually harmless but can
    # shift a heading's true start behind the page-floor search bound used
    # in locate_headings. Dehyphenating page by page and re-deriving the
    # offsets the same way joined_body does keeps offsets and text in sync.
    # (Verified: none of this document's line-wrap hyphens straddle a page
    # boundary, so splitting by page loses no genuine joins.)
    bounds = [offset for offset, _ in raw_offsets] + [len(joined)]
    joined_parts = []
    offsets = []
    pos = 0
    joins = []
    for i, (_, page_index) in enumerate(raw_offsets):
        page_text_slice = joined[bounds[i] : bounds[i + 1]]
        if i < len(raw_offsets) - 1:
            page_text_slice = page_text_slice[:-1]  # drop the joining "\n"
        new_text, page_joins = dehyphenate(page_text_slice, keep_list)
        joins.extend(page_joins)
        offsets.append((pos, page_index))
        joined_parts.append(new_text)
        pos += len(new_text) + 1
    joined = "\n".join(joined_parts)

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

    expected_chapters = cfg.get("expected_chapter_count")
    if expected_chapters is not None:
        chapter_numbers = {s.chapter_number for s in sections}
        if len(chapter_numbers) != expected_chapters:
            errors.append(
                f"expected {expected_chapters} chapters, got {len(chapter_numbers)}"
            )

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
