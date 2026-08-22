import logging
import re

DASHES = "–—"  # en dash, em dash
NUM_START = re.compile(r"^\s*\d+\.")
HEADING = re.compile(r"^(\d+)\.\s*[%s]?\s*(.*)$" % DASHES)
HAS_LETTER = re.compile(r"[A-Za-z]")

log = logging.getLogger(__name__)


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
    """All section headings as (section_number, title, printed_page).

    `merge_runs` only joins wrapped titles within a single page's runs; it
    is called fresh per page, so a title wrapping across a page break cannot
    be reassembled. That failure mode drops a section silently: the
    continuation run has no leading 'N.' and fails HEADING, so it is simply
    skipped, while the preceding page's partial heading already matched and
    looks like a complete (if truncated) title. Rather than let that pass
    unnoticed, the first merged run of every page is checked against
    HEADING; a page-initial run that fails to match is flagged as a
    candidate cross-page wrap, and any such run anywhere in the document
    raises instead of silently dropping a section.

    A page-initial bold run with no letters at all (e.g. a single stray
    decorative glyph from a symbol font, which this PDF renders bold at
    heading size on a number of pages) is not a candidate title continuation
    — a wrapped title is always readable text — so it is excluded from the
    check rather than raising a false alarm.
    """
    headings = []
    unparsed_first_runs = []
    for pno in range(body_start, doc.page_count):
        merged = merge_runs(bold_runs(doc, pno, cfg))
        for i, run in enumerate(merged):
            match = HEADING.match(run)
            if not match:
                if i == 0 and HAS_LETTER.search(run):
                    unparsed_first_runs.append((pno + 1, run))
                    log.warning(
                        "page %d: bold run at page start did not parse as "
                        "a heading (possible cross-page title wrap): %r",
                        pno + 1,
                        run,
                    )
                continue
            title = match.group(2).strip().rstrip(DASHES).strip().rstrip(".")
            headings.append(
                (int(match.group(1)), " ".join(title.split()), pno + 1)
            )

    if unparsed_first_runs:
        pages = [p for p, _ in unparsed_first_runs]
        raise ValueError(
            f"{len(unparsed_first_runs)} bold run(s) at page starts did not "
            f"parse as headings (pages {pages}). These are probably section "
            "titles wrapping across a page break, which this parser does "
            "not join. See the parser contract in "
            "docs/superpowers/specs/2026-08-21-nyaya-design.md section 4.1."
        )

    return headings
