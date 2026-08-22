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
