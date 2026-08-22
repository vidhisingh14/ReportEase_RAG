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
