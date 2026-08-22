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
