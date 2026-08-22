import re

# A hyphen with letters on both sides and no newline between them. Because
# the pattern contains no newline, it can only match within a single line,
# which is exactly what makes it evidence of a genuine hyphenate.
MIDLINE_HYPHEN = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")

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
