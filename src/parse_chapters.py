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
