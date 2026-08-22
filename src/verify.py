import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# The model cites subsections naturally: [BNS 303(2)], [BNS 303(2) Proviso].
# Only the base section number is verifiable, since the retrieved set is keyed
# by section. Anything up to the closing bracket is tolerated and ignored.
CITATION = re.compile(r"\[BNS\s+(\d+[A-Z]?)[^\]]*\]")


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

    fabricated_set = set(fabricated)
    for number in fabricated:
        log.warning("fabricated citation stripped: BNS %s not in retrieved set", number)

    def _drop_if_fabricated(match: "re.Match") -> str:
        return "" if match.group(1) in fabricated_set else match.group(0)

    # Strip by regex, not literal replace: a fabricated citation can carry a
    # subsection or proviso, e.g. [BNS 999(1)], which a literal
    # f"[BNS {number}]" match would never find, leaving it in cleaned_text
    # even though it was reported as fabricated.
    cleaned = CITATION.sub(_drop_if_fabricated, answer_text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Stripping a citation that ended a sentence can leave a dangling space
    # before terminal punctuation, e.g. "fraud is  ." -> "fraud is ."
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)

    return VerificationResult(
        cited=cited, valid=valid, fabricated=fabricated, cleaned_text=cleaned
    )
