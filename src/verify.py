import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# The model cites subsections naturally: [BNS 303(2)], [BNS 303(2) Proviso].
# Only the base section number is verifiable, since the retrieved set is keyed
# by section. Anything up to the closing bracket is tolerated and ignored.
#
# Match the whole bracketed citation; parse its contents separately. Case
# insensitive, because a lowercase [bns 999] would otherwise bypass
# verification entirely.
CITATION = re.compile(r"\[BNS\s+([^\]]*)\]", re.IGNORECASE)

# Subsection and clause markers are not section numbers. They must be removed
# before extracting numbers, or [BNS 303(2)] would yield a phantom citation to
# section 2.
_SUBSECTION = re.compile(r"\([^)]*\)")
_SECTION_NUMBER = re.compile(r"\b(\d+[A-Z]?)\b")


def _numbers_in(inner: str) -> list:
    """Every section number cited inside one bracket.

    A bracket may hold more than one. The prompt asks for one per bracket, but
    a model that writes "[BNS 303 and 304]" must not have the second number
    silently escape verification.
    """
    return _SECTION_NUMBER.findall(_SUBSECTION.sub(" ", inner))


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
    for inner in CITATION.findall(text):
        for number in _numbers_in(inner):
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
        # A bracket mixing a real and a fabricated number cannot be left
        # standing, since it would still assert the fabricated one: drop the
        # whole bracket if any number inside it is fabricated.
        numbers = _numbers_in(match.group(1))
        if any(n in fabricated_set for n in numbers):
            return ""
        return match.group(0)

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
