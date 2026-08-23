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
# verification entirely. The separator between "BNS" and its content is
# tolerant of whitespace, a hyphen, or nothing at all ([BNS 420], [BNS-420],
# [BNS420]), since a model that varies the separator must not thereby ship
# an unverified citation. The lookahead after BNS keeps this from matching
# an unrelated word like "BNSA" that merely starts with the same letters.
CITATION = re.compile(r"\[\s*BNS(?=[\s\-\d])[\s-]*([^\]]*)\]", re.IGNORECASE)

# Subsection and clause markers are not section numbers. They must be removed
# before extracting numbers, or [BNS 303(2)] would yield a phantom citation to
# section 2. Case insensitive so a lowercase suffix like "304a" still parses
# as a section number instead of silently yielding nothing.
_SUBSECTION = re.compile(r"\([^)]*\)")
_SECTION_NUMBER = re.compile(r"\b(\d+[A-Z]?)\b", re.IGNORECASE)


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
    unparseable: list = field(default_factory=list)
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

    # A bracket that matches [BNS ...] but yields zero section numbers is not
    # presumed innocent: an unparseable citation must fail closed, the same
    # as a fabricated one, rather than being passed through verbatim.
    unparseable = [
        inner.strip() for inner in CITATION.findall(answer_text) if not _numbers_in(inner)
    ]
    for inner in unparseable:
        log.warning("unparseable BNS bracket stripped: %r", inner)

    def _drop_if_fabricated_or_unparseable(match: "re.Match") -> str:
        # A bracket mixing a real and a fabricated number cannot be left
        # standing, since it would still assert the fabricated one: drop the
        # whole bracket if any number inside it is fabricated. A bracket with
        # no parseable number at all is dropped outright.
        numbers = _numbers_in(match.group(1))
        if not numbers or any(n in fabricated_set for n in numbers):
            return ""
        return match.group(0)

    # Strip by regex, not literal replace: a fabricated citation can carry a
    # subsection or proviso, e.g. [BNS 999(1)], which a literal
    # f"[BNS {number}]" match would never find, leaving it in cleaned_text
    # even though it was reported as fabricated.
    cleaned = CITATION.sub(_drop_if_fabricated_or_unparseable, answer_text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Stripping a citation that ended a sentence can leave a dangling space
    # before terminal punctuation, e.g. "fraud is  ." -> "fraud is ."
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)

    return VerificationResult(
        cited=cited,
        valid=valid,
        fabricated=fabricated,
        unparseable=unparseable,
        cleaned_text=cleaned,
    )
