import argparse
import logging
import sys

from src.db import connect
from src.generate import DISCLAIMER, answer
from src.store import verify_embedding_model
from src.verify import verify_citations


def format_output(result, verification) -> str:
    lines = [verification.cleaned_text, ""]

    if verification.fabricated:
        lines.append(
            "WARNING: the model cited sections that were not retrieved and they "
            f"have been removed: {', '.join(verification.fabricated)}"
        )
        lines.append("")

    lines.append("Sources retrieved:")
    for item in result.retrieved:
        lines.append(
            f"  [BNS {item.section_number}] {item.section_title} "
            f"(score {item.score:.3f})"
        )
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ask a question about the BNS.")
    parser.add_argument("question", help="your question, in plain language")
    parser.add_argument("-k", type=int, default=8, help="sections to retrieve")
    args = parser.parse_args()

    with connect() as conn:
        verify_embedding_model(conn)
        result = answer(conn, args.question, k=args.k)
        verification = verify_citations(result.text, result.retrieved)
        print(format_output(result, verification))

    return 0


if __name__ == "__main__":
    sys.exit(main())
