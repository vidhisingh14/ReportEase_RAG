import hashlib
import json
from pathlib import Path

DEFAULT_MANIFEST = "data/raw/manifest.json"


class ManifestMismatch(Exception):
    """Raised when a source PDF does not match its recorded hash or status."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(key: str, manifest_path: str = DEFAULT_MANIFEST) -> dict:
    """Verify a source PDF before any ingest reads it.

    Guards two distinct failure modes: a swapped or corrupted file, and a
    withdrawn bill masquerading as the enacted act. Bills read almost
    identically to the acts they became but carry different section numbers,
    so indexing one would produce confident citations to sections that do not
    legally exist.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if key not in manifest:
        raise ManifestMismatch(f"no manifest entry for {key!r}")
    entry = manifest[key]

    if entry["status"] != "enacted":
        raise ManifestMismatch(
            f"{key}: source status is {entry['status']!r}, not enacted. Refusing to ingest."
        )

    actual = sha256_file(entry["path"])
    if actual != entry["sha256"]:
        raise ManifestMismatch(
            f"{key}: sha256 mismatch. expected {entry['sha256']}, got {actual}"
        )
    return entry
