import json
import pytest
from src.manifest import sha256_file, verify_source, ManifestMismatch


def test_sha256_matches_recorded_hash():
    entry = verify_source("bns")
    assert entry["act_number"] == "45 of 2023"
    assert entry["pages"] == 112


def test_tampered_hash_raises(tmp_path):
    manifest = json.loads(open("data/raw/manifest.json").read())
    manifest["bns"]["sha256"] = "0" * 64
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatch, match="sha256 mismatch"):
        verify_source("bns", manifest_path=str(p))


def test_non_enacted_status_refused(tmp_path):
    manifest = json.loads(open("data/raw/manifest.json").read())
    manifest["bns"]["status"] = "bill"
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatch, match="not enacted"):
        verify_source("bns", manifest_path=str(p))


def test_sha256_of_known_file_is_stable():
    assert sha256_file("data/raw/a202345.pdf").startswith("ff92dcc7")
