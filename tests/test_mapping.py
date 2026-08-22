import pytest

from src.mapping import mapping_text, parse_mappings


@pytest.fixture(scope="module")
def mappings():
    return parse_mappings("bns")


def test_covers_most_sections(mappings):
    assert len(mappings) >= 330


def test_theft_maps_to_ipc_378_and_379(mappings):
    """SPEC.md section 4 uses exactly this as its worked example."""
    assert mappings["303"] == ["378", "379"]


def test_cheating_includes_ipc_420(mappings):
    """IPC 420 lives in a continuation row whose left cell reads '318 (4)',
    with no section number of its own. A parser that requires a leading
    'NNN.' drops it, and 'what is IPC 420 now' is an acceptance criterion."""
    assert "420" in mappings["318"]


def test_ipc_numbers_are_strings_without_duplicates(mappings):
    for bns, ipc in mappings.items():
        assert all(isinstance(n, str) for n in ipc)
        assert len(ipc) == len(set(ipc))


def test_no_placeholder_text_leaks_into_numbers(mappings):
    flat = [n for ipc in mappings.values() for n in ipc]
    assert "New" not in flat
    assert "Deleted" not in flat


def test_mapping_text_is_a_flat_searchable_string():
    assert mapping_text(["378", "379"], "IPC") == "IPC 378 IPC 379"


def test_mapping_text_is_empty_for_new_sections():
    assert mapping_text([], "IPC") == ""
