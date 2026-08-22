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


def test_multi_letter_suffix_is_captured(mappings):
    """BNS 70 (gang rape) cites IPC 376DA and 376DB. The original regex
    allowed at most one uppercase letter after the digits, so a two-letter
    suffix silently dropped these -- with 376D still present in the list,
    the section looked complete rather than obviously broken."""
    assert "376DA" in mappings["70"]
    assert "376DB" in mappings["70"]


def test_hyphen_roman_suffix_is_captured(mappings):
    """BNS 196 cites IPC 153AA alongside 153A. Same silent-drop shape as
    376DA/376DB: the truncated suffix (153A) was already present, so the
    missing citation (153AA) did not show up as an empty or missing entry."""
    assert "153AA" in mappings["196"]


def test_subsection_reference_collapses_without_leaking_the_parenthesis(mappings):
    """BNS 65 cites IPC 376(3) -- a subsection reference with no trailing
    dot. It must collapse to its parent section number '376' (a user
    searches by section, not subsection), and the bare parenthesised
    marker must never itself become a captured 'number'."""
    assert "376" in mappings["65"]
    flat = [n for ipc in mappings.values() for n in ipc]
    for n in flat:
        assert not n.startswith("(")
        assert n[0].isdigit()


def test_mapping_text_is_a_flat_searchable_string():
    assert mapping_text(["378", "379"], "IPC") == "IPC 378 IPC 379"


def test_mapping_text_is_empty_for_new_sections():
    assert mapping_text([], "IPC") == ""
