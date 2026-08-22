from src.dehyphen import build_keep_list, dehyphenate


def test_keep_list_harvests_midline_hyphenates():
    text = "The offence of house-breaking is defined.\nSee currency-notes above."
    keep = build_keep_list(text)
    assert "house-breaking" in keep
    assert "currency-notes" in keep


def test_keep_list_ignores_linebreak_hyphens():
    """A hyphen at a line break is evidence of splitting, not of a real
    hyphenate, so it must not seed the keep-list."""
    text = "counter-\nfeit coin"
    assert build_keep_list(text) == set()


def test_split_word_is_joined():
    text = "counter-\nfeit coin"
    out, joins = dehyphenate(text, build_keep_list(text))
    assert "counterfeit coin" in out
    assert joins == [("counter- feit", "counterfeit")]


def test_genuine_hyphenate_split_across_lines_is_preserved():
    """Section 179 renders 'bank-notes' as 'bank-\\nnotes'. Because
    'bank-notes' appears hyphenated mid-line elsewhere in the Act, the
    hyphen must be restored rather than removed."""
    text = "Possession of bank-notes here.\nUsing forged bank-\nnotes elsewhere."
    out, joins = dehyphenate(text, build_keep_list(text))
    assert "bank-notes elsewhere" in " ".join(out.split())
    assert ("bank-notes", "bank-notes") in joins


def test_every_join_is_logged():
    text = "counter-\nfeit and inter-\nnational"
    _, joins = dehyphenate(text, build_keep_list(text))
    assert len(joins) == 2


def test_real_corpus_join_count_is_small_and_reviewable():
    import pymupdf
    from src.config import load_act_config
    from src.pdf_text import joined_body

    doc = pymupdf.open("data/raw/a202345.pdf")
    cfg = load_act_config("bns")
    body, _, _ = joined_body(doc, cfg)
    _, joins = dehyphenate(body, build_keep_list(body))
    assert len(joins) < 20, f"unexpected join volume: {joins}"
