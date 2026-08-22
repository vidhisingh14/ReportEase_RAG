from src.config import load_act_config


def test_bns_config_has_expected_counts():
    cfg = load_act_config("bns")
    assert cfg["expected_section_count"] == 358
    assert cfg["expected_chapter_count"] == 20
    assert cfg["index_pages"] == [3, 15]
    assert cfg["heading_min_size"] == 10.0
