from dataclasses import dataclass, field, asdict


@dataclass
class Section:
    """One chunk. One chunk is always exactly one section — never split."""

    id: str
    act: str
    act_number: str
    status: str
    as_of_date: str
    section_number: str
    section_title: str
    chapter_number: str
    chapter_title: str
    text: str
    illustrations: list = field(default_factory=list)
    maps_to: dict = field(default_factory=dict)
    maps_to_text: str = ""
    illustrations_text: str = ""
    source_page: int = 0
    char_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Section":
        return Section(**d)
