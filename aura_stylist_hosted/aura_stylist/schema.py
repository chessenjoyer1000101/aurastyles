"""Data types shared by every part of the AI module."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class Slot(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    ONE_PIECE = "one_piece"
    OUTER = "outer"
    SHOES = "shoes"
    BAG = "bag"
    ACCESSORY = "accessory"


class Occasion(str, Enum):
    OFFICE = "office"
    CASUAL = "casual"
    DINNER = "dinner"
    FORMAL = "formal"      # weddings and formal occasions
    TRAVEL = "travel"
    SPECIAL = "special"


class Reason(str, Enum):
    """Why a user rejected a recommendation (US-013)."""
    COLORS = "colors"
    TOO_FORMAL = "too_formal"
    TOO_CASUAL = "too_casual"
    TOO_WARM = "too_warm"
    TOO_COLD = "too_cold"
    DISLIKE_ITEM = "dislike_item"
    NOT_MY_STYLE = "not_my_style"
    OTHER = "other"


@dataclass
class ColorInfo:
    name: str
    rgb: tuple
    share: float = 1.0


@dataclass
class WardrobeItem:
    id: str
    category: str                       # key of knowledge.CATEGORIES, or "unknown"
    slot: Optional[Slot]
    colors: list = field(default_factory=list)   # list[ColorInfo], dominant first
    pattern: str = "solid"
    formality: float = 0.5              # 0 = loungewear, 1 = black tie
    warmth: float = 0.5                 # 0 = hot weather only, 1 = deep winter
    coverage: float = 0.5               # 0 = minimal, 1 = full length / long sleeve
    styles: list = field(default_factory=list)
    open_shoe: bool = False
    confidence: float = 1.0             # classifier confidence in `category`
    needs_review: bool = False          # True => ask the user to confirm (US-003)
    alternatives: list = field(default_factory=list)  # [(category, prob), ...]
    image_path: Optional[str] = None
    wear_count: int = 0

    @property
    def main_color(self) -> Optional[ColorInfo]:
        return self.colors[0] if self.colors else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slot"] = self.slot.value if self.slot else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WardrobeItem":
        d = dict(d)
        d["slot"] = Slot(d["slot"]) if d.get("slot") else None
        d["colors"] = [ColorInfo(c["name"], tuple(c["rgb"]), c.get("share", 1.0))
                       for c in d.get("colors", [])]
        d["alternatives"] = [tuple(a) for a in d.get("alternatives", [])]
        return cls(**d)


@dataclass
class StyleProfile:
    """What the user told us (US-002). Every field is optional."""
    preferred_styles: list = field(default_factory=list)
    liked_colors: list = field(default_factory=list)
    avoided_colors: list = field(default_factory=list)
    occasions: list = field(default_factory=list)       # list[Occasion] the user dresses for
    min_coverage: Optional[float] = None                # modesty preference, 0..1
    always_include: list = field(default_factory=list)  # categories worn with every outfit
    language: str = "en"                                # "en" or "ar"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["occasions"] = [Occasion(o).value for o in self.occasions]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StyleProfile":
        d = dict(d)
        d["occasions"] = [Occasion(o) for o in d.get("occasions", [])]
        return cls(**d)


@dataclass
class Context:
    """What the outfit is for. Weather fields stay None unless a real source supplied them."""
    occasion: Occasion = Occasion.CASUAL
    temperature_c: Optional[float] = None
    condition: Optional[str] = None      # "clear", "rain", "snow", ...
    dress_code: Optional[str] = None     # "casual", "smart casual", "business", "formal"


@dataclass
class Outfit:
    items: list
    score: float = 0.0
    features: dict = field(default_factory=dict)
    explanation: str = ""

    @property
    def key(self) -> str:
        return "+".join(sorted(i.id for i in self.items))

    def item_ids(self) -> list:
        return [i.id for i in self.items]


@dataclass
class Recommendation:
    outfits: list
    notes: list = field(default_factory=list)   # limitations the user should be told about
