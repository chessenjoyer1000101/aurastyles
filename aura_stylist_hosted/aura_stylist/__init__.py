"""Aura Stylera AI module: wardrobe understanding and outfit recommendation."""
from .analysis import Gap, StyleSuggestion, discover_style, find_gaps
from .feedback import PreferenceModel
from .knowledge import CATEGORIES, build_item
from .recommender import Recommender, explain
from .schema import (ColorInfo, Context, Occasion, Outfit, Reason, Recommendation,
                     Slot, StyleProfile, WardrobeItem)
from .stylist import AuraStylist
from .vision import ClipBackend, ItemClassifier

__all__ = ["AuraStylist", "ItemClassifier", "ClipBackend", "Recommender", "PreferenceModel",
           "StyleProfile", "Context", "Occasion", "Reason", "Slot", "WardrobeItem",
           "ColorInfo", "Outfit", "Recommendation", "Gap", "StyleSuggestion",
           "build_item", "find_gaps", "discover_style", "explain", "CATEGORIES"]
