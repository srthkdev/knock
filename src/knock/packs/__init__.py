"""Scenario packs. Importing a pack registers its components, tools and tasks."""
from ..core import PACKS
from .bottling import BottlingPack
from .tabletop import TabletopPack

PACKS.register()(TabletopPack)
PACKS.register()(BottlingPack)

__all__ = ["BottlingPack", "TabletopPack"]
