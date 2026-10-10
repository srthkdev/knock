"""Scenario packs. Importing a pack registers its components, tools and tasks."""
from ..core import PACKS
from .tabletop import TabletopPack

PACKS.register()(TabletopPack)

__all__ = ["TabletopPack"]
