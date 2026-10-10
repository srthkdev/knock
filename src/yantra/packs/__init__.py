"""Scenario packs. Importing a pack registers its components, tools and tasks."""
from ..core import PACKS
from .bottling import BottlingPack
from .factory import FactoryPack
from .tabletop import TabletopPack

PACKS.register()(TabletopPack)
PACKS.register()(BottlingPack)
PACKS.register()(FactoryPack)

__all__ = ["BottlingPack", "FactoryPack", "TabletopPack"]
