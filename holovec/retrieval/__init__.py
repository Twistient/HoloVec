"""Lightweight retrieval primitives (Codebook, ItemStore).

These provide passive storage and retrieval over vector codebooks without
introducing heavy memory systems. Advanced learned/dynamic memories live in
the HoloMem package.
"""

from .codebook import Codebook
from .itemstore import ItemStore
from .assocstore import AssocStore

__all__ = [
    "Codebook",
    "ItemStore",
    "AssocStore",
]
