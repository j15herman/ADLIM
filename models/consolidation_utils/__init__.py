"""
Hippocampal-indexed consolidation module for attention layers.

Public API:
    CovarianceKeyEncoder      - input-side key encoder (covariance)
    SlicedWassersteinEncoder  - output-side target encoder (SW signature)
    HippocampalBank           - J-slot bank with cosine retrieval and EMA update
    ConsolidationModule       - per-layer orchestrator
"""

from .encoders import CovarianceKeyEncoder, SlicedWassersteinEncoder
from .bank import HippocampalBank
from .module import ConsolidationModule
from .batched import StackedConsolidation

__all__ = [
    "CovarianceKeyEncoder",
    "SlicedWassersteinEncoder",
    "HippocampalBank",
    "ConsolidationModule",
    "StackedConsolidation",
]
