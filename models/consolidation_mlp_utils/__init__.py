"""
Sparse hippocampal consolidation on a standard MLP (vector reduction of the
ViT `consolidation_utils`). Kept separate from `consolidation_utils` so the
MLP path pulls in none of the token / head / chunking machinery.

Public API:
    SparseLayerBank - one sparse-code hippocampal bank for a Linear layer:
                      (fixed-random | learned) projection -> k-WTA / L1
                      sparsify -> EMA "superposed" bank -> cosine match.
"""

from .sparse_bank import SparseLayerBank
from .task_router import TaskRouter

__all__ = ["SparseLayerBank", "TaskRouter"]
