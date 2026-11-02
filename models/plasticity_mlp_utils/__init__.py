"""
Dendritic geometry helpers for `plasticity_mlp`.

Public API:
    DendriteGeometry - per-postsynaptic-neuron 1-D dendritic lines for one
                       Linear layer: random synapse positions + barriers warped
                       into a coordinate `u`, so the spillover kernel between two
                       synapses of the same neuron is a plain exponential
                       exp(-|du|/sigma). Buffers are fixed (rolled once, re-rolled
                       only on neuron death). geometry='none' is never built here.
"""

from .dendrite_geometry import DendriteGeometry

__all__ = ["DendriteGeometry"]
