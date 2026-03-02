"""
damage/neuron_death.py — FFN neuron death with functional clustering.

Kills neurons by zeroing their corresponding rows/columns in the FFN gate,
up, and down projections. Death is applied within functional clusters
(neurons grouped by activation correlation), not randomly.

Spreading dynamics: death originates in seed layers (entorhinal analogue,
layers 0–5) and propagates along functional connectivity at a rate set by
NeuronDeathConfig.spreading_rate per epoch.

NOTE — GQA: When attention heads are ablated, remember that each KV head
maps to get_active_arch().gqa_ratio Q heads. Always use this multiplier,
never a hardcoded value — the ratio differs by model (3B: 8, 7B: 7, 32B: 5).

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from typing import Dict, List

from disease_state import DiseaseState, NeuronDeathConfig, WeightSnapshot


def kill_neurons(model: object, state: DiseaseState) -> WeightSnapshot:
    """Zero out rows/columns corresponding to dead neurons in FFN projections.

    Reads dead_neuron_indices from state.layer_states[i] — never recomputes.
    Zeros the corresponding rows in gate_proj/up_proj and columns in down_proj.

    Efficiency requirements:
    - Index into weight rows/cols directly (param.data[indices] = 0) rather
      than constructing a boolean mask over the full tensor.
    - Store only (flat_indices, original_values) sparse delta per parameter.
    - All ops in param.dtype.

    Returns:
        WeightSnapshot — sparse deltas for restoration.
    """
    raise NotImplementedError


def restore_neurons(model: object, snapshot: WeightSnapshot) -> None:
    """Restore FFN weights to pre-death values via sparse scatter."""
    raise NotImplementedError


def compute_functional_clusters(
    model: object,
    layer_idx: int,
    n_clusters: int = 64,
) -> List[List[int]]:
    """Group FFN neurons by activation correlation into functional clusters.

    This must be called on the *undamaged* model using representative input
    activations. Results should be cached and reused across damage epochs.

    Args:
        model: Qwen-32B model instance (undamaged).
        layer_idx: Which layer's FFN to cluster.
        n_clusters: Number of functional clusters to produce.

    Returns:
        List of clusters, each a list of neuron indices (FFN intermediate dim).
    """
    raise NotImplementedError
