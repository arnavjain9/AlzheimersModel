"""
damage/pruning.py — Weight magnitude pruning (synaptic elimination).

Zeros out the lowest-magnitude weights in target components of affected
layers. Compensation: surviving weights in the same layer are upregulated
according to the CompensationConfig, depleting the reserve.

Note: random pruning (prune_type="random") is a control condition only.
Primary implementation uses magnitude-based pruning.

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from disease_state import DiseaseState, PruningConfig, WeightSnapshot


def compute_pruning_masks(model: object, state: DiseaseState) -> None:
    """Compute and cache pruning mask indices into LayerDamageState.

    Writes flat weight indices into state.layer_states[i].pruning_mask_indices
    for every affected layer and target component.  Sets masks_computed=True.

    Must be called ONCE per epoch (before damage_context entry), not per
    forward pass.  prune_weights() reads these cached indices; it does not
    recompute them.

    Efficiency requirements:
    - Use torch.topk on the flattened parameter in its native dtype.
    - Avoid sorting the full tensor — topk is O(n log k), not O(n log n).
    - Store indices as plain Python List[int] (CPU) so they survive across
      forward passes without GPU memory pressure.
    """
    raise NotImplementedError


def prune_weights(model: object, state: DiseaseState) -> WeightSnapshot:
    """Zero out pre-computed pruning mask indices in affected layers.

    Reads indices from state.layer_states[i].pruning_mask_indices — never
    recomputes them.  Raises RuntimeError if masks_computed is False.

    Efficiency requirements:
    - In-place zero via param.data.view(-1)[flat_indices] = 0
    - Store only (flat_indices, original_values) sparse delta, not full tensor.
    - All ops in param.dtype — no FP32 upcasting.

    Returns:
        WeightSnapshot — sparse deltas for restoration.
    """
    raise NotImplementedError


def restore_pruning(model: object, snapshot: WeightSnapshot) -> None:
    """Restore pruned weights via sparse scatter."""
    raise NotImplementedError
