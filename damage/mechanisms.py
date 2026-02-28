"""
damage/mechanisms.py — Top-level damage orchestration.

Coordinates noise, pruning, neuron death, and connectivity disruption
according to a DiseaseState. Called by the damage_context manager in
qwen_hooks.py.

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from disease_state import DiseaseState


def apply_all_damage(model: object, state: DiseaseState) -> None:
    """Apply every active damage mechanism for the current DiseaseState.

    Order of operations:
    1. Noise injection (synaptic dysfunction)
    2. Weight pruning (synaptic elimination)
    3. FFN neuron death (structural loss)
    4. Connectivity disruption (positional / residual / KV cache)

    Args:
        model: Qwen-32B model with weights accessible via attribute paths.
        state: Current DiseaseState specifying what damage to apply.
    """
    raise NotImplementedError


def restore_all(model: object, state: DiseaseState) -> None:
    """Undo all in-place weight modifications applied by apply_all_damage.

    Must be called in the ``finally`` branch of the damage_context manager.
    """
    raise NotImplementedError
