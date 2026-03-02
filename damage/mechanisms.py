"""
damage/mechanisms.py — Top-level damage orchestration.

Coordinates noise, pruning, neuron death, and connectivity disruption
according to a DiseaseState. Called by the damage_context manager in
qwen_hooks.py via _apply_static_damage().

Phase implementation status
---------------------------
Phase 1 (implemented):
  - Noise injection       → damage/noise.py
  - Weight pruning        → damage/pruning.py

Phase 2 (stubs — not yet implemented):
  - FFN neuron death      → damage/neuron_death.py
  - Connectivity disruption → damage/connectivity.py

apply_all_damage() skips any mechanism whose effective rate is zero, and
logs a debug notice for Phase 2 mechanisms that are not yet implemented.
"""
from __future__ import annotations

import logging
from typing import List

from disease_state import DiseaseState, WeightSnapshot

logger = logging.getLogger(__name__)


def apply_all_damage(model: object, state: DiseaseState) -> WeightSnapshot:
    """Apply every active damage mechanism for the current DiseaseState.

    Order of operations matches clinical progression:
      1. Noise injection        (synaptic dysfunction — Phase 1)
      2. Weight pruning         (synaptic elimination — Phase 1/2)
      3. FFN neuron death       (structural loss — Phase 2, stub)
      4. Connectivity disruption (positional/residual — Phase 2, stub)

    Returns a WeightSnapshot (list of sparse deltas) that qwen_hooks uses to
    restore weights at damage_context.__exit__.

    Args:
        model: Qwen HuggingFace CausalLM instance.
        state: Current DiseaseState specifying what damage to apply.

    Returns:
        WeightSnapshot — combined sparse deltas from all applied mechanisms.
    """
    snapshot: WeightSnapshot = []
    cfg = state.damage_config

    # ------------------------------------------------------------------
    # 1. Noise injection
    # ------------------------------------------------------------------
    if cfg.effective_noise_std() > 0:
        from damage.noise import inject_noise
        snapshot.extend(inject_noise(model, state))

    # ------------------------------------------------------------------
    # 2. Weight pruning
    # ------------------------------------------------------------------
    if cfg.effective_prune_rate() > 0:
        from damage.pruning import prune_weights
        snapshot.extend(prune_weights(model, state))

    # ------------------------------------------------------------------
    # 3. FFN neuron death (Phase 2 — not yet implemented)
    # ------------------------------------------------------------------
    if cfg.effective_kill_rate() > 0:
        logger.debug(
            "apply_all_damage: neuron death (kill_rate=%.4f) skipped — "
            "damage/neuron_death.py not yet implemented.",
            cfg.effective_kill_rate(),
        )

    # ------------------------------------------------------------------
    # 4. Connectivity disruption (Phase 2 — not yet implemented)
    # ------------------------------------------------------------------
    conn = cfg.connectivity
    if conn.disrupt_positional_encoding or conn.disrupt_residual_stream:
        logger.debug(
            "apply_all_damage: connectivity disruption skipped — "
            "damage/connectivity.py not yet implemented.",
        )

    return snapshot


def restore_all(model: object, snapshot: WeightSnapshot) -> None:
    """Undo all in-place weight modifications applied by apply_all_damage.

    In normal usage this is called automatically by qwen_hooks via
    _restore_static_damage, which handles dispatch by delta_type.
    This function is provided as a standalone convenience for testing.
    """
    from qwen_hooks import _restore_static_damage
    _restore_static_damage(model, snapshot)
