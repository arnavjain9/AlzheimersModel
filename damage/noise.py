"""
damage/noise.py — Gaussian noise injection (synaptic dysfunction).

Maps to Phase 1 (amyloid-dominant) damage. Noise is applied additively to
weight tensors in the affected layers. Noise is scaled by layer depth so
that earlier layers (closer to input, higher AD vulnerability) receive
stronger perturbation.

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from disease_state import DiseaseState, NoiseConfig, WeightSnapshot


def inject_noise(model: object, state: DiseaseState) -> WeightSnapshot:
    """Inject Gaussian noise into weights of affected layers.

    Efficiency requirements:
    - Generate noise directly in param.dtype (BF16/FP16). Never upcast to FP32.
      Use: noise = torch.randn(param.shape, dtype=param.dtype, device=param.device)
    - Store only modified (index, original_value) pairs — not full tensors.
    - Operate under torch.no_grad() (guaranteed by damage_context caller).

    Args:
        model: Qwen-32B model instance.
        state: Current DiseaseState (reads noise params from damage_config.noise).

    Returns:
        WeightSnapshot — sparse deltas only, one entry per modified parameter.
    """
    raise NotImplementedError


def restore_noise(model: object, snapshot: WeightSnapshot) -> None:
    """Restore weights to pre-noise values via sparse scatter.

    For each entry: param.data.view(-1).scatter_(0, flat_indices, original_values)
    """
    raise NotImplementedError
