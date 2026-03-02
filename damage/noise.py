"""
damage/noise.py — Gaussian noise injection (synaptic dysfunction).

Maps to Phase 1 (amyloid-dominant) damage. Noise is applied additively to
all weight elements in targeted components of each affected layer. Because
every element is modified, the delta stores the noise tensor itself rather
than sparse indices, and restoration subtracts it back:

    apply:   param.data += noise          (noise in param.dtype)
    restore: param.data -= noise

BF16/FP16 note: (x + n) - n ≈ x with rounding error ≤ 1 ULP per operation.
In practice the error is ≪ the noise magnitude itself and is acceptable for
inference; exact restoration requires FP32 (used in sanity_check_noise).

Depth scaling: layer 0 (entorhinal analogue, most vulnerable) receives the
full base_std; the last affected layer receives 0.5 × base_std. All layers
between are linearly interpolated. This mirrors the entorhinal-first pattern
of amyloid-driven synaptic dysfunction.

Delta format produced:
    {
        "delta_type":  "additive_noise",
        "param_name":  str,           # exact key from model.named_parameters()
        "noise":       torch.Tensor,  # same shape and dtype as the parameter
    }
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from disease_state import (
    BraakStage,
    DamageConfig,
    DamageIntensity,
    DamagePhase,
    DiseaseState,
    NoiseConfig,
    SparseDeltaT,
    WeightSnapshot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_noise(model: object, state: DiseaseState) -> WeightSnapshot:
    """Inject Gaussian noise into weights of all affected layers.

    Reads ``state.damage_config.noise`` for component flags and
    ``state.damage_config.effective_noise_std()`` for the base standard
    deviation.  All operations are in-place on ``param.data`` and run under
    the caller's ``torch.no_grad()`` context.

    Updates ``LayerDamageState.noise_level`` for each affected layer.

    Args:
        model: Qwen-32B HuggingFace CausalLM (or mock with the same structure).
        state: Current DiseaseState.

    Returns:
        WeightSnapshot — one ``"additive_noise"`` delta per modified parameter.
    """
    cfg: NoiseConfig = state.damage_config.noise
    base_std: float = state.damage_config.effective_noise_std()
    affected: range = state.affected_layers
    snapshot: WeightSnapshot = []

    # Build name lookup once: id(parameter) → parameter name in named_parameters().
    id_to_name: Dict[int, str] = {
        id(p): name for name, p in model.named_parameters()
    }

    rng = torch.Generator()
    if cfg.seed is not None:
        rng.manual_seed(cfg.seed)

    with torch.no_grad():
        # Input embedding layer — not depth-scaled, sits at model boundary.
        if cfg.apply_to_embeddings:
            param = model.model.embed_tokens.weight
            noise = _randn_like(param, base_std, rng)
            param.data.add_(noise)
            snapshot.append(_noise_delta(id_to_name[id(param)], noise))

        for layer_idx in affected:
            scale = _depth_scale(layer_idx, affected) if cfg.scale_with_layer_depth else 1.0
            layer_std = base_std * scale
            layer = model.model.layers[layer_idx]

            if cfg.apply_to_attention:
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    param = getattr(layer.self_attn, proj_name).weight
                    noise = _randn_like(param, layer_std, rng)
                    param.data.add_(noise)
                    snapshot.append(_noise_delta(id_to_name[id(param)], noise))

            if cfg.apply_to_ffn:
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    param = getattr(layer.mlp, proj_name).weight
                    noise = _randn_like(param, layer_std, rng)
                    param.data.add_(noise)
                    snapshot.append(_noise_delta(id_to_name[id(param)], noise))

            if cfg.apply_to_layer_norm:
                for ln_name in ("input_layernorm", "post_attention_layernorm"):
                    param = getattr(layer, ln_name).weight
                    noise = _randn_like(param, layer_std, rng)
                    param.data.add_(noise)
                    snapshot.append(_noise_delta(id_to_name[id(param)], noise))

            # Accumulate damage record.  Clamp to [0, 1]; caller aggregates
            # across epochs by calling DiseaseState.record_damage_increment().
            ls = state.layer_state(layer_idx)
            ls.noise_level = min(1.0, ls.noise_level + layer_std)

    return snapshot


def restore_noise(model: object, snapshot: WeightSnapshot) -> None:
    """Subtract stored noise tensors to undo inject_noise.

    Called automatically by qwen_hooks._restore_static_damage via the delta
    dispatch table.  Also callable standalone for testing.

    In BF16/FP16 the restoration is approximate (≤1 ULP error per element).
    Use FP32 tensors for exact round-trip verification (see sanity_check_noise).
    """
    param_map: Dict[str, object] = dict(model.named_parameters())

    with torch.no_grad():
        for delta in snapshot:
            if delta.get("delta_type") != "additive_noise":
                continue
            pname: str = delta["param_name"]
            if pname not in param_map:
                logger.warning("restore_noise: %r not found, skipping.", pname)
                continue
            param_map[pname].data.sub_(delta["noise"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _randn_like(param: torch.Tensor, std: float, rng: torch.Generator) -> torch.Tensor:
    """Return Gaussian noise matching ``param``'s dtype, device, and shape.

    Noise is generated directly in param.dtype — no FP32 intermediate.
    ``torch.randn`` with ``dtype=torch.bfloat16`` samples natively in BF16.
    """
    return torch.randn(
        param.shape,
        dtype=param.dtype,
        device=param.device,
        generator=rng,
    ).mul_(std)


def _depth_scale(layer_idx: int, affected: range) -> float:
    """Linear depth-based noise scale over the affected layer range.

    Returns 1.0 for the first affected layer, 0.5 for the last.
    Earlier layers (closer to input, more AD-vulnerable) receive stronger noise.

        scale(layer_idx) = 1.0 - 0.5 * (layer_idx - start) / (n - 1)

    With n=1 (single affected layer) the scale is always 1.0.
    """
    n = len(affected)
    if n <= 1:
        return 1.0
    relative = (layer_idx - affected.start) / (n - 1)   # 0.0 → 1.0
    return 1.0 - 0.5 * relative                          # 1.0 → 0.5


def _noise_delta(param_name: str, noise: torch.Tensor) -> SparseDeltaT:
    return {
        "delta_type": "additive_noise",
        "param_name": param_name,
        "noise":      noise,
    }


# ---------------------------------------------------------------------------
# Mock model for sanity check (no real model required)
# ---------------------------------------------------------------------------


class _MockAttention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _MockMLP(nn.Module):
    def __init__(self, d: int, ffn_d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, ffn_d, bias=False)
        self.up_proj   = nn.Linear(d, ffn_d, bias=False)
        self.down_proj = nn.Linear(ffn_d, d, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, d: int, ffn_d: int) -> None:
        super().__init__()
        self.self_attn = _MockAttention(d)
        self.mlp = _MockMLP(d, ffn_d)
        self.input_layernorm = nn.LayerNorm(d)
        self.post_attention_layernorm = nn.LayerNorm(d)


class _MockQwenModel(nn.Module):
    def __init__(self, n_layers: int, d: int, ffn_d: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, d)
        self.layers = nn.ModuleList([_MockLayer(d, ffn_d) for _ in range(n_layers)])


class _MockQwen(nn.Module):
    """Minimal Qwen-like model for sanity checking — CPU, FP32, tiny dims."""

    def __init__(self, n_layers: int = 6, d: int = 16, ffn_d: int = 32) -> None:
        super().__init__()
        self.model = _MockQwenModel(n_layers, d, ffn_d)
        self.lm_head = nn.Linear(d, 64, bias=False)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def sanity_check_noise(verbose: bool = True) -> bool:
    """Verify noise injection and restoration on a small CPU/FP32 mock model.

    Tests
    -----
    1. All targeted weight tensors are modified after inject_noise.
    2. Layer-norm weights are NOT touched when apply_to_layer_norm=False.
    3. Depth scaling: layer 0 receives more noise than the last affected layer.
    4. LayerDamageState.noise_level is positive for all affected layers.
    5. restore_noise returns all weights to their original values within
       FP32 tolerance (atol=1e-6).  FP32 is used so rounding is not a concern.

    Returns True on success; raises AssertionError with a descriptive message
    on the first failure.
    """
    _log = print if verbose else (lambda *a, **k: None)
    _log("=" * 60)
    _log("sanity_check_noise — Gaussian noise injection & restoration")
    _log("=" * 60)

    # This sanity check uses a 6-layer mock whose BraakStage.I_II range
    # (layers 0–5) matches only when the active arch has 36 layers (QWEN_3B).
    # Save and restore the active arch so the check is self-contained.
    from disease_state import get_active_arch, set_active_arch, QWEN_3B
    _saved_arch = get_active_arch()
    set_active_arch(QWEN_3B)

    # --- Build mock model (FP32, CPU, 6 layers matching BraakStage.I_II) ---
    N_LAYERS = 6   # BraakStage.I_II affects layers 0-5 under QWEN_3B (36 layers)
    D, FFN_D = 16, 32
    mock = _MockQwen(n_layers=N_LAYERS, d=D, ffn_d=FFN_D)

    # Snapshot original weights before any modification.
    originals: Dict[str, torch.Tensor] = {
        name: param.data.clone()
        for name, param in mock.named_parameters()
    }

    # Build a DiseaseState whose affected_layers == range(0, 6).
    state = DiseaseState.from_braak_stage(
        stage=BraakStage.I_II,
        phase=DamagePhase.AMYLOID,
        intensity=DamageIntensity.MILD,
    )
    # Configure noise explicitly for a clear signal.
    state.damage_config.noise.noise_std = 0.1
    state.damage_config.noise.apply_to_embeddings = True
    state.damage_config.noise.apply_to_attention = True
    state.damage_config.noise.apply_to_ffn = True
    state.damage_config.noise.apply_to_layer_norm = False   # must stay pristine
    state.damage_config.noise.scale_with_layer_depth = True
    state.damage_config.noise.seed = 42
    state.damage_config.noise_std_override = 0.1            # base_std = 0.1

    # --- Apply noise ---
    with torch.no_grad():
        snapshot = inject_noise(mock, state)

    _log(f"  snapshot entries: {len(snapshot)}")

    # Test 1 — targeted components changed; untargeted ones did not.
    # Targeted: embed_tokens (apply_to_embeddings), per-layer attn projections
    # (apply_to_attention), per-layer FFN projections (apply_to_ffn).
    # Untargeted: lm_head, layer-norms (apply_to_layer_norm=False).
    _log("\n[1] Targeted weights modified; untargeted weights untouched...")
    ATTN_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
    FFN_PROJS  = {"gate_proj", "up_proj", "down_proj"}

    for name, param in mock.named_parameters():
        changed = not torch.equal(param.data, originals[name])
        parts = name.split(".")
        # Identify the component by the last two segments.
        leaf = parts[-2] if len(parts) >= 2 else ""   # e.g. "q_proj"
        is_attn_proj  = leaf in ATTN_PROJS
        is_ffn_proj   = leaf in FFN_PROJS
        is_embedding  = "embed_tokens" in name
        is_layernorm  = "layernorm" in name.lower()
        is_lm_head    = name.startswith("lm_head")
        should_change = is_attn_proj or is_ffn_proj or is_embedding
        should_stay   = is_layernorm or is_lm_head
        if should_change:
            assert changed, f"FAIL: targeted weight was not modified: {name}"
        if should_stay:
            assert not changed, f"FAIL: untargeted weight was modified: {name}"
    _log("    PASS — targeted weights changed; layer-norms and lm_head untouched.")

    # Test 2 — depth scaling: layer 0 gate_proj noise > layer 5 gate_proj noise.
    _log("\n[2] Depth scaling (layer 0 noisier than layer 5)...")
    delta_layer0 = (
        mock.model.layers[0].mlp.gate_proj.weight.data
        - originals["model.layers.0.mlp.gate_proj.weight"]
    ).abs().mean().item()
    delta_layer5 = (
        mock.model.layers[5].mlp.gate_proj.weight.data
        - originals["model.layers.5.mlp.gate_proj.weight"]
    ).abs().mean().item()
    assert delta_layer0 > delta_layer5, (
        f"FAIL: layer 0 noise ({delta_layer0:.5f}) should exceed "
        f"layer 5 noise ({delta_layer5:.5f})"
    )
    _log(f"    PASS — layer 0 mean |noise|={delta_layer0:.5f}, "
         f"layer 5 mean |noise|={delta_layer5:.5f}  "
         f"(ratio={delta_layer0 / delta_layer5:.2f}×, expected ≈2.0×).")

    # Test 3 — depth scale values are exactly right.
    _log("\n[3] _depth_scale values...")
    aff = state.affected_layers
    s0 = _depth_scale(0, aff)
    s5 = _depth_scale(5, aff)
    assert abs(s0 - 1.0) < 1e-9,  f"FAIL: _depth_scale(0) = {s0}, expected 1.0"
    assert abs(s5 - 0.5) < 1e-9,  f"FAIL: _depth_scale(5) = {s5}, expected 0.5"
    _log(f"    PASS — scale(layer 0)={s0:.3f}, scale(layer 5)={s5:.3f}.")

    # Test 4 — noise_level updated in LayerDamageState.
    _log("\n[4] LayerDamageState.noise_level updated...")
    for i in state.affected_layers:
        lvl = state.layer_state(i).noise_level
        assert lvl > 0, f"FAIL: layer {i} noise_level still 0."
    _log("    PASS — all affected layers have noise_level > 0.")

    # Test 5 — restoration.
    _log("\n[5] restore_noise returns weights to original (FP32 atol=1e-6)...")
    with torch.no_grad():
        restore_noise(mock, snapshot)
    for name, param in mock.named_parameters():
        if name not in originals:
            continue
        max_err = (param.data - originals[name]).abs().max().item()
        assert max_err < 1e-6, (
            f"FAIL: restore error too large for {name}: max|err|={max_err:.2e}"
        )
    _log("    PASS — all weights restored within tolerance.")

    _log("\n" + "=" * 60)
    _log("sanity_check_noise PASSED")
    _log("=" * 60)
    set_active_arch(_saved_arch)
    return True


if __name__ == "__main__":
    sanity_check_noise(verbose=True)
