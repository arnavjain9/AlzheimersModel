"""
qwen_hooks.py — Hook infrastructure and damage context manager.

Runtime efficiency contracts (enforced here, not just documented)
-----------------------------------------------------------------
1. torch.no_grad() wraps the entire context — applied once at entry.
2. Hooks registered ONCE at entry, removed ONCE in the finally block.
3. WeightSnapshot = sparse deltas only. Restore via scatter_, not copy.
4. All ops in param.dtype — upcasting to FP32 is a bug.
5. Masks read from LayerDamageState — never recomputed inside a hook.

Component attribute paths (from CLAUDE.md)
------------------------------------------
Layer i attention : model.model.layers[i].self_attn
FFN gate          : model.model.layers[i].mlp.gate_proj
KV cache output   : last element of self_attn forward output tuple (k, v)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Generator, List, Optional, Tuple

from disease_state import DiseaseState, WeightSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def damage_context(
    model: object,
    state: DiseaseState,
) -> Generator[None, None, None]:
    """Apply structural damage to ``model`` for the duration of the block.

    Usage::

        with damage_context(model, disease_state):
            outputs = model.generate(inputs, use_cache=True)

    Sequence of operations
    ----------------------
    __enter__:
      1. torch.no_grad() activated for the full block.
      2. Pruning masks computed (if not already cached in LayerDamageState).
      3. Static weight mutations applied (noise, pruning, neuron death,
         positional encoding) — all via sparse deltas.
      4. Forward hooks registered for KV cache corruption (runtime only).

    __exit__ (always, even on exception):
      5. All forward hooks removed.
      6. Original weights restored via sparse scatter.

    Args:
        model: Loaded Qwen-32B HuggingFace CausalLM instance.
        state: DiseaseState specifying what damage to apply.
    """
    import torch  # local import keeps disease_state.py torch-free

    snapshot: WeightSnapshot = []
    hook_handles: List = []

    with torch.no_grad():
        try:
            _ensure_masks_computed(model, state)
            snapshot = _apply_static_damage(model, state)
            hook_handles = _register_hooks(model, state)
            yield

        finally:
            # Hooks first — they may fire during cleanup in some generation
            # loops, so remove before restoring weights.
            for handle in hook_handles:
                handle.remove()
            hook_handles.clear()

            # Sparse scatter restore — runs even if an exception was raised.
            if snapshot:
                _restore_static_damage(model, snapshot)


# ---------------------------------------------------------------------------
# Mask computation
# ---------------------------------------------------------------------------


def _ensure_masks_computed(model: object, state: DiseaseState) -> None:
    """Compute and cache pruning masks for any layer where they're stale.

    Delegates to damage.pruning.compute_pruning_masks, which writes
    flat weight indices into each LayerDamageState.pruning_mask_indices
    and sets masks_computed=True.

    Neuron death indices (dead_neuron_indices) are expected to have been
    populated by the spreading-dynamics logic before damage_context is entered.
    """
    needs_recompute = any(
        not ls.masks_computed
        for i, ls in state.layer_states.items()
        if i in state.affected_layers
    )
    if needs_recompute:
        from damage.pruning import compute_pruning_masks
        compute_pruning_masks(model, state)


# ---------------------------------------------------------------------------
# Static damage application / restoration
# ---------------------------------------------------------------------------


def _apply_static_damage(model: object, state: DiseaseState) -> WeightSnapshot:
    """Delegate all static weight mutations to damage.mechanisms.

    Returns a WeightSnapshot (list of sparse deltas) for restoration.
    """
    from damage.mechanisms import apply_all_damage
    return apply_all_damage(model, state)


def _restore_static_damage(model: object, snapshot: WeightSnapshot) -> None:
    """Restore all static weight mutations using the delta-type dispatch table.

    Supported delta types
    ---------------------
    ``"additive_noise"``
        Restore by subtraction: ``param.data -= delta["noise"]``.
        Note: in BF16/FP16 ``(x + n) - n ≈ x`` with tiny rounding error.

    ``"sparse_zero_with_compensation"``
        1. Undo compensation scale: ``param.data /= delta["comp_mult"]``
           (zeroed positions stay 0; surviving weights revert to pre-prune).
        2. Scatter original values back: ``scatter_(0, flat_indices, original_values)``.

    Runs unconditionally in the finally block. Logs warnings on missing
    parameters rather than raising, so all other parameters are still restored.
    """
    if not snapshot:
        return

    import torch

    param_map: dict = dict(model.named_parameters())

    # Iterate in REVERSE application order (undo-stack semantics).
    # When noise and pruning are both applied to the same parameter,
    # the pruning delta must be undone first, then the noise delta.
    # Example: param = (orig + noise) * comp_mult
    #   Step 1 (reversed): div comp_mult → orig + noise; scatter → orig + noise
    #   Step 2 (reversed): subtract noise → orig  ✓
    # Forward order would subtract noise from the scaled weight first, giving wrong result.
    for delta in reversed(snapshot):
        param_name: str = delta["param_name"]
        delta_type: str = delta.get("delta_type", "sparse_zero")

        if param_name not in param_map:
            logger.warning(
                "restore: parameter %r not found — skipping (model may be stale).",
                param_name,
            )
            continue

        param = param_map[param_name]

        if delta_type == "additive_noise":
            # Additive restore: subtract the stored noise tensor in-place.
            # noise was generated in param.dtype, so no upcast needed.
            param.data.sub_(delta["noise"])

        elif delta_type == "sparse_zero_with_compensation":
            comp_mult: float = delta.get("comp_mult", 1.0)
            if comp_mult != 1.0:
                # Undo compensation scale on surviving weights.
                # Zeroed positions: 0 / comp_mult = 0 — correct, scatter fixes them.
                param.data.div_(comp_mult)
            param.data.view(-1).scatter_(
                0, delta["flat_indices"], delta["original_values"]
            )

        else:
            # Fallback: plain sparse-index restore (no compensation).
            param.data.view(-1).scatter_(
                0, delta["flat_indices"], delta["original_values"]
            )


# ---------------------------------------------------------------------------
# Forward hook registration
# ---------------------------------------------------------------------------


def _register_hooks(model: object, state: DiseaseState) -> List:
    """Register forward hooks for runtime (per-forward-pass) damage.

    Currently handles:
    - KV cache corruption  →  simulates working memory failure
    - (Positional noise is static — handled in damage.connectivity)

    Each hook is registered exactly once per affected attention module.
    Returns a list of hook handles for removal at context exit.
    """
    handles: List = []

    connectivity = state.damage_config.connectivity
    if connectivity.kv_cache_corruption_rate <= 0.0:
        return handles

    rate = connectivity.kv_cache_corruption_rate
    base_seed = connectivity.seed

    for layer_idx in state.affected_layers:
        attn_module = model.model.layers[layer_idx].self_attn
        hook_fn = _make_kv_corruption_hook(
            layer_idx=layer_idx,
            corruption_rate=rate,
            base_seed=base_seed,
        )
        handles.append(attn_module.register_forward_hook(hook_fn))

    return handles


def _make_kv_corruption_hook(
    layer_idx: int,
    corruption_rate: float,
    base_seed: Optional[int],
) -> Callable:
    """Factory: return a forward hook that corrupts KV outputs in-place.

    Qwen self_attn forward returns:
        (attn_output, attn_weights_or_None, past_key_value_or_None)
    where past_key_value — when present — is a (key, value) tensor tuple.

    The hook zeroes a random ``corruption_rate`` fraction of K and V elements
    before they are cached, modelling transient working memory corruption.

    Operates under the outer torch.no_grad() but adds its own guard in case
    the hook is somehow invoked outside the damage_context (defensive).
    """
    import torch

    # Deterministic per-layer RNG so corruption is reproducible.
    rng = torch.Generator()
    if base_seed is not None:
        rng.manual_seed(base_seed + layer_idx)

    def hook(
        module: object,
        inputs: Tuple,
        outputs: Tuple,
    ) -> Optional[Tuple]:
        if not isinstance(outputs, tuple) or len(outputs) < 1:
            return outputs

        # past_key_value is the last element when use_cache=True.
        # It is a 2-tuple (key_tensor, value_tensor).
        past_kv = outputs[-1]
        if not (
            isinstance(past_kv, tuple)
            and len(past_kv) == 2
            and isinstance(past_kv[0], torch.Tensor)
            and isinstance(past_kv[1], torch.Tensor)
        ):
            # No KV cache present (use_cache=False or unexpected structure).
            return outputs

        k, v = past_kv

        with torch.no_grad():
            _corrupt_tensor_inplace(k, corruption_rate, rng)
            _corrupt_tensor_inplace(v, corruption_rate, rng)

        # Return tuple with corrupted KV; leave attn_output unchanged.
        return outputs[:-1] + ((k, v),)

    return hook


def _corrupt_tensor_inplace(
    tensor: object,  # torch.Tensor
    rate: float,
    rng: object,     # torch.Generator
) -> None:
    """Zero a random fraction of ``tensor`` elements in-place.

    The Bernoulli mask is generated in float32 (required by torch.bernoulli)
    but applied via a bool cast — ``tensor`` itself is never upcast.
    """
    import torch

    # float32 only for the mask sampling step — tensor stays in its own dtype.
    mask: torch.Tensor = torch.bernoulli(
        torch.full(tensor.shape, rate, dtype=torch.float32, device=tensor.device),
        generator=rng,
    ).bool()
    tensor[mask] = 0
