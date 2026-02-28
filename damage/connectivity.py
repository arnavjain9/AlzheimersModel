"""
damage/connectivity.py — Residual stream and positional encoding disruption.

Mechanisms
----------
Positional encoding disruption
    Adds noise to RoPE frequency tensors or position embedding tables.
    Maps to episodic memory failure (recent context becomes unreliable).

KV cache corruption
    Corrupts a fraction of the key/value tensors in the runtime KV cache.
    Maps to working memory failure (intra-context attention degrades).
    NOTE: KV cache = working memory, NOT episodic memory.

Residual stream noise
    Adds noise to residual connections between layers, disrupting
    information flow without targeting any single component.

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from disease_state import ConnectivityConfig, DiseaseState, WeightSnapshot


def disrupt_connectivity(model: object, state: DiseaseState) -> WeightSnapshot:
    """Apply positional encoding and residual stream disruption.

    Efficiency requirements:
    - Positional encoding noise: generate in param.dtype, store sparse delta.
    - KV cache corruption is handled at runtime by a forward hook registered
      in qwen_hooks._register_hooks — do NOT do it here (no KV cache at
      static weight-mutation time).

    Returns:
        WeightSnapshot — sparse deltas for restoration.
    """
    raise NotImplementedError


def corrupt_kv_cache(kv_cache: object, state: DiseaseState) -> None:
    """Corrupt a fraction of the runtime KV cache in-place.

    This is called during a forward pass hook, not during the static
    weight-mutation phase. The KV cache object is the runtime past_key_values
    tuple produced by HuggingFace Transformers.
    """
    raise NotImplementedError


def restore_connectivity(model: object, snapshot: dict) -> None:
    """Restore positional encoding and residual stream weights."""
    raise NotImplementedError
