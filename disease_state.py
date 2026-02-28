"""
disease_state.py — Core data structures for the Alzheimer's disease simulation.

All configuration dataclasses, enums, and the central DiseaseState object live
here. Nothing in this file touches model weights or runs any damage functions.
It is pure data: the single source of truth that damage functions and evaluation
code read from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sparse delta contract — runtime efficiency
# ---------------------------------------------------------------------------
# All damage snapshot/restore functions MUST use sparse deltas, not full
# tensor copies.  For a 32B model a single weight matrix is hundreds of MB;
# copying even a handful of them per damage_context entry is prohibitive.
#
# The canonical sparse delta is a plain dict with this shape:
#
#   {
#     "param_name": str,               # e.g. "model.layers.0.mlp.gate_proj.weight"
#     "flat_indices": torch.LongTensor, # 1-D, indices of modified elements
#     "original_values": torch.Tensor, # 1-D, original values at those indices
#                                      # MUST match the parameter's own dtype
#                                      # (BF16/FP16) — never upcast to FP32
#   }
#
# A full snapshot is List[SparseDeltaT].  Use the type alias below anywhere a
# snapshot is passed or returned.
#
# SparseDeltaT and WeightSnapshot are intentionally kept torch-free here so
# disease_state.py has no heavy imports.  Implementation files import torch and
# populate these fields at runtime.

SparseDeltaT = Dict[str, Any]      # single param: {param_name, flat_indices, original_values}
WeightSnapshot = List[SparseDeltaT]  # full snapshot returned by each damage function

# ---------------------------------------------------------------------------
# Architecture configuration — model-agnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureConfig:
    """Immutable specification of a Qwen model variant's architecture.

    All layer-bound derivations (BraakStage, BrainRegion) are computed at
    runtime from ``num_layers``, so switching model size is a single call to
    ``set_active_arch()``.  No other file needs to change.

    Usage::

        from disease_state import set_active_arch, QWEN_7B
        set_active_arch(QWEN_7B)

    Verify values against the actual model's ``config.json`` before running
    experiments — load it with::

        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
        print(cfg.num_hidden_layers, cfg.hidden_size, ...)
    """

    name: str
    num_layers: int
    hidden_dim: int
    ffn_intermediate: int
    num_q_heads: int
    num_kv_heads: int
    vocab_size: int

    @property
    def gqa_ratio(self) -> int:
        """Number of Q heads sharing each KV head (Grouped Query Attention)."""
        return self.num_q_heads // self.num_kv_heads


# ---------------------------------------------------------------------------
# Pre-defined architecture configs
# Verify exact values against HuggingFace config.json for each checkpoint.
# ---------------------------------------------------------------------------

QWEN_3B = ArchitectureConfig(
    name="Qwen2.5-3B",
    num_layers=36,
    hidden_dim=2048,
    ffn_intermediate=11008,
    num_q_heads=16,
    num_kv_heads=2,
    vocab_size=151_936,
)

QWEN_7B = ArchitectureConfig(
    name="Qwen2.5-7B",
    num_layers=28,
    hidden_dim=3584,
    ffn_intermediate=18944,
    num_q_heads=28,
    num_kv_heads=4,
    vocab_size=151_936,
)

QWEN_32B = ArchitectureConfig(
    name="Qwen2.5-32B",
    num_layers=64,
    hidden_dim=5120,
    ffn_intermediate=27648,
    num_q_heads=40,
    num_kv_heads=8,
    vocab_size=151_936,
)

# Active model — default to 3B (fits on Colab A100 40 GB in bfloat16).
# Change with set_active_arch(QWEN_7B) before creating any DiseaseState.
_active_arch: ArchitectureConfig = QWEN_3B


def get_active_arch() -> "ArchitectureConfig":
    """Return the currently active architecture configuration."""
    return _active_arch


def set_active_arch(config: "ArchitectureConfig") -> None:
    """Switch the active model architecture.

    Call once at the top of a notebook/script *before* creating DiseaseState
    objects.  BraakStage bounds and BrainRegion layer ranges update
    automatically — no other code needs to change.

    Example::

        from disease_state import set_active_arch, QWEN_7B
        set_active_arch(QWEN_7B)
        state = DiseaseState.from_braak_stage(BraakStage.III_IV, ...)
    """
    global _active_arch
    _active_arch = config

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BraakStage(Enum):
    """Braak neurofibrillary tangle staging for Alzheimer's disease.

    Each stage defines the anatomical spread of tau pathology and maps to a
    contiguous prefix of transformer layers, per the layer-to-brain-region
    mapping in CLAUDE.md.

    Severity caps prevent individual stages from applying damage beyond
    clinically observed bounds — later stages allow more total damage.
    """

    I_II = "I-II"      # Transentorhinal — layers 0–5,  severity cap 0.3
    III_IV = "III-IV"  # Limbic          — layers 0–18, severity cap 0.5
    V = "V"            # Neocortical early — layers 0–29, severity cap 0.7
    VI = "VI"          # Neocortical late  — layers 0–35, severity cap 1.0

    @property
    def layer_bounds(self) -> Tuple[int, int]:
        """(start, end_inclusive) indices of affected layers.

        Bounds are derived as a fraction of the active architecture's layer
        count, calibrated to a 36-layer reference model:
            I-II   → first ~14% of layers (ref: 0–5)
            III-IV → first ~50% of layers (ref: 0–18)
            V      → first ~83% of layers (ref: 0–29)
            VI     → all layers           (ref: 0–35)

        Switch models with ``set_active_arch()`` — bounds update automatically.
        """
        n = get_active_arch().num_layers
        end = {
            BraakStage.I_II:   round(5  / 35 * (n - 1)),
            BraakStage.III_IV: round(18 / 35 * (n - 1)),
            BraakStage.V:      round(29 / 35 * (n - 1)),
            BraakStage.VI:     n - 1,
        }[self]
        return (0, end)

    @property
    def layer_range(self) -> range:
        """Iterate over affected layer indices."""
        start, end = self.layer_bounds
        return range(start, end + 1)

    @property
    def severity_cap(self) -> float:
        """Maximum damage intensity allowed at this stage."""
        return {
            BraakStage.I_II:   0.3,
            BraakStage.III_IV: 0.5,
            BraakStage.V:      0.7,
            BraakStage.VI:     1.0,
        }[self]

    @property
    def ordinal(self) -> int:
        """Numeric progression order (1 = earliest)."""
        return {
            BraakStage.I_II:   1,
            BraakStage.III_IV: 2,
            BraakStage.V:      3,
            BraakStage.VI:     4,
        }[self]


class DamagePhase(Enum):
    """Two-phase disease model corresponding to amyloid and tau dominance.

    Phase 1 (Amyloid): global low-level noise, mild synaptic dysfunction,
    relatively preserved structural integrity.

    Phase 2 (Tau): focal neuron death spreading along functional connectivity,
    structural degradation, compensation reserve depletion.
    """

    AMYLOID = 1  # Phase 1 — amyloid-dominant, diffuse mild damage
    TAU = 2      # Phase 2 — tau-dominant, focal structural degradation


class DamageIntensity(Enum):
    """Coarse-grained severity tiers, each with defined parameter ranges.

    These map to the table in CLAUDE.md:
      Subclinical | noise 0.01–0.05 | prune 0–5%    | kill 0–2%
      Mild        | noise 0.05–0.15 | prune 5–15%   | kill 2–10%
      Moderate    | noise 0.15–0.30 | prune 15–30%  | kill 10–25%
      Severe      | noise 0.30+     | prune 30%+    | kill 25%+
    """

    SUBCLINICAL = "subclinical"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

    @property
    def noise_range(self) -> Tuple[float, float]:
        """(min_std, max_std) for Gaussian noise injection."""
        return {
            DamageIntensity.SUBCLINICAL: (0.01, 0.05),
            DamageIntensity.MILD:        (0.05, 0.15),
            DamageIntensity.MODERATE:    (0.15, 0.30),
            DamageIntensity.SEVERE:      (0.30, 1.00),
        }[self]

    @property
    def pruning_range(self) -> Tuple[float, float]:
        """(min_rate, max_rate) for weight pruning."""
        return {
            DamageIntensity.SUBCLINICAL: (0.00, 0.05),
            DamageIntensity.MILD:        (0.05, 0.15),
            DamageIntensity.MODERATE:    (0.15, 0.30),
            DamageIntensity.SEVERE:      (0.30, 1.00),
        }[self]

    @property
    def neuron_kill_range(self) -> Tuple[float, float]:
        """(min_rate, max_rate) for FFN neuron death."""
        return {
            DamageIntensity.SUBCLINICAL: (0.00, 0.02),
            DamageIntensity.MILD:        (0.02, 0.10),
            DamageIntensity.MODERATE:    (0.10, 0.25),
            DamageIntensity.SEVERE:      (0.25, 1.00),
        }[self]

    @property
    def midpoint_noise(self) -> float:
        lo, hi = self.noise_range
        return (lo + hi) / 2

    @property
    def midpoint_prune(self) -> float:
        lo, hi = self.pruning_range
        return (lo + hi) / 2

    @property
    def midpoint_kill(self) -> float:
        lo, hi = self.neuron_kill_range
        return (lo + hi) / 2


class BrainRegion(Enum):
    """Functional brain region analogues for Qwen transformer layers.

    Vulnerability order follows known AD progression:
    entorhinal → hippocampal → temporal association → prefrontal.

    Layer ranges are percentage-derived from the active architecture's
    ``num_layers`` (see ``set_active_arch()``), so the same enum works across
    3B (36 layers), 7B (28 layers), and 32B (64 layers) models.
    """

    ENTORHINAL = "entorhinal"                      # First ~14% of layers
    HIPPOCAMPAL = "hippocampal"                    # ~14–34% of layers
    TEMPORAL_ASSOCIATION = "temporal_association"  # ~34–69% of layers
    PREFRONTAL = "prefrontal"                      # ~69–100% of layers

    @property
    def layer_range(self) -> range:
        """Layer range for this region in the active architecture.

        Boundary fractions calibrated to the 36-layer reference model:
            ENTORHINAL ends at          5/35 ≈ 14%  (ref end: layer 5)
            HIPPOCAMPAL ends at        12/35 ≈ 34%  (ref end: layer 12)
            TEMPORAL_ASSOCIATION ends at 24/35 ≈ 69%  (ref end: layer 24)
            PREFRONTAL ends at last layer            (ref end: layer 35)

        Regions are forced contiguous: each start = previous end + 1.
        For the 36-layer ref model this reproduces [0-5],[6-12],[13-24],[25-35].
        """
        n = get_active_arch().num_layers
        e0 = round(5  / 35 * (n - 1))  # ENTORHINAL end
        e1 = round(12 / 35 * (n - 1))  # HIPPOCAMPAL end
        e2 = round(24 / 35 * (n - 1))  # TEMPORAL_ASSOCIATION end
        e3 = n - 1                      # PREFRONTAL end
        return {
            BrainRegion.ENTORHINAL:           range(0,      e0 + 1),
            BrainRegion.HIPPOCAMPAL:          range(e0 + 1, e1 + 1),
            BrainRegion.TEMPORAL_ASSOCIATION: range(e1 + 1, e2 + 1),
            BrainRegion.PREFRONTAL:           range(e2 + 1, e3 + 1),
        }[self]

    @property
    def ad_vulnerability_order(self) -> int:
        """Vulnerability rank — 1 = most vulnerable (damaged earliest)."""
        return {
            BrainRegion.ENTORHINAL:           1,
            BrainRegion.HIPPOCAMPAL:          2,
            BrainRegion.TEMPORAL_ASSOCIATION: 3,
            BrainRegion.PREFRONTAL:           4,
        }[self]


class DamageTarget(Enum):
    """Fine-grained model component targets for damage functions.

    Attribute paths follow the Qwen-32B layout documented in CLAUDE.md.
    """

    EMBEDDING = "embedding"               # model.embed_tokens
    ATTENTION_Q = "attention_q"           # self_attn.q_proj
    ATTENTION_K = "attention_k"           # self_attn.k_proj
    ATTENTION_V = "attention_v"           # self_attn.v_proj
    ATTENTION_O = "attention_o"           # self_attn.o_proj
    FFN_GATE = "ffn_gate"                 # mlp.gate_proj
    FFN_UP = "ffn_up"                     # mlp.up_proj
    FFN_DOWN = "ffn_down"                 # mlp.down_proj
    LAYER_NORM = "layer_norm"             # input_layernorm / post_attention_layernorm
    KV_CACHE = "kv_cache"                 # Runtime KV cache (working memory)
    POSITIONAL_ENCODING = "positional_encoding"  # Positional enc. (episodic memory)
    LM_HEAD = "lm_head"                   # lm_head output projection


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NoiseConfig:
    """Parameters governing Gaussian noise injection (synaptic dysfunction).

    In the two-phase model, noise is the primary Phase 1 (amyloid) mechanism.
    It produces diffuse, global signal degradation without structural loss.

    Notes
    -----
    - ``scale_with_layer_depth=True`` applies stronger noise to lower-indexed
      layers, mirroring entorhinal-first vulnerability.
    - Layer norm parameters are stable until very late in AD; keep
      ``apply_to_layer_norm=False`` for all but Stage VI simulations.
    """

    noise_std: float = 0.01
    noise_type: str = "gaussian"           # "gaussian" | "structured" | "correlated"
    apply_to_embeddings: bool = True
    apply_to_attention: bool = True
    apply_to_ffn: bool = True
    apply_to_layer_norm: bool = False      # Last to degrade in real AD
    scale_with_layer_depth: bool = True    # Stronger at lower (earlier) layer indices
    seed: Optional[int] = None


@dataclass
class PruningConfig:
    """Parameters governing weight magnitude pruning (synaptic elimination).

    When ``apply_compensation=True``, surviving weights in the same layer are
    scaled up according to the formula in CLAUDE.md:
        surviving_weights *= (1 + compensation_factor * prune_rate)

    ``prune_type="random"`` is a control condition, not the primary method.
    """

    prune_rate: float = 0.0
    prune_type: str = "magnitude"          # "magnitude" | "random" (control)
    apply_compensation: bool = True
    target_components: List[DamageTarget] = field(
        default_factory=lambda: [
            DamageTarget.FFN_GATE,
            DamageTarget.FFN_UP,
            DamageTarget.FFN_DOWN,
            DamageTarget.ATTENTION_Q,
            DamageTarget.ATTENTION_K,
            DamageTarget.ATTENTION_V,
        ]
    )
    seed: Optional[int] = None


@dataclass
class NeuronDeathConfig:
    """Parameters governing FFN neuron death with functional clustering.

    Neurons must be killed within functional clusters (grouped by activation
    correlation), not randomly. Random killing is a control condition only.

    ``spreading_rate`` controls how quickly death propagates outward from the
    seed layer per epoch, modelling trans-synaptic tau spread.
    """

    kill_rate: float = 0.0
    use_functional_clustering: bool = True  # False = random control condition
    cluster_method: str = "activation_correlation"
    spreading_rate: float = 0.1            # Fraction of seed death that spreads per epoch
    seed_layer: Optional[int] = None       # None → default to layer 0 (entorhinal)
    seed: Optional[int] = None


@dataclass
class ConnectivityConfig:
    """Parameters governing residual stream and positional encoding disruption.

    Positional encoding disruption → episodic memory failure (early context
    tokens become unreliable).

    KV cache corruption → working memory failure (intra-context attention
    degrades). Note: KV cache is working memory, NOT episodic memory.
    """

    disrupt_positional_encoding: bool = False
    positional_noise_std: float = 0.0
    disrupt_residual_stream: bool = False
    residual_noise_std: float = 0.0
    kv_cache_corruption_rate: float = 0.0  # Fraction of KV entries to corrupt
    seed: Optional[int] = None


@dataclass
class CompensationConfig:
    """Models the brain's compensatory synaptic upregulation.

    When connections are pruned, surviving connections in the same layer
    upregulate. This reserve starts full and depletes as damage accumulates,
    modelling the clinical "tipping point" of rapid decline once compensation
    is exhausted.

    Effective compensation at any moment:
        effective_factor = base_compensation_factor * compensation_reserve
    """

    compensation_reserve: float = 1.0      # Starts at 1.0, depletes toward 0.0
    base_compensation_factor: float = 0.2  # Max upregulation per prune increment
    depletion_per_increment: float = 0.05  # Reserve lost per damage epoch
    min_reserve: float = 0.0              # Floor — once reached, no compensation

    def effective_compensation_factor(self) -> float:
        """Current upregulation factor, scaled by remaining reserve."""
        return self.base_compensation_factor * self.compensation_reserve

    def deplete(self, increments: int = 1) -> None:
        """Consume reserve for ``increments`` damage applications."""
        self.compensation_reserve = max(
            self.min_reserve,
            self.compensation_reserve - self.depletion_per_increment * increments,
        )

    @property
    def is_exhausted(self) -> bool:
        """True when compensation reserve has reached its floor."""
        return self.compensation_reserve <= self.min_reserve


@dataclass
class DamageConfig:
    """Full specification of damage to apply for a given disease state.

    Serves as the configuration contract between DiseaseState and the damage
    functions in damage/mechanisms.py. All effective parameter methods clamp
    to the Braak stage severity cap.
    """

    braak_stage: BraakStage
    damage_phase: DamagePhase
    intensity: DamageIntensity

    # Per-mechanism sub-configurations
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    neuron_death: NeuronDeathConfig = field(default_factory=NeuronDeathConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)

    # Optional overrides — if set, bypass intensity midpoint derivation
    noise_std_override: Optional[float] = None
    prune_rate_override: Optional[float] = None
    kill_rate_override: Optional[float] = None

    @property
    def affected_layers(self) -> range:
        return self.braak_stage.layer_range

    @property
    def severity_cap(self) -> float:
        return self.braak_stage.severity_cap

    def effective_noise_std(self) -> float:
        """Noise std to apply, capped by severity cap."""
        raw = (
            self.noise_std_override
            if self.noise_std_override is not None
            else self.intensity.midpoint_noise
        )
        return min(raw, self.severity_cap)

    def effective_prune_rate(self) -> float:
        """Pruning fraction to apply, capped by severity cap."""
        raw = (
            self.prune_rate_override
            if self.prune_rate_override is not None
            else self.intensity.midpoint_prune
        )
        return min(raw, self.severity_cap)

    def effective_kill_rate(self) -> float:
        """Neuron death fraction to apply, capped by severity cap."""
        raw = (
            self.kill_rate_override
            if self.kill_rate_override is not None
            else self.intensity.midpoint_kill
        )
        return min(raw, self.severity_cap)


# ---------------------------------------------------------------------------
# Per-layer tracking
# ---------------------------------------------------------------------------


@dataclass
class LayerDamageState:
    """Accumulated damage record for a single transformer layer.

    Updated in-place by damage functions as each damage epoch is applied.
    The composite ``total_damage`` score is used for spreading dynamics and
    clinical metric mapping.
    """

    layer_idx: int
    brain_region: BrainRegion

    # Cumulative damage levels in [0.0, 1.0]
    noise_level: float = 0.0
    prune_level: float = 0.0
    neuron_death_level: float = 0.0
    connectivity_disruption: float = 0.0

    # Structural state — also serve as precomputed mask caches.
    # Computed once per epoch by damage functions and reused on every forward
    # pass within that epoch.  Never recompute inside a forward hook.
    dead_neuron_indices: List[int] = field(default_factory=list)
    ablated_kv_heads: List[int] = field(default_factory=list)

    # Pruning mask cache: DamageTarget.value → sorted flat weight indices to zero.
    # Populated once by pruning.py and reused until the next epoch advances.
    # Keys absent from this dict mean that component has no pruning mask yet.
    pruning_mask_indices: Dict[str, List[int]] = field(default_factory=dict)

    # Set True once masks have been computed for the current epoch so damage
    # functions know they can skip recomputation.
    masks_computed: bool = False

    @property
    def total_damage(self) -> float:
        """Weighted composite damage score in [0.0, 1.0].

        Weights reflect relative contribution to functional loss:
        - Neuron death is weighted highest (most irreversible)
        - Connectivity disruption is weighted lowest (most compensatable)
        """
        return (
            0.25 * self.noise_level
            + 0.25 * self.prune_level
            + 0.35 * self.neuron_death_level
            + 0.15 * self.connectivity_disruption
        )

    @property
    def is_severely_damaged(self) -> bool:
        return self.total_damage > 0.6

    @property
    def dead_neuron_count(self) -> int:
        return len(self.dead_neuron_indices)

    @property
    def ablated_kv_head_count(self) -> int:
        return len(self.ablated_kv_heads)

    @property
    def affected_q_head_count(self) -> int:
        """Number of Q heads impaired via GQA coupling."""
        return self.ablated_kv_head_count * get_active_arch().gqa_ratio

    def invalidate_masks(self) -> None:
        """Mark precomputed masks stale so the next damage pass recomputes them.

        Call from DiseaseState.record_damage_increment() whenever epoch advances
        and pruning/death fractions change.
        """
        self.pruning_mask_indices.clear()
        self.masks_computed = False


# ---------------------------------------------------------------------------
# Main disease state
# ---------------------------------------------------------------------------


@dataclass
class DiseaseState:
    """Central state object representing a snapshot of simulated AD progression.

    This is the single source of truth passed into the damage context manager
    (``with damage_context(model, state): ...``). It encodes the current
    Braak stage, two-phase damage parameters, per-layer damage history,
    and the compensation reserve.

    Use the factory methods for standard starting points:
        DiseaseState.healthy()
        DiseaseState.from_braak_stage(BraakStage.III_IV, ...)
    """

    # Current disease parameters
    braak_stage: BraakStage
    damage_phase: DamagePhase
    intensity: DamageIntensity

    # Active damage configuration
    damage_config: DamageConfig

    # Compensation state
    compensation: CompensationConfig = field(default_factory=CompensationConfig)

    # Temporal tracking
    epoch: int = 0                         # Number of damage increments applied
    simulation_time_months: float = 0.0   # Simulated disease time in months

    # Per-layer damage records (populated in __post_init__ if empty)
    layer_states: Dict[int, LayerDamageState] = field(default_factory=dict)

    # Cumulative aggregate metrics
    total_damage_accumulated: float = 0.0

    def __post_init__(self) -> None:
        if not self.layer_states:
            self.layer_states = {
                i: LayerDamageState(
                    layer_idx=i,
                    brain_region=layer_to_brain_region(i),
                )
                for i in range(get_active_arch().num_layers)
            }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_braak_stage(
        cls,
        stage: BraakStage,
        phase: DamagePhase = DamagePhase.AMYLOID,
        intensity: DamageIntensity = DamageIntensity.MILD,
        compensation: Optional[CompensationConfig] = None,
    ) -> "DiseaseState":
        """Create a DiseaseState from a Braak stage with default sub-configs."""
        config = DamageConfig(
            braak_stage=stage,
            damage_phase=phase,
            intensity=intensity,
        )
        return cls(
            braak_stage=stage,
            damage_phase=phase,
            intensity=intensity,
            damage_config=config,
            compensation=compensation or CompensationConfig(),
        )

    @classmethod
    def healthy(cls) -> "DiseaseState":
        """Completely undamaged baseline state (subclinical Phase 1, Stage I-II)."""
        return cls.from_braak_stage(
            stage=BraakStage.I_II,
            phase=DamagePhase.AMYLOID,
            intensity=DamageIntensity.SUBCLINICAL,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def affected_layers(self) -> range:
        return self.braak_stage.layer_range

    @property
    def severity_cap(self) -> float:
        return self.braak_stage.severity_cap

    @property
    def is_phase_two(self) -> bool:
        return self.damage_phase == DamagePhase.TAU

    # ------------------------------------------------------------------
    # State mutation helpers (called by damage orchestration)
    # ------------------------------------------------------------------

    def record_damage_increment(self) -> None:
        """Advance epoch, deplete compensation, recompute aggregate damage.

        Also invalidates all precomputed pruning masks so the next epoch's
        damage pass recomputes them from the updated prune rates.

        Call once after each full damage pass has been applied.
        """
        self.epoch += 1
        self.compensation.deplete(increments=1)
        self.total_damage_accumulated = sum(
            s.total_damage for s in self.layer_states.values()
        )
        for s in self.layer_states.values():
            s.invalidate_masks()

    def layer_state(self, layer_idx: int) -> LayerDamageState:
        """Return the damage record for a specific layer."""
        return self.layer_states[layer_idx]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def severely_damaged_layers(self) -> List[int]:
        return [i for i, s in self.layer_states.items() if s.is_severely_damaged]

    def mean_damage_in_region(self, region: BrainRegion) -> float:
        """Average total_damage score across all layers in a brain region."""
        states = [self.layer_states[i] for i in region.layer_range]
        if not states:
            return 0.0
        return sum(s.total_damage for s in states) / len(states)

    def summary(self) -> Dict[str, object]:
        """Human-readable snapshot of current disease state."""
        region_damage = {
            region.value: round(self.mean_damage_in_region(region), 4)
            for region in BrainRegion
        }
        return {
            "braak_stage": self.braak_stage.value,
            "damage_phase": self.damage_phase.name,
            "intensity": self.intensity.value,
            "epoch": self.epoch,
            "simulation_time_months": self.simulation_time_months,
            "severity_cap": self.severity_cap,
            "compensation_reserve": round(self.compensation.compensation_reserve, 4),
            "compensation_exhausted": self.compensation.is_exhausted,
            "total_damage_accumulated": round(self.total_damage_accumulated, 4),
            "affected_layer_count": len(self.affected_layers),
            "severely_damaged_layers": self.severely_damaged_layers(),
            "region_damage": region_damage,
        }

    def __repr__(self) -> str:
        return (
            f"DiseaseState("
            f"braak={self.braak_stage.value}, "
            f"phase={self.damage_phase.name}, "
            f"intensity={self.intensity.value}, "
            f"epoch={self.epoch}, "
            f"total_damage={self.total_damage_accumulated:.3f}, "
            f"compensation_reserve={self.compensation.compensation_reserve:.3f}"
            f")"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def layer_to_brain_region(layer_idx: int) -> BrainRegion:
    """Map a transformer layer index to its brain region analogue.

    Uses the active architecture config (set via ``set_active_arch()``).
    """
    for region in BrainRegion:
        if layer_idx in region.layer_range:
            return region
    arch = get_active_arch()
    raise ValueError(
        f"Layer index {layer_idx} out of valid range "
        f"[0, {arch.num_layers - 1}] for {arch.name}"
    )
