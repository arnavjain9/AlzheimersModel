# Alzheimer's Disease LLM Simulation — CLAUDE.md

## Project Overview
This project simulates Alzheimer's disease progression in Qwen-32B by applying 
real structural damage to model weights and architecture — not prompt-based 
behavioral mimicry. The goal is cause-based degradation: damage the structure, 
observe emergent behavioral deficits that mirror clinical AD symptomatology.

## Core Design Philosophy
- **Cause-based, not symptom-based**: We structurally damage the network and 
  observe emergent behavior. Never prompt the model to "act confused."
- **Biological realism over simplicity**: Damage patterns follow Braak staging, 
  functional connectivity, and two-phase amyloid/tau progression.
- **Validate against clinical data**: Simulated decline curves should match 
  published AD patient profiles.

## Critical Mapping Corrections (read carefully)
These are non-obvious and easy to get wrong:

### KV Cache = Working Memory (NOT episodic memory)
The KV cache exists only within a single context window. It does not persist 
across sessions. It maps to working memory, not hippocampal episodic memory.
Hippocampal function maps to cross-attention integration mechanisms.

### DMN Mapping (Corrected)
- DMN (Default Mode Network) = automatic/non-effortful response generation = thinking mode OFF
- TPN (Task-Positive Network) = effortful reasoning = thinking mode ON
- AD symptom = FAILURE TO ENGAGE thinking mode when needed, not corruption of thinking mode itself
- Never simulate DMN dysfunction by corrupting the thinking/reasoning pathway. 
  Simulate it as inability to switch INTO effortful processing.

### Memory System Mapping
| Memory Type | LLM Analogue | Damage Mechanism |
|---|---|---|
| Sensory buffer | Input embedding layer | Embedding noise injection |
| Working memory | KV cache within context window | KV cache corruption |
| Short-term consolidation | Cross-attention integration | Attention weight disruption |
| Episodic memory (recent) | Early context tokens | Positional encoding disruption |
| Semantic memory | FFN weights, embedding space | FFN neuron death, weight pruning |
| Procedural memory | Learned attention patterns | Attention head ablation |

## Qwen-32B Architecture Parameters
- Total layers: 36
- Hidden dimension: 3584
- FFN intermediate size: 18944
- Attention: 32 Q heads / 8 KV heads (GQA, 4:1 ratio)
- Context window: 32K native, 131K with YaRN
- Vocabulary: 151,646 tokens

### GQA Critical Constraint
Qwen uses Grouped Query Attention: 32 Q heads share 8 KV heads (4 Q per KV group).
Damaging 1 KV head affects 4 Q heads simultaneously. Always account for this 
multiplier when implementing attention damage. Never treat Q and KV heads as 
independent.

### Layer-to-Brain-Region Mapping (36 layers)
| Brain Region | Layers | Proportion | AD Vulnerability |
|---|---|---|---|
| Entorhinal cortex analogue | 0-5 | 0-14% | Very early (Braak I-II) |
| Hippocampal analogue | 6-12 | 17-33% | Early (Braak III-IV) |
| Temporal association analogue | 13-24 | 36-67% | Early-middle |
| Prefrontal analogue | 25-35 | 69-100% | Middle-late |

### Key Component Access Points
| Component | Attribute Path |
|---|---|
| Input embeddings | `model.embed_tokens` |
| Per-layer attention | `model.layers[i].self_attn` |
| Q projection | `self_attn.q_proj` |
| K projection | `self_attn.k_proj` |
| V projection | `self_attn.v_proj` |
| O projection | `self_attn.o_proj` |
| FFN gate | `model.layers[i].mlp.gate_proj` |
| FFN up | `model.layers[i].mlp.up_proj` |
| FFN down | `model.layers[i].mlp.down_proj` |
| Layer norm | `input_layernorm`, `post_attention_layernorm` |
| Output projection | `lm_head` |

## Two-Phase Damage Model
Always implement damage in two phases:
1. **Phase 1 — Amyloid dominant**: Global low-level noise, mild synaptic 
   dysfunction, relatively preserved structure
2. **Phase 2 — Tau dominant**: Focal neuron death spreading along functional 
   connectivity, structural degradation

## Braak Staging Implementation
| Stage | Layers | Mechanisms | Severity Cap |
|---|---|---|---|
| I-II (Transentorhinal) | 0-5 | Noise injection, mild synaptic dysfunction | 0.3 |
| III-IV (Limbic) | 0-18 | Synaptic dysfunction, attention disruption, KV corruption | 0.5 |
| V (Neocortical early) | 0-29 | All above + clustered neuron death, FFN pruning | 0.7 |
| VI (Neocortical late) | 0-35 | All above + widespread structural damage | 1.0 |

## Project Structure
```
alzheimer_sim/
  CLAUDE.md                        # This file
  requirements.txt                 # Dependencies
  disease_state.py                 # DiseaseState, all config dataclasses
  qwen_hooks.py                    # Hook infrastructure, damage context manager
  damage/
    __init__.py
    mechanisms.py                  # Top-level damage orchestration
    noise.py                       # Noise injection (synaptic dysfunction)
    pruning.py                     # Weight pruning (synaptic elimination)
    neuron_death.py                # FFN neuron killing with functional clustering
    connectivity.py                # Residual stream + positional encoding disruption
  evaluation/
    __init__.py
    tasks.py                       # Cognitive task definitions
    metrics.py                     # Scoring, error taxonomy
    clinical_instruments.py        # MMSE adaptation, verbal fluency, etc.
    longitudinal.py                # Cross-epoch tracking
  experiments/
    run_experiment.py              # Main experiment runner script
    configs/                       # YAML experiment configurations
  notebooks/
    01_phase1_foundation.ipynb     # Foundation validation
    02_phase2_biological.ipynb     # Biological realism experiments
    03_phase3_temporal.ipynb       # Temporal dynamics experiments
    04_phase4_evaluation.ipynb     # Full evaluation battery
    05_phase5_validation.ipynb     # Clinical comparison and ablations
```

## Implementation Constraints
- **Always use a context manager pattern** (`with damage_context(model, config)`)
  rather than in-place weight mutation. This is non-negotiable — it makes
  evaluation and ablation studies possible without reloading the model.
- **Functional clustering for neuron death**: Neurons must be killed within
  functional clusters (grouped by activation correlation), not randomly.
  Random killing is a control condition, not the primary implementation.
- **Spreading dynamics**: Damage spreads from seed regions (entorhinal-analogue
  layers first) along functional connectivity, not randomly or uniformly.
- **Compensation modeling**: Surviving connections upregulate when neighbors
  are pruned. Track a compensation reserve that depletes as damage accumulates.

## Runtime Efficiency Constraints (non-negotiable)
These apply to every function in `damage/`, `qwen_hooks.py`, and any code
that runs inside a forward pass or damage context.

### Sparse deltas — never copy full weight tensors
Store only the indices and original values of modified weights, not the full
tensor. For Qwen-32B, a single weight matrix is hundreds of MB. The canonical
snapshot format is defined in `disease_state.py`:
```
SparseDeltaT = {"param_name": str, "flat_indices": LongTensor, "original_values": Tensor}
WeightSnapshot = List[SparseDeltaT]
```
Restore via: `param.data.view(-1).scatter_(0, flat_indices, original_values)`

### torch.no_grad() — always, everywhere
`damage_context` wraps everything in `torch.no_grad()`. No damage or evaluation
code should ever allow gradient computation. Grads double memory for no benefit.

### Dtype preservation — never upcast to FP32
All damage operations (noise generation, mask application, scatter ops) must
run in the parameter's native dtype (BF16 or FP16 for Qwen-32B). Always cast
before operating: `noise = noise.to(param.dtype)`. FP32 upcasting during damage
doubles the memory footprint of every touched tensor.

### Precomputed masks — compute once per epoch, read many times
Pruning masks (`LayerDamageState.pruning_mask_indices`), dead neuron indices
(`dead_neuron_indices`), and ablated KV heads (`ablated_kv_heads`) are computed
once at the start of each epoch by `damage/pruning.py` and `damage/neuron_death.py`,
then cached in `LayerDamageState`. Forward hooks and damage functions READ these
caches — they never recompute. `DiseaseState.record_damage_increment()` calls
`invalidate_masks()` on all layers so the next epoch recomputes fresh masks.

### Forward hooks registered once per context, not per inference call
`qwen_hooks._register_hooks()` is called once at `damage_context.__enter__`.
`handle.remove()` is called once at `__exit__` (in the finally block).
Hook registration itself allocates; doing it inside a generation loop is wrong.

## Damage Intensity Scale
| Level | Noise | Pruning | Neuron Kill |
|---|---|---|---|
| Subclinical | 0.01-0.05 | 0-5% | 0-2% |
| Mild | 0.05-0.15 | 5-15% | 2-10% |
| Moderate | 0.15-0.30 | 15-30% | 10-25% |
| Severe | 0.30+ | 30%+ | 25%+ |

## Evaluation Philosophy
AD patients don't just make more errors — they make specific error types. 
Track error taxonomy, not just accuracy:
- **Semantic paraphasia** (fork → spoon): same semantic category substitution
- **Circumlocution**: talking around a word, response length inflates
- **Perseveration**: repeating previous responses inappropriately
- **Confabulation**: false memories stated with confidence
- **Tangentiality**: topic drift across response

Expected error trajectory:
- Early: semantic paraphasias, mild circumlocution
- Middle: perseveration, intrusions, tangentiality
- Late: neologisms, confabulation, incoherence

## Thinking Mode (Extended Thinking) Handling
Qwen's extended thinking generates explicit reasoning in tags.

Staged simulation strategy:
1. **Early**: Subtle engagement failures — model skips thinking mode when it should use it
2. **Middle**: More frequent failures, clear depth reduction, occasional inappropriate intrusion
3. **Late**: Severe depth reduction, thinking content degradation, or complete failure to engage

Never simulate thinking mode dysfunction by corrupting the reasoning content directly.
Simulate it as the salience network failing to trigger effortful processing.

## Compensation Modeling
- When connections are pruned, surviving connections in the same layer upregulate
- `surviving_weights *= (1 + compensation_factor * prune_rate)`
- Compensation factor decreases as total damage increases
- Track a `compensation_reserve` (starts at 1.0, depletes with each damage increment)
- When reserve is exhausted, damage has full unmitigated impact
- Models the clinical "tipping point" of rapid decline

## Validation Strategy
Run these control conditions to validate biological specificity:
| Condition | Purpose |
|---|---|
| Undamaged model | Ceiling baseline |
| Random noise (non-patterned) | Distinguish AD-specific from generic damage |
| Uniform damage (all layers equal) | Test importance of regional specificity |
| Reversed staging (late regions first) | Test importance of correct staging order |

If AD-patterned damage produces more realistic symptom profiles than these controls,
the biological mapping is validated.

## Colab Usage Notes
- Clone the repo at the start of each Colab session: `!git clone <repo_url>`
- Add repo root to path: `import sys; sys.add_path('/content/alzheimer_sim')`
- Each notebook corresponds to one project phase
- Notebooks import from Python modules — do not put logic directly in notebook cells
- Commit and push changes back to GitHub at the end of each session