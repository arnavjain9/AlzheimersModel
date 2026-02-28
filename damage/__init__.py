"""
damage/ — Weight-level damage mechanisms for the Alzheimer's simulation.

Submodules
----------
mechanisms   Top-level orchestration: applies all damage types for a given DiseaseState.
noise        Gaussian noise injection (synaptic dysfunction, Phase 1).
pruning      Weight magnitude pruning (synaptic elimination).
neuron_death FFN neuron killing with functional clustering and spreading dynamics.
connectivity Residual stream and positional encoding disruption.
"""
