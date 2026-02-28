"""
evaluation/longitudinal.py — Cross-epoch tracking and decline curve analysis.

Stores BatteryResult snapshots across epochs and provides utilities for
plotting decline trajectories and comparing against published clinical data.

Expected clinical trajectory
-----------------------------
MMSE decline     : ~3 points/year in mild-moderate AD
Fluency decline  : category > letter fluency (disproportionate semantic loss)
Error trajectory : paraphasia → perseveration → confabulation → incoherence

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from evaluation.metrics import BatteryResult


@dataclass
class LongitudinalRecord:
    """Container for all evaluation snapshots across a simulation run."""

    experiment_id: str
    battery_results: List[BatteryResult] = field(default_factory=list)

    def add_result(self, result: BatteryResult) -> None:
        self.battery_results.append(result)

    @property
    def epochs(self) -> List[int]:
        return [r.epoch for r in self.battery_results]

    @property
    def mean_scores_by_epoch(self) -> Dict[int, float]:
        return {r.epoch: r.mean_score for r in self.battery_results}


def compute_decline_rate(record: LongitudinalRecord) -> float:
    """Estimate average score decline per epoch across the simulation.

    Returns:
        Mean drop in score per epoch (positive = declining).
    """
    raise NotImplementedError


def compare_to_clinical_profile(
    record: LongitudinalRecord,
    clinical_mmse_per_year: float = 3.0,
) -> Dict[str, float]:
    """Compute similarity between simulated and clinical decline curves.

    Args:
        record: Longitudinal results from the simulation.
        clinical_mmse_per_year: Expected MMSE decline rate for calibration.

    Returns:
        Dict with correlation, RMSE, and calibration factor.
    """
    raise NotImplementedError
