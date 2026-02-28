"""
evaluation/metrics.py — Scoring functions and error taxonomy.

AD patients make specific error types, not just more errors. This module
tracks and classifies response errors according to the clinical taxonomy.

Error taxonomy (from CLAUDE.md)
--------------------------------
semantic_paraphasia     Same semantic category substitution (fork → spoon)
circumlocution          Talking around a word; response length inflates
perseveration           Repeating previous responses inappropriately
confabulation           False memories stated with confidence
tangentiality           Topic drift across a response
neologism               Invented words with no standard meaning

Expected trajectory
-------------------
Early   : semantic paraphasias, mild circumlocution
Middle  : perseveration, intrusions, tangentiality
Late    : neologisms, confabulation, incoherence

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ErrorType(Enum):
    SEMANTIC_PARAPHASIA = "semantic_paraphasia"
    CIRCUMLOCUTION = "circumlocution"
    PERSEVERATION = "perseveration"
    CONFABULATION = "confabulation"
    TANGENTIALITY = "tangentiality"
    NEOLOGISM = "neologism"
    CORRECT = "correct"
    OTHER_ERROR = "other_error"


@dataclass
class TaskResult:
    """Result of running a single CognitiveTask."""

    task_id: str
    response: str
    error_types: List[ErrorType] = field(default_factory=list)
    score: float = 0.0                     # 0.0 (worst) to 1.0 (perfect)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class BatteryResult:
    """Aggregated results for a full TaskBattery run."""

    battery_id: str
    epoch: int
    task_results: List[TaskResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.score for r in self.task_results) / len(self.task_results)

    def error_type_counts(self) -> Dict[ErrorType, int]:
        counts: Dict[ErrorType, int] = {e: 0 for e in ErrorType}
        for result in self.task_results:
            for err in result.error_types:
                counts[err] += 1
        return counts


def classify_error(
    response: str,
    expected: Optional[str],
    task_domain: str,
) -> List[ErrorType]:
    """Classify the error types present in a model response.

    Args:
        response: Raw model output string.
        expected: Ground-truth answer, if available.
        task_domain: Which cognitive domain the task belongs to.

    Returns:
        List of ErrorType values detected in the response.
    """
    raise NotImplementedError


def score_response(response: str, expected: Optional[str], task_domain: str) -> float:
    """Return a scalar score in [0.0, 1.0] for a model response."""
    raise NotImplementedError
