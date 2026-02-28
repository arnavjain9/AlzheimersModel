"""
evaluation/clinical_instruments.py — Standardised clinical assessment adaptations.

Adapts validated clinical instruments to the LLM evaluation context.
Scores are normalised to [0, 1] for cross-instrument comparability.

Instruments implemented (stubs)
--------------------------------
MMSE (Mini-Mental State Examination)
    Orientation, registration, attention, recall, language.
    Standard score: 0–30. Normalised: score / 30.

Verbal Fluency Battery
    Category fluency (animals / supermarket in 60 s).
    Letter fluency (FAS test — words beginning with F, A, S).

Boston Naming Test (abbreviated)
    60-item confrontation naming task; adapted to text description prompts.

Clock Drawing Test (analogue)
    Adapted: model generates a textual / structured description of a clock face.

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MMSEResult:
    """Result from a simulated MMSE administration."""

    raw_score: float           # 0–30
    subscores: Dict[str, float] = field(default_factory=dict)

    @property
    def normalised_score(self) -> float:
        return self.raw_score / 30.0

    @property
    def severity_label(self) -> str:
        if self.raw_score >= 24:
            return "normal"
        elif self.raw_score >= 18:
            return "mild"
        elif self.raw_score >= 10:
            return "moderate"
        return "severe"


@dataclass
class VerbalFluencyResult:
    """Result from a verbal fluency trial."""

    condition: str             # e.g. "animals", "F", "A", "S"
    words_generated: List[str] = field(default_factory=list)
    intrusions: List[str] = field(default_factory=list)
    perseverations: List[str] = field(default_factory=list)

    @property
    def valid_word_count(self) -> int:
        return len(self.words_generated) - len(self.intrusions) - len(self.perseverations)


def administer_mmse(model: object, state: object) -> MMSEResult:
    """Run a full simulated MMSE on the model under the given DiseaseState."""
    raise NotImplementedError


def administer_verbal_fluency(
    model: object,
    state: object,
    conditions: Optional[List[str]] = None,
) -> List[VerbalFluencyResult]:
    """Run verbal fluency trials for each condition."""
    raise NotImplementedError
