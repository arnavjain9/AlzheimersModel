"""
evaluation/tasks.py — Cognitive task definitions.

Each task probes one or more cognitive domains that are differentially
impaired across Braak stages. Tasks are designed so that the expected
error types match the clinical AD error taxonomy (see metrics.py).

Cognitive domains probed
------------------------
Naming            Object / face naming — semantic memory
Verbal recall     Delayed word list recall — episodic consolidation
Verbal fluency    Category / letter fluency — semantic search
Orientation       Time, place, person orientation — episodic + working memory
Reasoning         Multi-step logical reasoning — executive / TPN engagement
Repetition        Sentence repetition — phonological loop (working memory)

Implementation is deferred — this module is a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CognitiveTask:
    """Abstract base for a single cognitive task."""

    task_id: str
    domain: str
    prompt_template: str
    expected_answer: Optional[str] = None
    scoring_rubric: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskBattery:
    """A collection of tasks forming one evaluation pass."""

    battery_id: str
    tasks: List[CognitiveTask] = field(default_factory=list)

    def add_task(self, task: CognitiveTask) -> None:
        self.tasks.append(task)


def build_standard_battery() -> TaskBattery:
    """Construct the standard evaluation battery used across all experiments.

    Returns a TaskBattery with tasks spanning all major cognitive domains.
    """
    raise NotImplementedError
