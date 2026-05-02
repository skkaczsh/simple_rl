from __future__ import annotations

from typing import Any

from northstar.metrics.locomotion import summarize_steps


class MetricsAccumulator:
    def __init__(self, default_base_height_m: float) -> None:
        self.default_base_height_m = default_base_height_m
        self.steps: list[dict[str, Any]] = []

    def update(self, step_record: dict[str, Any]) -> None:
        self.steps.append(step_record)

    def summary(self) -> dict[str, Any]:
        return summarize_steps(self.steps, self.default_base_height_m)
