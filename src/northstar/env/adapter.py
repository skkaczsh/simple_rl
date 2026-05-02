from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class StepResult:
    observation: dict[str, Any]
    confidence: dict[str, Any]
    dangerous_signal: dict[str, Any]
    reward_debug: dict[str, float]
    events: list[dict[str, Any]]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class EnvAdapter(Protocol):
    def reset(self, seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, action: dict[str, Any], command: dict[str, Any]) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
