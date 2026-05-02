from __future__ import annotations

import random
from typing import Any

from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.embodiment.manifest import EmbodimentManifest
from northstar.env.adapter import StepResult
from northstar.env.state import MockPhase1State


class MockPhase1Env:
    env_id = "mock_phase1_env_v0"

    def __init__(self, manifest: EmbodimentManifest, dt_s: float, horizon_steps: int) -> None:
        self.manifest = manifest
        self.dt_s = float(dt_s)
        self.horizon_steps = int(horizon_steps)
        self.rng = random.Random(0)
        self.state = MockPhase1State(base_height_m=manifest.default_base_height_m)
        self.last_observation: dict[str, Any] | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        self.rng = random.Random(seed)
        self.state = MockPhase1State(base_height_m=self.manifest.default_base_height_m)
        command = {
            "schema_version": "command.northstar.v0",
            "command_id": "reset_cmd",
            "mode_mask": {
                "stand": True,
                "locomotion": True,
                "upper_body": False,
                "light_axis": False,
                "semantic_intent": False,
            },
            "locomotion": {
                "target_base_height_m": 0.0,
                "target_velocity_base_m_s": [0.0, 0.0, 0.0],
                "target_yaw_rate_rad_s": 0.0,
                "target_heading_rad": None,
                "stop_request": False,
                "brace_request": False,
            },
            "upper_body": None,
            "light_axis_hint": None,
            "semantic_hint": None,
        }
        self.last_observation = make_observation(self.manifest, command, 0.0, self.dt_s)
        return self.last_observation

    def step(self, action: dict[str, Any], command: dict[str, Any]) -> StepResult:
        events: list[dict[str, Any]] = []
        locomotion = command["locomotion"]
        target_velocity = list(locomotion["target_velocity_base_m_s"])
        if locomotion.get("stop_request", False):
            target_velocity = [0.0, 0.0, 0.0]
            events.append(self._event("stop_request", "info", {}))
        if locomotion.get("brace_request", False):
            events.append(self._event("brace_request", "info", {}))
        self.state.base_linear_velocity_m_s = [float(value) for value in target_velocity]
        self.state.base_angular_velocity_rad_s = [0.0, 0.0, float(locomotion["target_yaw_rate_rad_s"])]
        self.state.step_index += 1
        self.state.time_s = self.state.step_index * self.dt_s
        observation = make_observation(
            self.manifest,
            command,
            timestamp_s=self.state.time_s,
            dt_s=self.dt_s,
            base_linear_velocity_m_s=self.state.base_linear_velocity_m_s,
            base_angular_velocity_rad_s=self.state.base_angular_velocity_rad_s,
            base_height_m=self.state.base_height_m,
            previous_action=action,
        )
        triggered = []
        near_fall_risk = 0.0
        if self.state.base_height_m < self.manifest.default_base_height_m - 0.2:
            triggered.append("near_fall")
            near_fall_risk = 0.8
            events.append(self._event("near_fall", "warning", {"base_height_m": self.state.base_height_m}))
        dangerous = make_dangerous_signal(
            overall_risk=near_fall_risk,
            triggered=triggered,
            near_fall_risk=near_fall_risk,
        )
        terminated = self.state.step_index >= self.horizon_steps
        self.last_observation = observation
        return StepResult(
            observation=observation,
            confidence=make_confidence(1.0 - near_fall_risk, 1.0 - near_fall_risk, 1.0, 1.0),
            dangerous_signal=dangerous,
            reward_debug={"mock_reward": 1.0 - near_fall_risk},
            events=events,
            terminated=terminated,
            truncated=False,
            info={"env_id": self.env_id},
        )

    def inject_event(self, event_type: str) -> list[dict[str, Any]]:
        if event_type == "near_fall":
            self.state.base_height_m = self.manifest.default_base_height_m - 0.25
        return [self._event(event_type, "warning", {"injected": True})]

    def close(self) -> None:
        return None

    def _event(self, event_type: str, severity: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "event_record.v0",
            "episode_id": "",
            "step_index": self.state.step_index,
            "event_type": event_type,
            "severity": severity,
            "source": self.env_id,
            "payload": payload,
        }
