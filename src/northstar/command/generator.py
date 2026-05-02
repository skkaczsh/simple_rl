from __future__ import annotations

from copy import deepcopy
from typing import Any

from northstar.abi.command import make_locomotion_command


class CommandGenerator:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario

    def command_at_step(self, step_index: int) -> dict[str, Any]:
        base = deepcopy(self.scenario["command"])
        for item in self.scenario.get("command_schedule", []):
            if int(item["step"]) <= step_index:
                for key, value in item.items():
                    if key != "step":
                        base[key] = value
        return make_locomotion_command(
            command_id=f"{self.scenario['scenario_id']}_cmd_{step_index}",
            target_velocity_base_m_s=base.get("target_velocity_base_m_s", [0.0, 0.0, 0.0]),
            target_yaw_rate_rad_s=base.get("target_yaw_rate_rad_s", 0.0),
            stop_request=base.get("stop_request", False),
            brace_request=base.get("brace_request", False),
        )
