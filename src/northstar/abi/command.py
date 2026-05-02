from __future__ import annotations

from typing import Any


def make_locomotion_command(
    command_id: str,
    target_velocity_base_m_s: list[float],
    target_yaw_rate_rad_s: float,
    target_base_height_m: float = 0.0,
    stop_request: bool = False,
    brace_request: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "command.northstar.v0",
        "command_id": command_id,
        "mode_mask": {
            "stand": True,
            "locomotion": True,
            "upper_body": False,
            "light_axis": False,
            "semantic_intent": False,
        },
        "locomotion": {
            "target_base_height_m": float(target_base_height_m),
            "target_velocity_base_m_s": [float(v) for v in target_velocity_base_m_s],
            "target_yaw_rate_rad_s": float(target_yaw_rate_rad_s),
            "target_heading_rad": None,
            "stop_request": bool(stop_request),
            "brace_request": bool(brace_request),
        },
        "upper_body": None,
        "light_axis_hint": None,
        "semantic_hint": None,
    }
