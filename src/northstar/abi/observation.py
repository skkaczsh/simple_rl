from __future__ import annotations

from typing import Any

from northstar.abi.action import make_zero_action
from northstar.embodiment.manifest import EmbodimentManifest


def make_observation(
    manifest: EmbodimentManifest,
    command: dict[str, Any],
    timestamp_s: float,
    dt_s: float,
    base_linear_velocity_m_s: list[float] | None = None,
    base_angular_velocity_rad_s: list[float] | None = None,
    base_height_m: float | None = None,
    previous_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "observation.northstar.v0",
        "timestamp_s": float(timestamp_s),
        "dt_s": float(dt_s),
        "frame": "base",
        "joint_position_rad": [0.0] * manifest.active_joint_count,
        "joint_velocity_rad_s": [0.0] * manifest.active_joint_count,
        "base_linear_velocity_m_s": base_linear_velocity_m_s or [0.0, 0.0, 0.0],
        "base_angular_velocity_rad_s": base_angular_velocity_rad_s or [0.0, 0.0, 0.0],
        "projected_gravity_base": [0.0, 0.0, -1.0],
        "base_height_m": manifest.default_base_height_m if base_height_m is None else float(base_height_m),
        "foot_contact": [True] * manifest.foot_contact_site_count,
        "previous_action": previous_action or make_zero_action("act_initial", manifest, "initial"),
        "command": command,
        "mode_mask": dict(command["mode_mask"]),
        "masks": {
            "privileged": False,
            "upper_body_command_enabled": False,
            "light_axis_enabled": False,
            "semantic_hint_enabled": False,
        },
    }
