from __future__ import annotations

from typing import Any

from northstar.embodiment.manifest import EmbodimentManifest


def make_zero_action(
    action_id: str,
    manifest: EmbodimentManifest,
    action_source: str,
) -> dict[str, Any]:
    count = manifest.active_joint_count
    return {
        "schema_version": "action.northstar.v0",
        "action_id": action_id,
        "joint_position_delta_rad": [0.0] * count,
        "joint_velocity_delta_rad_s": [0.0] * count,
        "feedforward_torque_nm": [0.0] * count,
        "action_source": action_source,
        "clipped": False,
        "clip_summary": [],
    }
