from __future__ import annotations

from copy import deepcopy
from typing import Any

from northstar.embodiment.manifest import EmbodimentManifest


def _clip_value(value: float, limit: float) -> tuple[float, bool]:
    clipped = max(-limit, min(limit, float(value)))
    return clipped, clipped != float(value)


def _clip_list(values: list[float], limit: float) -> tuple[list[float], list[int]]:
    clipped_values: list[float] = []
    clipped_indices: list[int] = []
    for index, value in enumerate(values):
        clipped, changed = _clip_value(value, limit)
        clipped_values.append(clipped)
        if changed:
            clipped_indices.append(index)
    return clipped_values, clipped_indices


def clip_action(
    action: dict[str, Any],
    manifest: EmbodimentManifest,
    episode_id: str,
    step_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = deepcopy(action)
    pos, pos_indices = _clip_list(result["joint_position_delta_rad"], manifest.action_limit_rad)
    vel, vel_indices = _clip_list(result["joint_velocity_delta_rad_s"], manifest.velocity_limit_rad_s)
    torque, torque_indices = _clip_list(result["feedforward_torque_nm"], manifest.torque_limit_nm)
    result["joint_position_delta_rad"] = pos
    result["joint_velocity_delta_rad_s"] = vel
    result["feedforward_torque_nm"] = torque
    clip_summary = []
    if pos_indices:
        clip_summary.append({"field": "joint_position_delta_rad", "indices": pos_indices})
    if vel_indices:
        clip_summary.append({"field": "joint_velocity_delta_rad_s", "indices": vel_indices})
    if torque_indices:
        clip_summary.append({"field": "feedforward_torque_nm", "indices": torque_indices})
    result["clipped"] = bool(clip_summary)
    result["clip_summary"] = clip_summary
    events = []
    if clip_summary:
        events.append(
            {
                "schema_version": "event_record.v0",
                "episode_id": episode_id,
                "step_index": int(step_index),
                "event_type": "action_clip",
                "severity": "warning",
                "source": "action_adapter",
                "payload": {"clip_summary": clip_summary},
            }
        )
    return result, events
