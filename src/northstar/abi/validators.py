from __future__ import annotations

from typing import Any

from northstar.embodiment.manifest import EmbodimentManifest


class ValidationError(ValueError):
    """Raised when an ABI payload violates the Phase 0/1 schema."""


def _require_keys(payload: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        if key not in payload:
            raise ValidationError(f"{path}.{key} is required")


def validate_command(command: dict[str, Any]) -> None:
    _require_keys(command, ["schema_version", "command_id", "mode_mask", "locomotion"], "command")
    if command["schema_version"] != "command.northstar.v0":
        raise ValidationError("command.schema_version must be command.northstar.v0")
    mask = command["mode_mask"]
    for disabled_key in ["upper_body", "light_axis", "semantic_intent"]:
        if bool(mask.get(disabled_key, False)):
            raise ValidationError(f"command.mode_mask.{disabled_key} must be false in Phase 1")
    locomotion = command["locomotion"]
    velocity = locomotion["target_velocity_base_m_s"]
    if len(velocity) != 3:
        raise ValidationError("command.locomotion.target_velocity_base_m_s must have length 3")
    if float(velocity[2]) != 0.0:
        raise ValidationError("command.locomotion.target_velocity_base_m_s.z must be 0.0 in Phase 1")
    if not -0.6 <= float(velocity[0]) <= 1.0:
        raise ValidationError("command.locomotion.target_velocity_base_m_s.x outside [-0.6, 1.0]")
    if not -0.4 <= float(velocity[1]) <= 0.4:
        raise ValidationError("command.locomotion.target_velocity_base_m_s.y outside [-0.4, 0.4]")
    if not -1.0 <= float(locomotion["target_yaw_rate_rad_s"]) <= 1.0:
        raise ValidationError("command.locomotion.target_yaw_rate_rad_s outside [-1.0, 1.0]")


def validate_action(action: dict[str, Any], manifest: EmbodimentManifest) -> None:
    _require_keys(
        action,
        [
            "schema_version",
            "action_id",
            "joint_position_delta_rad",
            "joint_velocity_delta_rad_s",
            "feedforward_torque_nm",
            "action_source",
            "clipped",
            "clip_summary",
        ],
        "action",
    )
    if action["schema_version"] != "action.northstar.v0":
        raise ValidationError("action.schema_version must be action.northstar.v0")
    expected = manifest.active_joint_count
    for key in ["joint_position_delta_rad", "joint_velocity_delta_rad_s", "feedforward_torque_nm"]:
        values = action[key]
        if len(values) != expected:
            raise ValidationError(f"action.{key} length must be {expected}")
