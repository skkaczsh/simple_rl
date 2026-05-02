from __future__ import annotations

import math
from typing import Any


def rmse(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError("actual and expected must have same length")
    if not actual:
        return 0.0
    return math.sqrt(sum((a - e) ** 2 for a, e in zip(actual, expected)) / len(actual))


def summarize_steps(steps: list[dict[str, Any]], default_base_height_m: float) -> dict[str, Any]:
    if not steps:
        return {
            "step_count": 0,
            "near_fall_count": 0,
            "fall_count": 0,
            "action_clipping_count": 0,
            "base_height_rmse_m": 0.0,
            "velocity_rmse_m_s": 0.0,
            "yaw_rate_rmse_rad_s": 0.0,
        }
    base_heights = [float(step["observation"]["base_height_m"]) for step in steps]
    velocity_errors = []
    yaw_errors = []
    near_fall_count = 0
    fall_count = 0
    action_clipping_count = 0
    for step in steps:
        observation = step["observation"]
        command = step["command"]["locomotion"]
        velocity_errors.append(
            rmse(
                list(observation["base_linear_velocity_m_s"]),
                list(command["target_velocity_base_m_s"]),
            )
        )
        yaw_errors.append(
            abs(float(observation["base_angular_velocity_rad_s"][2]) - float(command["target_yaw_rate_rad_s"]))
        )
        triggered = set(step.get("dangerous_signal", {}).get("triggered", []))
        if "near_fall" in triggered:
            near_fall_count += 1
        if "fall" in triggered:
            fall_count += 1
        if bool(step["action"].get("clipped", False)):
            action_clipping_count += 1
    return {
        "step_count": len(steps),
        "near_fall_count": near_fall_count,
        "fall_count": fall_count,
        "action_clipping_count": action_clipping_count,
        "base_height_rmse_m": rmse(base_heights, [default_base_height_m] * len(base_heights)),
        "velocity_rmse_m_s": sum(velocity_errors) / len(velocity_errors),
        "yaw_rate_rmse_rad_s": sum(yaw_errors) / len(yaw_errors),
    }
