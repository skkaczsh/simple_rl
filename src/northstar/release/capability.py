"""Capability summary generation for model versions."""
from __future__ import annotations

import time
from typing import Any


def generate_capability_summary(
    model_id: str,
    eval_metrics: dict[str, float],
    training_config: dict[str, Any] | None = None,
    embodiment_id: str = "unitree_g1_43dof_sim_v0",
) -> dict[str, Any]:
    """Generate a capability summary JSON for a model version.

    Args:
        model_id: Model identifier
        eval_metrics: Evaluation metrics (fall_rate, velocity_rmse, etc.)
        training_config: Training configuration used
        embodiment_id: Embodiment identifier
    """
    fall_rate = eval_metrics.get("fall_rate", 1.0)
    survival_rate = eval_metrics.get("survival_rate", 0.0)
    velocity_rmse = eval_metrics.get("velocity_rmse_m_s", 999.0)
    yaw_rmse = eval_metrics.get("yaw_rate_rmse_rad_s", 999.0)

    capabilities = []
    warnings = []

    # Standing capability
    if survival_rate > 0.95:
        capabilities.append({
            "name": "standing",
            "status": "supported",
            "confidence": min(1.0, survival_rate),
        })
    else:
        capabilities.append({
            "name": "standing",
            "status": "limited",
            "confidence": survival_rate,
        })
        warnings.append(f"Standing survival rate below threshold: {survival_rate:.2f}")

    # Forward walk capability
    if velocity_rmse < 0.3 and survival_rate > 0.8:
        capabilities.append({
            "name": "forward_walk",
            "status": "supported",
            "confidence": max(0.0, 1.0 - velocity_rmse / 0.3),
        })
    else:
        capabilities.append({
            "name": "forward_walk",
            "status": "limited",
            "confidence": max(0.0, 1.0 - velocity_rmse / 0.5),
        })

    # Velocity tracking capability
    if velocity_rmse < 0.2:
        capabilities.append({
            "name": "velocity_tracking",
            "status": "supported",
            "confidence": max(0.0, 1.0 - velocity_rmse / 0.2),
        })
    else:
        capabilities.append({
            "name": "velocity_tracking",
            "status": "limited",
            "confidence": max(0.0, 1.0 - velocity_rmse / 0.5),
        })
        warnings.append(f"Velocity RMSE above threshold: {velocity_rmse:.3f} m/s")

    # Yaw tracking capability
    if yaw_rmse < 0.3:
        capabilities.append({
            "name": "yaw_tracking",
            "status": "supported",
            "confidence": max(0.0, 1.0 - yaw_rmse / 0.3),
        })
    else:
        capabilities.append({
            "name": "yaw_tracking",
            "status": "limited",
            "confidence": max(0.0, 1.0 - yaw_rmse / 0.5),
        })
        warnings.append(f"Yaw RMSE above threshold: {yaw_rmse:.3f} rad/s")

    # Overall assessment
    if fall_rate > 0.1:
        warnings.append(f"High fall rate: {fall_rate:.2%}")

    return {
        "schema_version": "capability_summary.v0",
        "model_id": model_id,
        "embodiment_id": embodiment_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "capabilities": capabilities,
        "bounds": {
            "max_velocity_m_s": 1.0,
            "max_yaw_rate_rad_s": 1.0,
            "requires_flat_terrain": True,
        },
        "warnings": warnings,
        "metrics": {
            "fall_rate": fall_rate,
            "survival_rate": survival_rate,
            "velocity_rmse_m_s": velocity_rmse,
            "yaw_rate_rmse_rad_s": yaw_rmse,
        },
    }
