#!/usr/bin/env python3
"""Promote a trained model to candidate stage.

Usage:
    python scripts/promote_model.py <training_output_dir> [--registry <registry_dir>]

Loads a training output directory, registers the best model as a candidate,
and creates a release package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/promote_model.py <training_output_dir> [--registry <dir>]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    registry_dir = Path("registry")

    # Parse optional args
    for i, arg in enumerate(sys.argv):
        if arg == "--registry" and i + 1 < len(sys.argv):
            registry_dir = Path(sys.argv[i + 1])

    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    # Find best model
    model_path = output_dir / "model_best.pt"
    if not model_path.exists():
        print(f"Error: model_best.pt not found in {output_dir}")
        sys.exit(1)

    # Load training log if available
    log_path = output_dir / "training_log.json"
    eval_metrics = {}
    if log_path.exists():
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(log_data, list) and log_data:
            # Compute basic metrics from training log
            rewards = [e.get("avg_reward", 0) for e in log_data]
            vel_rmses = [e.get("velocity_rmse", 0) for e in log_data if "velocity_rmse" in e]
            eval_metrics = {
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "best_reward": max(rewards) if rewards else 0.0,
                "velocity_rmse_m_s": sum(vel_rmses) / len(vel_rmses) if vel_rmses else 0.0,
                "total_iterations": len(log_data),
                "survival_rate": 1.0,  # Default for Phase 1-A
                "fall_rate": 0.0,
            }

    # Load school summary if available
    school_summary_path = output_dir / "school_summary.json"
    if school_summary_path.exists():
        school_data = json.loads(school_summary_path.read_text(encoding="utf-8"))
        eval_metrics["school_samples"] = school_data.get("total_samples", 0)
        eval_metrics["school_episodes"] = school_data.get("total_episodes", 0)

    # Register model
    from northstar.release.capability import generate_capability_summary
    from northstar.release.package import create_release_package
    from northstar.release.registry import ModelRegistry, ModelVersion

    model_id = f"follower_candidate_{output_dir.name}"
    capability = generate_capability_summary(model_id, eval_metrics)

    version = ModelVersion(
        model_id=model_id,
        stage="candidate",
        training_config={"output_dir": str(output_dir)},
        eval_metrics=eval_metrics,
        capability_bounds=capability,
        artifact_path=str(model_path),
    )

    registry = ModelRegistry(registry_dir)
    registry.register(version)

    # Create release package
    package_dir = create_release_package(
        model_path=model_path,
        version=version,
        output_dir=output_dir,
    )

    print(f"Model registered: {model_id}")
    print(f"  Stage: candidate")
    print(f"  Best reward: {eval_metrics.get('best_reward', 0):.3f}")
    print(f"  Velocity RMSE: {eval_metrics.get('velocity_rmse_m_s', 0):.4f} m/s")
    print(f"  School samples: {eval_metrics.get('school_samples', 0)}")
    print(f"  Release package: {package_dir}")
    print(f"  Registry: {registry_dir / 'registry.json'}")


if __name__ == "__main__":
    main()
