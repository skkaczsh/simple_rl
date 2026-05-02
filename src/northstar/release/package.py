"""Release package creation for model deployment."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from northstar.release.registry import ModelVersion


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def create_release_package(
    model_path: Path,
    version: ModelVersion,
    output_dir: Path,
    manifest_path: Path | None = None,
    training_config_path: Path | None = None,
) -> Path:
    """Create a release package with model and metadata.

    Args:
        model_path: Path to the model checkpoint (.pt)
        version: ModelVersion with metadata
        output_dir: Directory to create the package in
        manifest_path: Optional path to embodiment manifest
        training_config_path: Optional path to training config YAML

    Returns:
        Path to the created release package directory
    """
    package_dir = output_dir / f"release_{version.model_id}"
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy model artifact
    dest_model = package_dir / "model.pt"
    shutil.copy2(model_path, dest_model)

    # Copy manifest if provided
    if manifest_path and manifest_path.exists():
        shutil.copy2(manifest_path, package_dir / "manifest.json")

    # Copy training config if provided
    if training_config_path and training_config_path.exists():
        shutil.copy2(training_config_path, package_dir / "training_config.yaml")

    # Write eval metrics if available
    if version.eval_metrics:
        metrics_path = package_dir / "eval_metrics.json"
        metrics_path.write_text(json.dumps(version.eval_metrics, indent=2), encoding="utf-8")

    # Write capability summary if available
    if version.capability_bounds:
        cap_path = package_dir / "capability_summary.json"
        cap_path.write_text(json.dumps(version.capability_bounds, indent=2), encoding="utf-8")

    # Compute hashes
    artifact_hashes = {}
    for f in package_dir.iterdir():
        if f.is_file():
            artifact_hashes[f.name] = _sha256_file(f)

    # Create release manifest
    release_manifest = {
        "schema_version": "release_manifest.v0",
        "model_id": version.model_id,
        "version_stage": version.stage,
        "created_at": version.created_at,
        "parent_version": version.parent_version,
        "artifact_hashes": artifact_hashes,
        "compatibility": {
            "embodiment_id": "unitree_g1_43dof_sim_v0",
            "min_software_version": "0.1.0",
        },
    }

    manifest_out = package_dir / "release_manifest.json"
    manifest_out.write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")

    return package_dir
