#!/usr/bin/env python3
"""Run school curriculum training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from northstar.training.school_curriculum_runner import run_school_curriculum_training

if __name__ == "__main__":
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/train/phase1_school_curriculum.yaml")
    manifest_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("manifests/unitree_g1.yaml")
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runs/phase1_school_curriculum")

    print(f"Config: {config_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_dir}")

    result = run_school_curriculum_training(
        config_path=config_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        collect_ratio=0.2,
        max_pool_samples=50000,
    )

    print(f"\nSchool curriculum training complete: {result}")
