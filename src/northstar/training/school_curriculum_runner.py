"""School-integrated curriculum training runner.

Runs curriculum training with school sample collection enabled.
Collects interesting episodes (falls, near-falls, tracking errors)
into an experience pool for future training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from northstar.embodiment.manifest import load_manifest
from northstar.env.physics_mock_env import DomainRandomizationConfig, PhysicsConfig, PhysicsMockEnv
from northstar.rewards.locomotion import RewardConfig
from northstar.school.experience_pool import SchoolExperiencePool
from northstar.training.curriculum import CurriculumManager, CurriculumStage
from northstar.training.school_loop import train_with_school


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_curriculum_stages(cfg_raw: dict[str, Any]) -> list[CurriculumStage]:
    """Create CurriculumStage objects from config."""
    stages = []
    for stage_cfg in cfg_raw.get("curriculum", {}).get("stages", []):
        stages.append(CurriculumStage(
            name=stage_cfg["name"],
            description=stage_cfg.get("description", ""),
            vx_range=tuple(stage_cfg.get("vx_range", [0.0, 0.0])),
            vy_range=tuple(stage_cfg.get("vy_range", [0.0, 0.0])),
            yaw_rate_range=tuple(stage_cfg.get("yaw_rate_range", [0.0, 0.0])),
            height_range=tuple(stage_cfg.get("height_range", [0.71, 0.77])),
            num_iterations=stage_cfg.get("num_iterations", 200),
            learning_rate=stage_cfg.get("learning_rate", 5e-5),
            min_iterations=stage_cfg.get("min_iterations", 100),
            min_reward=stage_cfg.get("min_reward", 3.5),
            min_survival_rate=stage_cfg.get("min_survival_rate", 0.8),
        ))
    return stages


def make_envs(
    num_envs: int,
    manifest: Any,
    physics: PhysicsConfig,
    reward: RewardConfig,
    horizon_steps: int,
    vx_range: tuple[float, float],
    vy_range: tuple[float, float],
    yaw_rate_range: tuple[float, float],
    height_range: tuple[float, float],
) -> list[PhysicsMockEnv]:
    return [
        PhysicsMockEnv(
            manifest=manifest,
            physics_config=physics,
            reward_config=reward,
            horizon_steps=horizon_steps,
            vx_range=vx_range,
            vy_range=vy_range,
            yaw_rate_range=yaw_rate_range,
            height_range=height_range,
        )
        for _ in range(num_envs)
    ]


def run_school_curriculum_training(
    config_path: Path,
    manifest_path: Path,
    output_dir: Path,
    collect_ratio: float = 0.2,
    max_pool_samples: int = 50000,
) -> dict[str, Any]:
    """Run curriculum training with school sample collection.

    Args:
        config_path: Path to training config YAML
        manifest_path: Path to embodiment manifest JSON
        output_dir: Output directory
        collect_ratio: Fraction of episodes to collect (0.2 = 20%)
        max_pool_samples: Maximum samples in the pool
    """
    from northstar.training.ppo import PPOConfig

    cfg_raw = load_config(config_path)
    manifest = load_manifest(manifest_path)

    env_cfg = cfg_raw.get("env", {})
    physics_cfg = cfg_raw.get("physics", {})
    training_cfg = cfg_raw.get("training", {})

    # Domain randomization
    dr_cfg = physics_cfg.pop("domain_randomization", {})
    physics = PhysicsConfig.from_dict(physics_cfg)
    if dr_cfg:
        physics.domain_randomization = DomainRandomizationConfig.from_dict(dr_cfg)

    reward = RewardConfig.from_yaml(Path(cfg_raw["reward"]["config_path"]))
    ppo_cfg = PPOConfig.from_dict(training_cfg)
    horizon = env_cfg.get("horizon_steps", 500)

    # Create curriculum
    stages = make_curriculum_stages(cfg_raw)
    curriculum = CurriculumManager(stages)

    # Create school experience pool
    school_pool = SchoolExperiencePool(
        output_dir=output_dir,
        max_samples=max_pool_samples,
    )

    def make_envs_fn(n):
        stage = curriculum.current_stage
        return make_envs(
            num_envs=n,
            manifest=manifest,
            physics=physics,
            reward=reward,
            horizon_steps=horizon,
            vx_range=stage.vx_range,
            vy_range=stage.vy_range,
            yaw_rate_range=stage.yaw_rate_range,
            height_range=stage.height_range,
        )

    # Run training with school collection
    result = train_with_school(
        make_envs_fn=make_envs_fn,
        manifest=manifest,
        cfg=ppo_cfg,
        output_dir=output_dir,
        school_pool=school_pool,
        collect_ratio=collect_ratio,
        save_pool_interval=100,
    )

    # Add curriculum info to result
    result["completed_stages"] = curriculum.state.completed_stages
    result["total_iterations"] = curriculum.state.total_iterations
    result["school_summary"] = school_pool.get_summary()

    return result
