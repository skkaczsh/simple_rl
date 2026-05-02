"""Curriculum training runner that loads config, creates curriculum, and runs staged training."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from northstar.embodiment.manifest import EmbodimentManifest, load_manifest
from northstar.env.physics_mock_env import DomainRandomizationConfig, PhysicsConfig, PhysicsMockEnv
from northstar.rewards.locomotion import RewardConfig
from northstar.training.curriculum import CurriculumManager, CurriculumStage


def load_curriculum_config(path: Path) -> dict[str, Any]:
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
    manifest: EmbodimentManifest,
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


def run_curriculum_training(config_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run curriculum training from config file."""
    from northstar.training.ppo import PPOConfig, train_curriculum

    cfg_raw = load_curriculum_config(config_path)
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

    # We'll use the first stage's command ranges for the initial env creation
    # The make_envs_fn will be called with the current stage's ranges
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

    return train_curriculum(make_envs_fn, manifest, ppo_cfg, output_dir, curriculum)
