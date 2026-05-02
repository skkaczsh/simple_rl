"""Training runner that loads config, creates envs, and runs PPO."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from northstar.embodiment.manifest import EmbodimentManifest, load_manifest
from northstar.env.physics_mock_env import PhysicsConfig, PhysicsMockEnv
from northstar.rewards.locomotion import RewardConfig


def load_training_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def run_training(config_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    from northstar.training.ppo import PPOConfig, train

    cfg_raw = load_training_config(config_path)
    manifest = load_manifest(manifest_path)

    env_cfg = cfg_raw.get("env", {})
    cmd_cfg = cfg_raw.get("command", {})
    physics_cfg = cfg_raw.get("physics", {})
    training_cfg = cfg_raw.get("training", {})

    physics = PhysicsConfig.from_dict(physics_cfg)
    reward = RewardConfig.from_yaml(Path(cfg_raw["reward"]["config_path"]))
    ppo_cfg = PPOConfig.from_dict(training_cfg)
    horizon = env_cfg.get("horizon_steps", 500)

    def make_envs_fn(n):
        return make_envs(
            num_envs=n,
            manifest=manifest,
            physics=physics,
            reward=reward,
            horizon_steps=horizon,
            vx_range=tuple(cmd_cfg.get("vx_range", [0.0, 0.0])),
            vy_range=tuple(cmd_cfg.get("vy_range", [0.0, 0.0])),
            yaw_rate_range=tuple(cmd_cfg.get("yaw_rate_range", [0.0, 0.0])),
            height_range=tuple(cmd_cfg.get("height_range", [0.71, 0.77])),
        )

    return train(make_envs_fn, manifest, ppo_cfg, output_dir)
