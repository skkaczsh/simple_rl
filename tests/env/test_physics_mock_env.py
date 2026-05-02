import math
from pathlib import Path

from northstar.embodiment.manifest import load_manifest
from northstar.env.physics_mock_env import PhysicsConfig, PhysicsMockEnv
from northstar.rewards.locomotion import RewardConfig


def _make_env(manifest_path="configs/embodiment/unitree_g1_43dof_sim_v0.json", **kwargs):
    manifest = load_manifest(Path(manifest_path))
    physics = PhysicsConfig()
    reward = RewardConfig()
    return PhysicsMockEnv(
        manifest=manifest,
        physics_config=physics,
        reward_config=reward,
        horizon_steps=100,
        **kwargs,
    )


def test_physics_env_reset_returns_valid_observation():
    env = _make_env()
    obs = env.reset(seed=42)
    assert obs["schema_version"] == "observation.northstar.v0"
    assert len(obs["joint_position_rad"]) == 12
    assert obs["base_height_m"] > 0.0


def test_physics_env_step_returns_positive_reward_standing():
    env = _make_env()
    obs = env.reset(seed=42)
    action = {
        "joint_position_delta_rad": [0.0] * 12,
        "joint_velocity_delta_rad_s": [0.0] * 12,
        "feedforward_torque_nm": [0.0] * 12,
    }
    result = env.step(action)
    assert result.reward_debug["total"] > 0.0
    assert result.observation["base_height_m"] > 0.0


def test_physics_env_terminates_on_low_height():
    env = _make_env()
    env.reset(seed=42)
    env.state.base_pos[2] = 0.3
    action = {
        "joint_position_delta_rad": [0.0] * 12,
        "joint_velocity_delta_rad_s": [0.0] * 12,
        "feedforward_torque_nm": [0.0] * 12,
    }
    result = env.step(action)
    assert result.terminated


def test_physics_env_responds_to_velocity_command():
    env = _make_env()
    env.reset(seed=42)
    env.set_command(vx=0.5, vy=0.0, yaw_rate=0.0)
    action = {
        "joint_position_delta_rad": [0.01] * 12,
        "joint_velocity_delta_rad_s": [0.0] * 12,
        "feedforward_torque_nm": [0.0] * 12,
    }
    for _ in range(50):
        result = env.step(action)
    assert result.observation["base_linear_velocity_m_s"][0] > 0.0
