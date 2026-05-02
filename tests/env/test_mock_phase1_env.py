from pathlib import Path

from northstar.abi.command import make_locomotion_command
from northstar.abi.action import make_zero_action
from northstar.embodiment.manifest import load_manifest
from northstar.env.mock_phase1_env import MockPhase1Env


def test_mock_env_reset_returns_valid_observation():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    env = MockPhase1Env(manifest=manifest, dt_s=0.02, horizon_steps=5)

    observation = env.reset(seed=1)

    assert observation["schema_version"] == "observation.northstar.v0"
    assert observation["base_height_m"] == manifest.default_base_height_m


def test_mock_env_step_applies_stop_request_event():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    env = MockPhase1Env(manifest=manifest, dt_s=0.02, horizon_steps=5)
    env.reset(seed=1)
    command = make_locomotion_command("cmd", [0.4, 0.0, 0.0], 0.0, stop_request=True)
    action = make_zero_action("act", manifest, "test")

    result = env.step(action, command)

    assert any(event["event_type"] == "stop_request" for event in result.events)
    assert result.observation["base_linear_velocity_m_s"][0] == 0.0
