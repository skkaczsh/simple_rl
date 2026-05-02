from pathlib import Path
import random

from northstar.abi.command import make_locomotion_command
from northstar.abi.observation import make_observation
from northstar.embodiment.manifest import load_manifest
from northstar.policy.debug_baselines import get_debug_policy


def test_noop_policy_outputs_zero_action():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, 0.0, 0.02)
    policy = get_debug_policy("debug_noop_v0", manifest)

    action = policy.act(observation, command)

    assert action["action_source"] == "debug_noop_v0"
    assert all(value == 0.0 for value in action["joint_position_delta_rad"])


def test_random_legal_policy_is_deterministic_with_rng_seed():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, 0.0, 0.02)
    policy_a = get_debug_policy("debug_random_legal_v0", manifest, rng=random.Random(7))
    policy_b = get_debug_policy("debug_random_legal_v0", manifest, rng=random.Random(7))

    assert policy_a.act(observation, command) == policy_b.act(observation, command)


def test_simple_pd_policy_responds_to_forward_velocity_command():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.4, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, 0.0, 0.02)
    policy = get_debug_policy("debug_simple_pd_v0", manifest)

    action = policy.act(observation, command)

    assert any(value != 0.0 for value in action["joint_position_delta_rad"])
