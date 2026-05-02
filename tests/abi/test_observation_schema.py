from pathlib import Path

from northstar.abi.command import make_locomotion_command
from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.abi.validators import ValidationError, validate_observation
from northstar.embodiment.manifest import load_manifest


def test_observation_matches_manifest_and_contains_command():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd_1", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, timestamp_s=0.0, dt_s=0.02)

    validate_observation(observation, manifest)
    assert observation["mode_mask"]["upper_body"] is False
    assert len(observation["joint_position_rad"]) == manifest.active_joint_count


def test_observation_rejects_wrong_foot_contact_count():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd_1", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, timestamp_s=0.0, dt_s=0.02)
    observation["foot_contact"] = [True]

    try:
        validate_observation(observation, manifest)
    except ValidationError as exc:
        assert "foot_contact" in str(exc)
    else:
        raise AssertionError("expected foot_contact validation error")


def test_signal_helpers_return_expected_schema_versions():
    confidence = make_confidence(overall=0.5, stability=0.6, tracking=0.7, fallback=1.0)
    dangerous = make_dangerous_signal(overall_risk=0.2, triggered=["near_fall"])

    assert confidence["schema_version"] == "confidence.northstar.v0"
    assert dangerous["schema_version"] == "dangerous_signal.northstar.v0"
