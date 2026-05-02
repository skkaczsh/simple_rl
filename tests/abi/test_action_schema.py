from pathlib import Path

from northstar.abi.action import make_zero_action
from northstar.abi.validators import ValidationError, validate_action
from northstar.embodiment.manifest import load_manifest


def test_zero_action_matches_manifest_joint_count():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    action = make_zero_action("act_1", manifest, action_source="debug_policy")

    validate_action(action, manifest)
    assert len(action["joint_position_delta_rad"]) == manifest.active_joint_count
    assert action["clipped"] is False


def test_action_rejects_wrong_joint_count():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    action = make_zero_action("act_bad", manifest, action_source="debug_policy")
    action["joint_position_delta_rad"] = [0.0]

    try:
        validate_action(action, manifest)
    except ValidationError as exc:
        assert "joint_position_delta_rad" in str(exc)
    else:
        raise AssertionError("expected joint count validation error")
