from pathlib import Path

from northstar.embodiment.manifest import EmbodimentManifest, load_manifest


def test_load_manifest_counts_joints_and_feet():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))

    assert manifest.embodiment_id == "unitree_g1_43dof_sim_v0"
    assert manifest.active_joint_count == 12
    assert manifest.foot_contact_site_count == 2
    assert manifest.action_limit_rad == 0.25
    assert manifest.velocity_limit_rad_s == 1.0
    assert manifest.torque_limit_nm == 20.0


def test_manifest_from_dict_rejects_empty_joint_names():
    payload = {
        "embodiment_id": "bad",
        "active_joint_names": [],
        "foot_contact_site_names": ["left_foot"],
        "action_limit_rad": 0.25,
        "velocity_limit_rad_s": 1.0,
        "torque_limit_nm": 20.0,
    }

    try:
        EmbodimentManifest.from_dict(payload)
    except ValueError as exc:
        assert "active_joint_names" in str(exc)
    else:
        raise AssertionError("expected active_joint_names validation error")
