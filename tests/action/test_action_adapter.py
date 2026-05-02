from pathlib import Path

from northstar.abi.action import make_zero_action
from northstar.action.adapter import clip_action
from northstar.embodiment.manifest import load_manifest


def test_clip_action_clips_position_velocity_and_torque():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    action = make_zero_action("act_clip", manifest, "test")
    action["joint_position_delta_rad"][0] = 99.0
    action["joint_velocity_delta_rad_s"][1] = -99.0
    action["feedforward_torque_nm"][2] = 99.0

    clipped, events = clip_action(action, manifest, episode_id="ep_1", step_index=3)

    assert clipped["clipped"] is True
    assert clipped["joint_position_delta_rad"][0] == manifest.action_limit_rad
    assert clipped["joint_velocity_delta_rad_s"][1] == -manifest.velocity_limit_rad_s
    assert clipped["feedforward_torque_nm"][2] == manifest.torque_limit_nm
    assert events[0]["event_type"] == "action_clip"
    assert events[0]["step_index"] == 3
