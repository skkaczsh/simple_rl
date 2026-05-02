import math
from pathlib import Path

from northstar.rewards.locomotion import (
    RewardConfig,
    compute_velocity_reward,
    compute_yaw_rate_reward,
    compute_base_height_reward,
    compute_upright_reward,
    compute_foot_slip_penalty,
    compute_action_rate_penalty,
    compute_reward,
)


def test_velocity_reward_perfect_tracking():
    assert compute_velocity_reward(0.5, 0.0, 0.5, 0.0, 0.25) == 1.0


def test_velocity_reward_degrades_with_error():
    perfect = compute_velocity_reward(0.5, 0.0, 0.5, 0.0, 0.25)
    noisy = compute_velocity_reward(0.5, 0.0, 0.7, 0.0, 0.25)
    assert perfect > noisy
    assert noisy > 0.0


def test_yaw_rate_reward_perfect_tracking():
    assert compute_yaw_rate_reward(0.3, 0.3, 0.35) == 1.0


def test_base_height_reward_default_height():
    assert compute_base_height_reward(0.74, 0.74, 0.06) == 1.0


def test_upright_reward_full_gravity():
    assert compute_upright_reward(-1.0) == 1.0


def test_upright_reward_tilted():
    reward = compute_upright_reward(-0.5)
    assert reward == 0.0


def test_foot_slip_penalty_no_slip():
    penalty = compute_foot_slip_penalty([True, True], [[0.0, 0.0], [0.0, 0.0]])
    assert penalty == 0.0


def test_foot_slip_penalty_with_slip():
    penalty = compute_foot_slip_penalty([True, True], [[1.0, 0.0], [0.0, 0.0]])
    assert penalty == 1.0


def test_action_rate_penalty_same_action():
    assert compute_action_rate_penalty([0.1, 0.2], [0.1, 0.2]) == 0.0


def test_action_rate_penalty_different_action():
    penalty = compute_action_rate_penalty([0.1, 0.2], [0.3, 0.4])
    assert penalty > 0.0


def test_reward_config_from_yaml():
    cfg = RewardConfig.from_yaml(Path("configs/rewards/phase1_locomotion_rewards.yaml"))
    assert cfg.w_alive == 2.0
    assert cfg.w_vel_xy == 2.0
    assert cfg.sigma_vel == 0.25


def test_compute_reward_zero_command_standing():
    cfg = RewardConfig()
    breakdown = compute_reward(
        cfg,
        terminated=False,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0, cmd_height=0.74,
        base_vx=0.0, base_vy=0.0, base_yaw_rate=0.0, base_height=0.74,
        projected_gravity_z=-1.0,
        foot_contact=[True, True],
        foot_velocities_xy=[[0.0, 0.0], [0.0, 0.0]],
        current_action=[0.0] * 12,
        previous_action=[0.0] * 12,
        joint_positions=[0.0] * 12,
        joint_limits_lower=[-1.0] * 12,
        joint_limits_upper=[1.0] * 12,
        torques=[0.0] * 12,
        torque_limits=[20.0] * 12,
        joint_velocities=[0.0] * 12,
        stop_request=False,
        brace_request=False,
        collision_detected=False,
    )
    assert breakdown.r_alive == 1.0
    assert breakdown.r_vel_xy == 1.0
    assert breakdown.r_base_height == 1.0
    assert breakdown.r_upright == 1.0
    assert breakdown.total > 0.0


def test_compute_reward_terminated():
    cfg = RewardConfig()
    breakdown = compute_reward(
        cfg,
        terminated=True,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0, cmd_height=0.74,
        base_vx=0.0, base_vy=0.0, base_yaw_rate=0.0, base_height=0.74,
        projected_gravity_z=-1.0,
        foot_contact=[True, True],
        foot_velocities_xy=[[0.0, 0.0], [0.0, 0.0]],
        current_action=[0.0] * 12,
        previous_action=[0.0] * 12,
        joint_positions=[0.0] * 12,
        joint_limits_lower=[-1.0] * 12,
        joint_limits_upper=[1.0] * 12,
        torques=[0.0] * 12,
        torque_limits=[20.0] * 12,
        joint_velocities=[0.0] * 12,
        stop_request=False,
        brace_request=False,
        collision_detected=False,
    )
    assert breakdown.r_alive == 0.0


def test_compute_reward_collision_penalty():
    cfg = RewardConfig()
    no_collision = compute_reward(
        cfg,
        terminated=False,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0, cmd_height=0.74,
        base_vx=0.0, base_vy=0.0, base_yaw_rate=0.0, base_height=0.74,
        projected_gravity_z=-1.0,
        foot_contact=[True, True],
        foot_velocities_xy=[[0.0, 0.0], [0.0, 0.0]],
        current_action=[0.0] * 12,
        previous_action=[0.0] * 12,
        joint_positions=[0.0] * 12,
        joint_limits_lower=[-1.0] * 12,
        joint_limits_upper=[1.0] * 12,
        torques=[0.0] * 12,
        torque_limits=[20.0] * 12,
        joint_velocities=[0.0] * 12,
        stop_request=False,
        brace_request=False,
        collision_detected=False,
    )
    with_collision = compute_reward(
        cfg,
        terminated=False,
        cmd_vx=0.0, cmd_vy=0.0, cmd_yaw_rate=0.0, cmd_height=0.74,
        base_vx=0.0, base_vy=0.0, base_yaw_rate=0.0, base_height=0.74,
        projected_gravity_z=-1.0,
        foot_contact=[True, True],
        foot_velocities_xy=[[0.0, 0.0], [0.0, 0.0]],
        current_action=[0.0] * 12,
        previous_action=[0.0] * 12,
        joint_positions=[0.0] * 12,
        joint_limits_lower=[-1.0] * 12,
        joint_limits_upper=[1.0] * 12,
        torques=[0.0] * 12,
        torque_limits=[20.0] * 12,
        joint_velocities=[0.0] * 12,
        stop_request=False,
        brace_request=False,
        collision_detected=True,
    )
    assert no_collision.total > with_collision.total
