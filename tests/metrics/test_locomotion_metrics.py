from northstar.metrics.locomotion import rmse, summarize_steps


def test_rmse_computes_root_mean_squared_error():
    assert rmse([0.0, 1.0], [0.0, 3.0]) == 1.4142135623730951


def test_summarize_steps_counts_fall_and_action_clip_events():
    steps = [
        {
            "observation": {"base_height_m": 0.74, "base_linear_velocity_m_s": [0.0, 0.0, 0.0], "base_angular_velocity_rad_s": [0.0, 0.0, 0.0]},
            "command": {"locomotion": {"target_velocity_base_m_s": [0.0, 0.0, 0.0], "target_yaw_rate_rad_s": 0.0}},
            "action": {"clipped": False},
            "dangerous_signal": {"triggered": []},
        },
        {
            "observation": {"base_height_m": 0.44, "base_linear_velocity_m_s": [0.2, 0.0, 0.0], "base_angular_velocity_rad_s": [0.0, 0.0, 0.1]},
            "command": {"locomotion": {"target_velocity_base_m_s": [0.0, 0.0, 0.0], "target_yaw_rate_rad_s": 0.0}},
            "action": {"clipped": True},
            "dangerous_signal": {"triggered": ["near_fall"]},
        },
    ]

    summary = summarize_steps(steps, default_base_height_m=0.74)

    assert summary["step_count"] == 2
    assert summary["near_fall_count"] == 1
    assert summary["action_clipping_count"] == 1
