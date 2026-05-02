from northstar.abi.command import make_locomotion_command
from northstar.abi.validators import ValidationError, validate_command


def test_valid_phase1_command_passes():
    command = make_locomotion_command(
        command_id="cmd_1",
        target_velocity_base_m_s=[0.2, 0.0, 0.0],
        target_yaw_rate_rad_s=0.3,
    )

    validate_command(command)


def test_phase1_command_rejects_vertical_velocity():
    command = make_locomotion_command(
        command_id="cmd_bad",
        target_velocity_base_m_s=[0.0, 0.0, 0.1],
        target_yaw_rate_rad_s=0.0,
    )

    try:
        validate_command(command)
    except ValidationError as exc:
        assert "target_velocity_base_m_s.z" in str(exc)
    else:
        raise AssertionError("expected vertical velocity validation error")


def test_phase1_command_rejects_enabled_upper_body():
    command = make_locomotion_command(
        command_id="cmd_upper",
        target_velocity_base_m_s=[0.0, 0.0, 0.0],
        target_yaw_rate_rad_s=0.0,
    )
    command["mode_mask"]["upper_body"] = True

    try:
        validate_command(command)
    except ValidationError as exc:
        assert "upper_body" in str(exc)
    else:
        raise AssertionError("expected upper_body validation error")
