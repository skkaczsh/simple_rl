from pathlib import Path

from northstar.command.generator import CommandGenerator
from northstar.command.scenarios import load_scenario_set


def test_load_phase1_scenario_set():
    scenario_set = load_scenario_set(Path("configs/eval/phase1_skeleton_scenarios.yaml"))

    assert scenario_set["scenario_set_id"] == "phase1_skeleton_scenarios_v001"
    assert len(scenario_set["scenarios"]) == 8


def test_command_generator_applies_stop_schedule():
    scenario_set = load_scenario_set(Path("configs/eval/phase1_skeleton_scenarios.yaml"))
    scenario = next(s for s in scenario_set["scenarios"] if s["scenario_id"] == "phase1_stop_request_smoke")
    generator = CommandGenerator(scenario)

    command_before = generator.command_at_step(0)
    command_after = generator.command_at_step(120)

    assert command_before["locomotion"]["stop_request"] is False
    assert command_after["locomotion"]["stop_request"] is True
    assert command_after["locomotion"]["target_velocity_base_m_s"] == [0.4, 0.0, 0.0]
