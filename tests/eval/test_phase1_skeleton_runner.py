import json
from pathlib import Path

from northstar.eval.runner import run_scenario_set


def test_phase1_skeleton_runner_generates_report_and_episodes(tmp_path: Path):
    report = run_scenario_set(
        scenario_set_path=Path("configs/eval/phase1_skeleton_scenarios.yaml"),
        manifest_path=Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"),
        output_dir=tmp_path,
    )

    assert report["summary"]["pass"] is True
    assert report["summary"]["schema_validation_pass_rate"] == 1.0
    assert report["summary"]["replay_validation_pass_rate"] == 1.0
    report_path = Path(report["artifacts"]["report_path"])
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert loaded["schema_version"] == "evaluation_report.v0"
