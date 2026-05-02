import json
from pathlib import Path

from northstar.abi.command import make_locomotion_command
from northstar.abi.action import make_zero_action
from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.embodiment.manifest import load_manifest
from northstar.episode_io.episode_logger import EpisodeLogger


def test_episode_logger_writes_required_files(tmp_path: Path):
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.0, 0.0, 0.0], 0.0)
    action = make_zero_action("act", manifest, "test")
    observation = make_observation(manifest, command, 0.0, 0.02)
    logger = EpisodeLogger(tmp_path, run_id="run_test", episode_id="ep_test")

    logger.start(
        phase="phase_1_skeleton",
        scenario_id="phase1_stand_smoke",
        seed=1,
        abi_version="abi.northstar.v0",
        embodiment_id=manifest.embodiment_id,
        env_id="mock_phase1_env_v0",
        policy_id="debug_noop_v0",
    )
    logger.append_step(
        step_index=0,
        time_s=0.0,
        observation=observation,
        command=command,
        action=action,
        confidence=make_confidence(1.0, 1.0, 1.0, 1.0),
        dangerous_signal=make_dangerous_signal(0.0, []),
        reward_debug={"mock_reward": 1.0},
        terminated=False,
        truncated=False,
        info={},
    )
    logger.append_event({"schema_version": "event_record.v0", "episode_id": "ep_test", "step_index": 0, "event_type": "episode_end", "severity": "info", "source": "test", "payload": {}})
    logger.finalize(metrics={"step_count": 1}, termination_reason="time_limit")

    episode_dir = tmp_path / "episodes" / "ep_test"
    assert (episode_dir / "episode_manifest.json").exists()
    assert (episode_dir / "steps.jsonl").exists()
    assert (episode_dir / "events.jsonl").exists()
    assert (episode_dir / "metrics.json").exists()
    manifest_payload = json.loads((episode_dir / "episode_manifest.json").read_text())
    assert manifest_payload["artifact_hashes"]["steps_jsonl"].startswith("sha256:")
