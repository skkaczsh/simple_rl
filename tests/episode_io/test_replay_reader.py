from pathlib import Path

from northstar.episode_io.replay_reader import read_jsonl, replay_episode


def test_read_jsonl_reads_records(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_replay_episode_recomputes_step_count(tmp_path: Path):
    episode_dir = tmp_path / "episodes" / "ep_1"
    episode_dir.mkdir(parents=True)
    (episode_dir / "steps.jsonl").write_text(
        '{"observation": {"base_height_m": 0.74, "base_linear_velocity_m_s": [0.0, 0.0, 0.0], "base_angular_velocity_rad_s": [0.0, 0.0, 0.0]}, "command": {"locomotion": {"target_velocity_base_m_s": [0.0, 0.0, 0.0], "target_yaw_rate_rad_s": 0.0}}, "action": {"clipped": false}, "dangerous_signal": {"triggered": []}}\n',
        encoding="utf-8",
    )

    metrics = replay_episode(episode_dir, default_base_height_m=0.74)

    assert metrics["step_count"] == 1
