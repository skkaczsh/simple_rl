from northstar.school.sample_builder import build_sample_for_event


def test_build_sample_for_near_fall_event():
    event = {
        "event_type": "near_fall",
        "step_index": 10,
        "severity": "warning",
        "payload": {"base_height_m": 0.42},
    }

    sample = build_sample_for_event(
        event=event,
        run_id="run_1",
        episode_id="ep_1",
        artifact_uri="runs/run_1/episodes/ep_1",
        replay_valid=True,
        artifact_hash_valid=True,
    )

    assert sample["schema_version"] == "school_sample_envelope.v0"
    assert sample["segment_type"] == "near_failure"
    assert sample["labels"]["usable_for_training"] is True
    assert sample["priority"] > 0.0
