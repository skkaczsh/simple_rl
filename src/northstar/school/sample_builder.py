from __future__ import annotations

from typing import Any

from northstar.abi.school_sample import validate_school_sample
from northstar.school.priority import score_event_priority


EVENT_TO_SEGMENT = {
    "near_fall": "near_failure",
    "fall": "fall",
    "invalid_command": "invalid_command",
    "action_clip": "action_clip",
    "tracking_error_high": "tracking_error_high",
    "fallback_transition": "fallback_transition",
    "event_injection": "event_injection",
}


def build_sample_for_event(
    event: dict[str, Any],
    run_id: str,
    episode_id: str,
    artifact_uri: str,
    replay_valid: bool,
    artifact_hash_valid: bool,
) -> dict[str, Any]:
    event_type = str(event["event_type"])
    step = int(event["step_index"])
    segment_type = EVENT_TO_SEGMENT.get(event_type, "event_injection")
    sample = {
        "schema_version": "school_sample_envelope.v0",
        "sample_id": f"sample_{run_id}_{episode_id}_{step}_{event_type}",
        "source": "phase1_skeleton_eval",
        "phase": "phase_1_skeleton",
        "source_episode_id": episode_id,
        "segment_type": segment_type,
        "step_range": [max(0, step - 20), step + 20],
        "priority": score_event_priority(event_type, replay_valid, artifact_hash_valid),
        "labels": {
            "usable_for_training": bool(replay_valid and artifact_hash_valid),
            "usable_for_release_gate": bool(replay_valid and artifact_hash_valid and event_type in {"near_fall", "fall", "action_clip"}),
            "requires_human_review": False,
        },
        "artifact_uri": artifact_uri,
        "metrics": {
            "event_step": step,
            "event_type": event_type,
        },
        "data_quality": {
            "schema_valid": True,
            "artifact_hash_valid": bool(artifact_hash_valid),
            "replay_valid": bool(replay_valid),
        },
    }
    validate_school_sample(sample)
    return sample
