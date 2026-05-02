from __future__ import annotations


def score_event_priority(event_type: str, replay_valid: bool, artifact_hash_valid: bool) -> float:
    severity = 1.0 if event_type in {"fall", "near_fall"} else 0.7 if event_type == "perturbation" else 0.5
    replay = 1.0 if replay_valid else 0.0
    metric_error = 0.8 if event_type in {"near_fall", "tracking_error_high", "action_clip", "perturbation"} else 0.4
    rarity = 0.6 if event_type in {"fall", "near_fall"} else 0.5 if event_type == "perturbation" else 0.3
    data_quality = 1.0 if replay_valid and artifact_hash_valid else 0.0
    return (
        0.30 * severity
        + 0.25 * replay
        + 0.20 * metric_error
        + 0.15 * rarity
        + 0.10 * data_quality
    )
