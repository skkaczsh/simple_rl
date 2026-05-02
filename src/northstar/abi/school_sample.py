from __future__ import annotations

from typing import Any


def validate_school_sample(sample: dict[str, Any]) -> None:
    required = [
        "schema_version",
        "sample_id",
        "source",
        "phase",
        "source_episode_id",
        "segment_type",
        "step_range",
        "priority",
        "labels",
        "artifact_uri",
        "metrics",
        "data_quality",
    ]
    for key in required:
        if key not in sample:
            raise ValueError(f"school_sample.{key} is required")
    if sample["schema_version"] != "school_sample_envelope.v0":
        raise ValueError("school_sample.schema_version must be school_sample_envelope.v0")
    if len(sample["step_range"]) != 2:
        raise ValueError("school_sample.step_range must have two values")
