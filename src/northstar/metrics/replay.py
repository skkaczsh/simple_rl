from __future__ import annotations

from pathlib import Path
from typing import Any

from northstar.episode_io.replay_reader import replay_episode


def replay_validation_passes(
    episode_dir: Path,
    saved_metrics: dict[str, Any],
    default_base_height_m: float,
) -> bool:
    recomputed = replay_episode(episode_dir, default_base_height_m)
    return recomputed.get("step_count") == saved_metrics.get("step_count")
