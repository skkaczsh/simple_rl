from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from northstar.metrics.locomotion import summarize_steps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def replay_episode(episode_dir: Path, default_base_height_m: float) -> dict[str, Any]:
    steps = read_jsonl(episode_dir / "steps.jsonl")
    return summarize_steps(steps, default_base_height_m)
