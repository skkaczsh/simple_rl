from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_scenario_set(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    required = ["schema_version", "scenario_set_id", "phase", "defaults", "scenarios"]
    for key in required:
        if key not in payload:
            raise ValueError(f"scenario_set.{key} is required")
    return payload
