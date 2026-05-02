from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_evaluation_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = run_dir / "evaluation_report.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return path
