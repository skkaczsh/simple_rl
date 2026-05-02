"""Model registry for tracking model versions and their lifecycle stages."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_STAGES = {"draft", "candidate", "staged", "stable", "rejected", "rolled_back", "archived"}
VALID_TRANSITIONS = {
    "draft": {"candidate", "rejected", "archived"},
    "candidate": {"staged", "rejected", "archived"},
    "staged": {"stable", "rolled_back", "archived"},
    "stable": {"rolled_back", "archived"},
    "rejected": {"archived"},
    "rolled_back": {"archived"},
    "archived": set(),
}


@dataclass
class ModelVersion:
    model_id: str
    stage: str = "draft"
    created_at: str = ""
    parent_version: str | None = None
    training_config: dict[str, Any] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)
    capability_bounds: dict[str, Any] = field(default_factory=dict)
    artifact_path: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelVersion:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """Manages model versions and their lifecycle stages.

    Registry is stored as a JSON file in the registry directory.
    """

    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.versions: list[ModelVersion] = []
        self._load()

    def _registry_path(self) -> Path:
        return self.registry_dir / "registry.json"

    def _load(self) -> None:
        path = self._registry_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.versions = [ModelVersion.from_dict(v) for v in data.get("versions", [])]

    def _save(self) -> None:
        data = {"versions": [v.to_dict() for v in self.versions]}
        self._registry_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register(self, version: ModelVersion) -> str:
        """Register a new model version. Returns the model_id."""
        if version.stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage: {version.stage}")
        self.versions.append(version)
        self._save()
        return version.model_id

    def promote(self, model_id: str, to_stage: str) -> None:
        """Promote a model to a new stage."""
        if to_stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage: {to_stage}")
        version = self.get(model_id)
        if version is None:
            raise ValueError(f"Model not found: {model_id}")
        if to_stage not in VALID_TRANSITIONS.get(version.stage, set()):
            raise ValueError(f"Cannot promote from {version.stage} to {to_stage}")
        version.stage = to_stage
        self._save()

    def get(self, model_id: str) -> ModelVersion | None:
        """Get a model version by ID."""
        for v in self.versions:
            if v.model_id == model_id:
                return v
        return None

    def get_latest(self, stage: str) -> ModelVersion | None:
        """Get the latest model version at a given stage."""
        candidates = [v for v in self.versions if v.stage == stage]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.created_at)

    def list_versions(self, stage: str | None = None) -> list[ModelVersion]:
        """List all versions, optionally filtered by stage."""
        if stage is None:
            return list(self.versions)
        return [v for v in self.versions if v.stage == stage]
