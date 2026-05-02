from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbodimentManifest:
    embodiment_id: str
    active_joint_names: list[str]
    foot_contact_site_names: list[str]
    action_limit_rad: float
    velocity_limit_rad_s: float
    torque_limit_nm: float
    default_base_height_m: float

    @property
    def active_joint_count(self) -> int:
        return len(self.active_joint_names)

    @property
    def foot_contact_site_count(self) -> int:
        return len(self.foot_contact_site_names)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmbodimentManifest":
        active_joint_names = list(payload.get("active_joint_names", []))
        foot_contact_site_names = list(payload.get("foot_contact_site_names", []))
        if not active_joint_names:
            raise ValueError("active_joint_names must not be empty")
        if not foot_contact_site_names:
            raise ValueError("foot_contact_site_names must not be empty")
        return cls(
            embodiment_id=str(payload["embodiment_id"]),
            active_joint_names=[str(name) for name in active_joint_names],
            foot_contact_site_names=[str(name) for name in foot_contact_site_names],
            action_limit_rad=float(payload["action_limit_rad"]),
            velocity_limit_rad_s=float(payload["velocity_limit_rad_s"]),
            torque_limit_nm=float(payload["torque_limit_nm"]),
            default_base_height_m=float(payload.get("default_base_height_m", 0.74)),
        )


def load_manifest(path: Path) -> EmbodimentManifest:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EmbodimentManifest.from_dict(payload)
