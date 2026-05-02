from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Protocol

from northstar.abi.action import make_zero_action
from northstar.embodiment.manifest import EmbodimentManifest


class DebugPolicy(Protocol):
    policy_id: str

    def act(self, observation: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class NoopPolicy:
    manifest: EmbodimentManifest
    policy_id: str = "debug_noop_v0"

    def act(self, observation: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
        return make_zero_action("act_noop", self.manifest, self.policy_id)


@dataclass
class RandomLegalPolicy:
    manifest: EmbodimentManifest
    rng: random.Random
    policy_id: str = "debug_random_legal_v0"

    def act(self, observation: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
        action = make_zero_action("act_random", self.manifest, self.policy_id)
        action["joint_position_delta_rad"] = [
            self.rng.uniform(-self.manifest.action_limit_rad, self.manifest.action_limit_rad)
            for _ in range(self.manifest.active_joint_count)
        ]
        return action


@dataclass
class SimplePDPolicy:
    manifest: EmbodimentManifest
    policy_id: str = "debug_simple_pd_v0"

    def act(self, observation: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
        action = make_zero_action("act_simple_pd", self.manifest, self.policy_id)
        target_vx = float(command["locomotion"]["target_velocity_base_m_s"][0])
        target_yaw = float(command["locomotion"]["target_yaw_rate_rad_s"])
        base_value = max(-0.05, min(0.05, 0.05 * target_vx + 0.02 * target_yaw))
        action["joint_position_delta_rad"] = [base_value] * self.manifest.active_joint_count
        return action


def get_debug_policy(
    policy_id: str,
    manifest: EmbodimentManifest,
    rng: random.Random | None = None,
) -> DebugPolicy:
    if policy_id == "debug_noop_v0":
        return NoopPolicy(manifest)
    if policy_id == "debug_random_legal_v0":
        return RandomLegalPolicy(manifest, rng or random.Random(0))
    if policy_id == "debug_simple_pd_v0":
        return SimplePDPolicy(manifest)
    raise ValueError(f"unknown debug policy: {policy_id}")
