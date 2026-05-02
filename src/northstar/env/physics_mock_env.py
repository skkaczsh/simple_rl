from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.embodiment.manifest import EmbodimentManifest
from northstar.env.adapter import StepResult
from northstar.rewards.locomotion import RewardConfig, compute_reward


@dataclass
class PhysicsState:
    step_index: int = 0
    time_s: float = 0.0
    base_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.74])
    base_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_rpy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_pos: list[float] = field(default_factory=lambda: [])
    joint_vel: list[float] = field(default_factory=lambda: [])
    joint_target: list[float] = field(default_factory=lambda: [])
    foot_contact: list[bool] = field(default_factory=lambda: [True, True])
    previous_action: list[float] = field(default_factory=lambda: [])


@dataclass
class PhysicsConfig:
    gravity_m_s2: float = -9.81
    base_mass_kg: float = 30.0
    joint_stiffness: float = 50.0
    joint_damping: float = 5.0
    ground_friction: float = 0.8
    action_noise_std: float = 0.01
    dt_s: float = 0.02
    action_scale: float = 0.25

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhysicsConfig":
        return cls(
            gravity_m_s2=d.get("gravity_m_s2", -9.81),
            base_mass_kg=d.get("base_mass_kg", 30.0),
            joint_stiffness=d.get("joint_stiffness", 50.0),
            joint_damping=d.get("joint_damping", 5.0),
            ground_friction=d.get("ground_friction", 0.8),
            action_noise_std=d.get("action_noise_std", 0.01),
            dt_s=d.get("dt_s", 0.02),
            action_scale=d.get("action_scale", 0.25),
        )


class PhysicsMockEnv:
    env_id = "physics_mock_env_v0"

    def __init__(
        self,
        manifest: EmbodimentManifest,
        physics_config: PhysicsConfig,
        reward_config: RewardConfig,
        horizon_steps: int,
        vx_range: tuple[float, float] = (0.0, 0.0),
        vy_range: tuple[float, float] = (0.0, 0.0),
        yaw_rate_range: tuple[float, float] = (0.0, 0.0),
        height_range: tuple[float, float] = (0.71, 0.77),
    ) -> None:
        self.manifest = manifest
        self.physics = physics_config
        self.reward_cfg = reward_config
        self.horizon_steps = horizon_steps
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.yaw_rate_range = yaw_rate_range
        self.height_range = height_range
        self.rng = random.Random(0)
        self.state = PhysicsState()
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_yaw = 0.0
        self._cmd_height = manifest.default_base_height_m

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = random.Random(seed)
        n = self.manifest.active_joint_count
        self.state = PhysicsState(
            base_pos=[0.0, 0.0, self.manifest.default_base_height_m],
            joint_pos=[0.0] * n,
            joint_vel=[0.0] * n,
            joint_target=[0.0] * n,
            previous_action=[0.0] * n,
        )
        self._cmd_vx = self.rng.uniform(*self.vx_range)
        self._cmd_vy = self.rng.uniform(*self.vy_range)
        self._cmd_yaw = self.rng.uniform(*self.yaw_rate_range)
        self._cmd_height = self.rng.uniform(*self.height_range)
        return self._make_observation()

    def set_command(self, vx: float, vy: float, yaw_rate: float, height: float | None = None) -> None:
        self._cmd_vx = vx
        self._cmd_vy = vy
        self._cmd_yaw = yaw_rate
        if height is not None:
            self._cmd_height = height

    def step(self, action: dict[str, Any], command: dict[str, Any] | None = None) -> StepResult:
        events: list[dict[str, Any]] = []
        joint_delta = list(action.get("joint_position_delta_rad", [0.0] * self.manifest.active_joint_count))

        if command is not None:
            loc = command.get("locomotion", {})
            self._cmd_vx = float(loc.get("target_velocity_base_m_s", [0, 0, 0])[0])
            self._cmd_vy = float(loc.get("target_velocity_base_m_s", [0, 0, 0])[1])
            self._cmd_yaw = float(loc.get("target_yaw_rate_rad_s", 0.0))
            h = loc.get("target_base_height_m", 0.0)
            if h != 0.0:
                self._cmd_height = self.manifest.default_base_height_m + h

        for i in range(len(joint_delta)):
            noise = self.rng.gauss(0, self.physics.action_noise_std)
            joint_delta[i] += noise

        s = self.state
        s.joint_target = [
            t + d * self.physics.action_scale
            for t, d in zip(s.joint_target, joint_delta)
        ]

        for _ in range(1):
            torque = [
                self.physics.joint_stiffness * (tgt - pos) - self.physics.joint_damping * vel
                for tgt, pos, vel in zip(s.joint_target, s.joint_pos, s.joint_vel)
            ]
            for i in range(len(s.joint_vel)):
                s.joint_vel[i] += (torque[i] / max(1.0, self.physics.base_mass_kg * 0.1)) * self.physics.dt_s
                s.joint_pos[i] += s.joint_vel[i] * self.physics.dt_s

        vx_error = self._cmd_vx - s.base_vel[0]
        vy_error = self._cmd_vy - s.base_vel[1]
        s.base_vel[0] += vx_error * 0.3 * self.physics.dt_s * 50
        s.base_vel[1] += vy_error * 0.3 * self.physics.dt_s * 50
        s.base_vel[2] += self.physics.gravity_m_s2 * self.physics.dt_s
        s.base_pos[0] += s.base_vel[0] * self.physics.dt_s
        s.base_pos[1] += s.base_vel[1] * self.physics.dt_s
        s.base_pos[2] += s.base_vel[2] * self.physics.dt_s

        if s.base_pos[2] <= 0.0:
            s.base_pos[2] = 0.0
            s.base_vel[2] = 0.0

        height_target = self._cmd_height
        height_error = height_target - s.base_pos[2]
        s.base_vel[2] += height_error * 2.0 * self.physics.dt_s

        yaw_error = self._cmd_yaw - s.base_ang_vel[2]
        s.base_ang_vel[2] += yaw_error * 2.0 * self.physics.dt_s
        s.base_rpy[2] += s.base_ang_vel[2] * self.physics.dt_s

        roll_perturb = self.rng.gauss(0, 0.01)
        pitch_perturb = self.rng.gauss(0, 0.01)
        s.base_rpy[0] = s.base_rpy[0] * 0.95 + roll_perturb
        s.base_rpy[1] = s.base_rpy[1] * 0.95 + pitch_perturb

        s.foot_contact = [s.base_pos[2] < self.manifest.default_base_height_m + 0.1] * 2

        s.step_index += 1
        s.time_s = s.step_index * self.physics.dt_s

        terminated = self._check_termination()
        near_fall = self._check_near_fall()

        if near_fall:
            events.append(self._event("near_fall", "warning", {"base_height_m": s.base_pos[2]}))

        projected_gravity_z = -math.cos(s.base_rpy[0]) * math.cos(s.base_rpy[1])

        foot_vel_xy = [[s.base_vel[0], s.base_vel[1]]] * 2
        torques = [
            self.physics.joint_stiffness * (tgt - pos) - self.physics.joint_damping * vel
            for tgt, pos, vel in zip(s.joint_target, s.joint_pos, s.joint_vel)
        ]
        torque_limits = [self.manifest.torque_limit_nm] * self.manifest.active_joint_count
        joint_limits_lower = [-3.14] * self.manifest.active_joint_count
        joint_limits_upper = [3.14] * self.manifest.active_joint_count

        reward_breakdown = compute_reward(
            self.reward_cfg,
            terminated=terminated,
            cmd_vx=self._cmd_vx,
            cmd_vy=self._cmd_vy,
            cmd_yaw_rate=self._cmd_yaw,
            cmd_height=self._cmd_height,
            base_vx=s.base_vel[0],
            base_vy=s.base_vel[1],
            base_yaw_rate=s.base_ang_vel[2],
            base_height=s.base_pos[2],
            projected_gravity_z=projected_gravity_z,
            foot_contact=s.foot_contact,
            foot_velocities_xy=foot_vel_xy,
            current_action=list(joint_delta),
            previous_action=list(s.previous_action),
            joint_positions=list(s.joint_pos),
            joint_limits_lower=joint_limits_lower,
            joint_limits_upper=joint_limits_upper,
            torques=torques,
            torque_limits=torque_limits,
            joint_velocities=list(s.joint_vel),
            stop_request=False,
            brace_request=False,
            collision_detected=False,
        )

        s.previous_action = list(joint_delta)

        obs = self._make_observation()
        near_fall_risk = 0.8 if near_fall else 0.0
        confidence_val = 1.0 - near_fall_risk

        return StepResult(
            observation=obs,
            confidence=make_confidence(confidence_val, confidence_val, 1.0, 1.0),
            dangerous_signal=make_dangerous_signal(
                overall_risk=near_fall_risk,
                triggered=["near_fall"] if near_fall else [],
                near_fall_risk=near_fall_risk,
            ),
            reward_debug=reward_breakdown.to_dict(),
            events=events,
            terminated=terminated,
            truncated=s.step_index >= self.horizon_steps,
            info={"env_id": self.env_id, "reward_total": reward_breakdown.total},
        )

    def _check_termination(self) -> bool:
        s = self.state
        if s.base_pos[2] < 0.45:
            return True
        if abs(s.base_rpy[0]) > 0.8 or abs(s.base_rpy[1]) > 0.8:
            return True
        return False

    def _check_near_fall(self) -> bool:
        s = self.state
        if s.base_pos[2] < self.manifest.default_base_height_m - 0.15:
            return True
        if abs(s.base_rpy[0]) > 0.4 or abs(s.base_rpy[1]) > 0.4:
            return True
        return False

    def _make_observation(self) -> dict[str, Any]:
        s = self.state
        cmd = {
            "schema_version": "command.northstar.v0",
            "command_id": f"cmd_{s.step_index}",
            "mode_mask": {
                "stand": True, "locomotion": True,
                "upper_body": False, "light_axis": False, "semantic_intent": False,
            },
            "locomotion": {
                "target_base_height_m": self._cmd_height - self.manifest.default_base_height_m,
                "target_velocity_base_m_s": [self._cmd_vx, self._cmd_vy, 0.0],
                "target_yaw_rate_rad_s": self._cmd_yaw,
                "target_heading_rad": None,
                "stop_request": False,
                "brace_request": False,
            },
            "upper_body": None,
            "light_axis_hint": None,
            "semantic_hint": None,
        }

        projected_gravity = [
            -math.sin(s.base_rpy[1]),
            math.sin(s.base_rpy[0]) * math.cos(s.base_rpy[1]),
            -math.cos(s.base_rpy[0]) * math.cos(s.base_rpy[1]),
        ]

        return {
            "schema_version": "observation.northstar.v0",
            "timestamp_s": s.time_s,
            "dt_s": self.physics.dt_s,
            "frame": "base",
            "joint_position_rad": list(s.joint_pos),
            "joint_velocity_rad_s": list(s.joint_vel),
            "base_linear_velocity_m_s": list(s.base_vel),
            "base_angular_velocity_rad_s": list(s.base_ang_vel),
            "projected_gravity_base": projected_gravity,
            "base_height_m": s.base_pos[2],
            "foot_contact": list(s.foot_contact),
            "previous_action": {
                "schema_version": "action.northstar.v0",
                "action_id": f"act_{s.step_index}",
                "joint_position_delta_rad": list(s.previous_action),
                "joint_velocity_delta_rad_s": [0.0] * self.manifest.active_joint_count,
                "feedforward_torque_nm": [0.0] * self.manifest.active_joint_count,
                "action_source": "physics_mock",
                "clipped": False,
                "clip_summary": [],
            },
            "command": cmd,
            "mode_mask": cmd["mode_mask"],
            "masks": {
                "privileged": False,
                "upper_body_command_enabled": False,
                "light_axis_enabled": False,
                "semantic_hint_enabled": False,
            },
        }

    def _event(self, event_type: str, severity: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "event_record.v0",
            "episode_id": "",
            "step_index": self.state.step_index,
            "event_type": event_type,
            "severity": severity,
            "source": self.env_id,
            "payload": payload,
        }

    def close(self) -> None:
        pass
