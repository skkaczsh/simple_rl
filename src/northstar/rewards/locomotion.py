from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import yaml
from pathlib import Path


@dataclass(frozen=True)
class RewardConfig:
    w_alive: float = 1.0
    w_vel_xy: float = 2.0
    w_yaw_rate: float = 1.0
    w_base_height: float = 0.8
    w_upright: float = 1.0
    w_contact: float = 0.2
    w_stop_brace: float = 0.5
    w_foot_slip: float = 0.2
    w_action_rate: float = 0.05
    w_joint_limit: float = 0.5
    w_torque: float = 0.02
    w_energy: float = 0.01
    w_collision: float = 1.0
    sigma_vel: float = 0.25
    sigma_yaw: float = 0.35
    sigma_height: float = 0.06

    @classmethod
    def from_yaml(cls, path: Path) -> "RewardConfig":
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        weights = payload.get("weights", {})
        sigma = payload.get("sigma", {})
        return cls(
            w_alive=weights.get("w_alive", 1.0),
            w_vel_xy=weights.get("w_vel_xy", 2.0),
            w_yaw_rate=weights.get("w_yaw_rate", 1.0),
            w_base_height=weights.get("w_base_height", 0.8),
            w_upright=weights.get("w_upright", 1.0),
            w_contact=weights.get("w_contact", 0.2),
            w_stop_brace=weights.get("w_stop_brace", 0.5),
            w_foot_slip=weights.get("w_foot_slip", 0.2),
            w_action_rate=weights.get("w_action_rate", 0.05),
            w_joint_limit=weights.get("w_joint_limit", 0.5),
            w_torque=weights.get("w_torque", 0.02),
            w_energy=weights.get("w_energy", 0.01),
            w_collision=weights.get("w_collision", 1.0),
            sigma_vel=sigma.get("sigma_vel", 0.25),
            sigma_yaw=sigma.get("sigma_yaw", 0.35),
            sigma_height=sigma.get("sigma_height", 0.06),
        )


def _exp_neg_sq(x: float, sigma: float) -> float:
    return math.exp(-(x * x) / (sigma * sigma))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_velocity_reward(
    cmd_vx: float, cmd_vy: float, base_vx: float, base_vy: float, sigma_vel: float
) -> float:
    dx = cmd_vx - base_vx
    dy = cmd_vy - base_vy
    return _exp_neg_sq(math.sqrt(dx * dx + dy * dy), sigma_vel)


def compute_yaw_rate_reward(cmd_yaw: float, base_yaw: float, sigma_yaw: float) -> float:
    return _exp_neg_sq(cmd_yaw - base_yaw, sigma_yaw)


def compute_base_height_reward(cmd_h: float, base_h: float, sigma_height: float) -> float:
    return _exp_neg_sq(cmd_h - base_h, sigma_height)


def compute_upright_reward(projected_gravity_z: float) -> float:
    return _clamp((abs(projected_gravity_z) - 0.5) / 0.5, 0.0, 1.0)


def compute_foot_slip_penalty(
    foot_contact: list[bool], foot_velocities_xy: list[list[float]]
) -> float:
    penalty = 0.0
    for contact, vel in zip(foot_contact, foot_velocities_xy):
        if contact:
            penalty += vel[0] * vel[0] + vel[1] * vel[1]
    return penalty


def compute_action_rate_penalty(
    current_action: list[float], previous_action: list[float]
) -> float:
    if not current_action or not previous_action:
        return 0.0
    total = 0.0
    for c, p in zip(current_action, previous_action):
        diff = c - p
        total += diff * diff
    return total / len(current_action)


def compute_joint_limit_penalty(
    joint_positions: list[float], joint_limits_lower: list[float], joint_limits_upper: list[float]
) -> float:
    if not joint_positions:
        return 0.0
    total = 0.0
    for pos, lo, hi in zip(joint_positions, joint_limits_lower, joint_limits_upper):
        range_half = (hi - lo) / 2.0
        if range_half <= 0:
            continue
        center = (hi + lo) / 2.0
        ratio = abs(pos - center) / range_half
        total += ratio * ratio
    return total / len(joint_positions)


def compute_torque_penalty(torques: list[float], torque_limits: list[float]) -> float:
    if not torques:
        return 0.0
    total = 0.0
    for t, lim in zip(torques, torque_limits):
        if lim <= 0:
            continue
        ratio = t / lim
        total += ratio * ratio
    return total / len(torques)


def compute_energy_penalty(torques: list[float], joint_velocities: list[float]) -> float:
    if not torques or not joint_velocities:
        return 0.0
    total = 0.0
    for t, v in zip(torques, joint_velocities):
        total += abs(t * v)
    return total / len(torques)


def compute_contact_reward(foot_contact: list[bool], expected_contact: bool) -> float:
    if not foot_contact:
        return 0.0
    matching = sum(1 for c in foot_contact if c == expected_contact)
    return matching / len(foot_contact)


@dataclass
class RewardBreakdown:
    total: float
    r_alive: float
    r_vel_xy: float
    r_yaw_rate: float
    r_base_height: float
    r_upright: float
    r_contact: float
    r_stop_brace: float
    p_foot_slip: float
    p_action_rate: float
    p_joint_limit: float
    p_torque: float
    p_energy: float
    p_collision: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "r_alive": self.r_alive,
            "r_vel_xy": self.r_vel_xy,
            "r_yaw_rate": self.r_yaw_rate,
            "r_base_height": self.r_base_height,
            "r_upright": self.r_upright,
            "r_contact": self.r_contact,
            "r_stop_brace": self.r_stop_brace,
            "p_foot_slip": self.p_foot_slip,
            "p_action_rate": self.p_action_rate,
            "p_joint_limit": self.p_joint_limit,
            "p_torque": self.p_torque,
            "p_energy": self.p_energy,
            "p_collision": self.p_collision,
        }


def compute_reward(
    cfg: RewardConfig,
    *,
    terminated: bool,
    cmd_vx: float,
    cmd_vy: float,
    cmd_yaw_rate: float,
    cmd_height: float,
    base_vx: float,
    base_vy: float,
    base_yaw_rate: float,
    base_height: float,
    projected_gravity_z: float,
    foot_contact: list[bool],
    foot_velocities_xy: list[list[float]],
    current_action: list[float],
    previous_action: list[float],
    joint_positions: list[float],
    joint_limits_lower: list[float],
    joint_limits_upper: list[float],
    torques: list[float],
    torque_limits: list[float],
    joint_velocities: list[float],
    stop_request: bool,
    brace_request: bool,
    collision_detected: bool,
) -> RewardBreakdown:
    r_alive = 0.0 if terminated else 1.0
    r_vel_xy = compute_velocity_reward(cmd_vx, cmd_vy, base_vx, base_vy, cfg.sigma_vel)
    r_yaw_rate = compute_yaw_rate_reward(cmd_yaw_rate, base_yaw_rate, cfg.sigma_yaw)
    r_base_height = compute_base_height_reward(cmd_height, base_height, cfg.sigma_height)
    r_upright = compute_upright_reward(projected_gravity_z)
    r_contact = compute_contact_reward(foot_contact, expected_contact=True)
    r_stop_brace = 0.0
    if stop_request:
        speed = math.sqrt(base_vx * base_vx + base_vy * base_vy)
        r_stop_brace = _clamp(1.0 - speed / 0.5, 0.0, 1.0)
    elif brace_request:
        r_stop_brace = r_upright

    p_foot_slip = compute_foot_slip_penalty(foot_contact, foot_velocities_xy)
    p_action_rate = compute_action_rate_penalty(current_action, previous_action)
    p_joint_limit = compute_joint_limit_penalty(joint_positions, joint_limits_lower, joint_limits_upper)
    p_torque = compute_torque_penalty(torques, torque_limits)
    p_energy = compute_energy_penalty(torques, joint_velocities)
    p_collision = 1.0 if collision_detected else 0.0

    total = (
        cfg.w_alive * r_alive
        + cfg.w_vel_xy * r_vel_xy
        + cfg.w_yaw_rate * r_yaw_rate
        + cfg.w_base_height * r_base_height
        + cfg.w_upright * r_upright
        + cfg.w_contact * r_contact
        + cfg.w_stop_brace * r_stop_brace
        - cfg.w_foot_slip * p_foot_slip
        - cfg.w_action_rate * p_action_rate
        - cfg.w_joint_limit * p_joint_limit
        - cfg.w_torque * p_torque
        - cfg.w_energy * p_energy
        - cfg.w_collision * p_collision
    )

    return RewardBreakdown(
        total=total,
        r_alive=r_alive,
        r_vel_xy=r_vel_xy,
        r_yaw_rate=r_yaw_rate,
        r_base_height=r_base_height,
        r_upright=r_upright,
        r_contact=r_contact,
        r_stop_brace=r_stop_brace,
        p_foot_slip=p_foot_slip,
        p_action_rate=p_action_rate,
        p_joint_limit=p_joint_limit,
        p_torque=p_torque,
        p_energy=p_energy,
        p_collision=p_collision,
    )
