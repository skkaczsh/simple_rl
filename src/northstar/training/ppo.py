"""Minimal PPO training loop for Phase 1 locomotion.

This module requires PyTorch. It is designed to run on the remote GPU host.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


@dataclass
class PPOConfig:
    num_envs: int = 4
    num_iterations: int = 100
    horizon_steps: int = 500
    batch_size: int = 256
    mini_batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    save_interval: int = 10
    eval_interval: int = 10
    log_interval: int = 5
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PPOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


@dataclass
class RolloutBuffer:
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def compute_gae(self, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.stack(self.dones)
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns


def obs_to_tensor(obs: dict[str, Any], device: torch.device) -> torch.Tensor:
    joint_pos = obs.get("joint_position_rad", [])
    joint_vel = obs.get("joint_velocity_rad_s", [])
    base_vel = obs.get("base_linear_velocity_m_s", [0, 0, 0])
    base_ang = obs.get("base_angular_velocity_rad_s", [0, 0, 0])
    gravity = obs.get("projected_gravity_base", [0, 0, -1])
    height = [obs.get("base_height_m", 0.74)]
    contact = [float(c) for c in obs.get("foot_contact", [True, True])]
    cmd = obs.get("command", {}).get("locomotion", {})
    cmd_vel = cmd.get("target_velocity_base_m_s", [0, 0, 0])
    cmd_yaw = [cmd.get("target_yaw_rate_rad_s", 0.0)]

    flat = list(joint_pos) + list(joint_vel) + list(base_vel) + list(base_ang) + list(gravity) + height + contact + list(cmd_vel) + cmd_yaw
    return torch.tensor(flat, dtype=torch.float32, device=device)


def obs_dim_for_manifest(manifest: Any) -> int:
    n = manifest.active_joint_count
    return n + n + 3 + 3 + 3 + 1 + 2 + 3 + 1


def collect_rollout(
    model: ActorCritic,
    envs: list[Any],
    buffer: RolloutBuffer,
    horizon_steps: int,
    device: torch.device,
) -> dict[str, float]:
    total_reward = 0.0
    total_steps = 0
    episode_count = 0

    obs_list = [env.reset(seed=i) for i, env in enumerate(envs)]
    done_list = [False] * len(envs)

    for step in range(horizon_steps):
        obs_tensors = [obs_to_tensor(o, device) for o in obs_list]
        obs_batch = torch.stack(obs_tensors)

        with torch.no_grad():
            action, log_prob, value = model.act(obs_batch)

        buffer.observations.append(obs_batch)
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)

        action_np = action.cpu().numpy()
        rewards = torch.zeros(len(envs), device=device)
        dones = torch.zeros(len(envs), device=device)

        for i, env in enumerate(envs):
            if done_list[i]:
                obs_list[i] = env.reset()
                done_list[i] = False

            act_dict = {
                "joint_position_delta_rad": action_np[i].tolist(),
                "joint_velocity_delta_rad_s": [0.0] * env.manifest.active_joint_count,
                "feedforward_torque_nm": [0.0] * env.manifest.active_joint_count,
            }
            result = env.step(act_dict)
            rewards[i] = float(result.reward_debug.get("total", 0.0))
            total_reward += rewards[i].item()
            total_steps += 1

            if result.terminated or result.truncated:
                done_list[i] = True
                episode_count += 1
                obs_list[i] = env.reset()
            else:
                obs_list[i] = result.observation

        buffer.rewards.append(rewards)
        buffer.dones.append(dones.float())

    avg_reward = total_reward / max(total_steps, 1)
    return {
        "avg_reward": avg_reward,
        "total_steps": total_steps,
        "episodes": episode_count,
    }


def update_policy(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    advantages, returns = buffer.compute_gae(cfg.gamma, cfg.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = torch.cat(buffer.observations)
    actions = torch.cat(buffer.actions)
    old_log_probs = torch.cat(buffer.log_probs)
    old_values = torch.cat(buffer.values)

    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    update_count = 0

    dataset_size = len(obs)
    for _ in range(4):
        indices = torch.randperm(dataset_size, device=device)
        for start in range(0, dataset_size, cfg.mini_batch_size):
            end = min(start + cfg.mini_batch_size, dataset_size)
            mb_idx = indices[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages.view(-1)[mb_idx]
            mb_returns = returns.view(-1)[mb_idx]
            mb_old_values = old_values.view(-1)[mb_idx]

            new_log_probs, entropy, new_values = model.evaluate(mb_obs, mb_actions)

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            # Huber loss for value function (more robust to outliers than MSE)
            vf_loss = nn.functional.smooth_l1_loss(new_values, mb_returns)

            entropy_loss = -entropy.mean()

            loss = pg_loss + cfg.value_loss_coef * vf_loss + cfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.mean().item()
            update_count += 1

    return {
        "pg_loss": total_pg_loss / max(update_count, 1),
        "vf_loss": total_vf_loss / max(update_count, 1),
        "entropy": total_entropy / max(update_count, 1),
    }


def train(
    make_envs_fn,
    manifest: Any,
    cfg: PPOConfig,
    output_dir: Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = obs_dim_for_manifest(manifest)
    act_dim = manifest.active_joint_count

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    buffer = RolloutBuffer()

    log: list[dict[str, Any]] = []
    best_reward = float("-inf")

    for iteration in range(cfg.num_iterations):
        t0 = time.time()
        envs = make_envs_fn(cfg.num_envs)
        rollout_stats = collect_rollout(model, envs, buffer, cfg.horizon_steps, device)
        update_stats = update_policy(model, optimizer, buffer, cfg, device)
        buffer.clear()
        dt = time.time() - t0

        entry = {
            "iteration": iteration,
            "avg_reward": rollout_stats["avg_reward"],
            "episodes": rollout_stats["episodes"],
            "pg_loss": update_stats["pg_loss"],
            "vf_loss": update_stats["vf_loss"],
            "entropy": update_stats["entropy"],
            "time_s": dt,
        }
        log.append(entry)

        if (iteration + 1) % cfg.log_interval == 0:
            print(
                f"[{iteration+1}/{cfg.num_iterations}] "
                f"reward={rollout_stats['avg_reward']:.3f} "
                f"pg_loss={update_stats['pg_loss']:.4f} "
                f"vf_loss={update_stats['vf_loss']:.4f} "
                f"entropy={update_stats['entropy']:.3f} "
                f"time={dt:.1f}s"
            )

        if (iteration + 1) % cfg.save_interval == 0:
            ckpt_path = output_dir / f"model_{iteration+1}.pt"
            torch.save(model.state_dict(), ckpt_path)

        if rollout_stats["avg_reward"] > best_reward:
            best_reward = rollout_stats["avg_reward"]
            best_path = output_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)

    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    return {
        "best_reward": best_reward,
        "iterations": cfg.num_iterations,
        "output_dir": str(output_dir),
    }
