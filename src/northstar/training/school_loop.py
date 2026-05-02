"""School-integrated training loop.

Extends PPO training with school sample collection during rollouts.
Collects interesting episodes (falls, near-falls, tracking errors) into
an experience pool for future training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from northstar.school.experience_pool import SchoolExperiencePool
from northstar.training.ppo import (
    ActorCritic,
    PPOConfig,
    RolloutBuffer,
    collect_rollout,
    obs_dim_for_manifest,
    obs_to_tensor,
    update_policy,
)


def collect_rollout_with_school(
    model: ActorCritic,
    envs: list[Any],
    buffer: RolloutBuffer,
    horizon_steps: int,
    device: torch.device,
    school_pool: SchoolExperiencePool | None = None,
    collect_ratio: float = 0.1,
) -> dict[str, float]:
    """Collect rollout with optional school sample collection.

    Args:
        model: ActorCritic model
        envs: List of environments
        buffer: Rollout buffer
        horizon_steps: Steps per rollout
        device: Torch device
        school_pool: Optional school experience pool
        collect_ratio: Fraction of episodes to collect samples from
    """
    total_reward = 0.0
    total_steps = 0
    episode_count = 0

    obs_list = [env.reset(seed=i) for i, env in enumerate(envs)]
    done_list = [False] * len(envs)

    # Track per-env episode state for school collection
    env_episode_ids = [f"ep_{i}" for i in range(len(envs))]
    collecting = [False] * len(envs)

    for step in range(horizon_steps):
        obs_tensors = [obs_to_tensor(o, device) for o in obs_list]
        obs_batch = torch.stack(obs_tensors)

        # Check for NaN in observations
        if torch.isnan(obs_batch).any():
            obs_batch = torch.nan_to_num(obs_batch, nan=0.0)

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
                # End previous episode in school pool
                if school_pool and collecting[i]:
                    school_pool.end_episode()

                obs_list[i] = env.reset()
                done_list[i] = False
                env_episode_ids[i] = f"ep_{episode_count}_{i}"

                # Decide whether to collect this episode
                collecting[i] = school_pool is not None and (episode_count % max(1, int(1/collect_ratio)) == 0)

                if collecting[i]:
                    school_pool.start_episode(env_episode_ids[i])

                episode_count += 1

            act_dict = {
                "joint_position_delta_rad": action_np[i].tolist(),
                "joint_velocity_delta_rad_s": [0.0] * env.manifest.active_joint_count,
                "feedforward_torque_nm": [0.0] * env.manifest.active_joint_count,
            }
            result = env.step(act_dict)
            rewards[i] = float(result.reward_debug.get("total", 0.0))
            total_reward += rewards[i].item()
            total_steps += 1

            # Collect school sample
            if collecting[i] and school_pool:
                school_pool.add_step(
                    observation=result.observation,
                    action=action_np[i].tolist(),
                    reward=rewards[i].item(),
                    done=result.terminated or result.truncated,
                    info={"terminated": result.terminated, "truncated": result.truncated},
                    events=result.events,
                )

            if result.terminated or result.truncated:
                done_list[i] = True
                obs_list[i] = env.reset()
            else:
                obs_list[i] = result.observation

        buffer.rewards.append(rewards)
        buffer.dones.append(dones.float())

    # End any remaining episodes
    if school_pool:
        for i in range(len(envs)):
            if collecting[i]:
                school_pool.end_episode()

    avg_reward = total_reward / max(total_steps, 1)
    return {
        "avg_reward": avg_reward,
        "total_steps": total_steps,
        "episodes": episode_count,
    }


def train_with_school(
    make_envs_fn,
    manifest: Any,
    cfg: PPOConfig,
    output_dir: Path,
    school_pool: SchoolExperiencePool | None = None,
    collect_ratio: float = 0.1,
    save_pool_interval: int = 50,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train with school sample collection.

    Args:
        make_envs_fn: Function to create environments
        manifest: Embodiment manifest
        cfg: PPO config
        output_dir: Output directory
        school_pool: Optional school experience pool
        collect_ratio: Fraction of episodes to collect
        save_pool_interval: How often to save the pool
        device: Torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = obs_dim_for_manifest(manifest)
    act_dim = manifest.active_joint_count

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    buffer = RolloutBuffer()

    log: list[dict[str, Any]] = []
    best_reward = float("-inf")

    for iteration in range(cfg.num_iterations):
        t0 = time.time()
        envs = make_envs_fn(cfg.num_envs)

        # Use school-integrated rollout collection
        rollout_stats = collect_rollout_with_school(
            model, envs, buffer, cfg.horizon_steps, device,
            school_pool=school_pool,
            collect_ratio=collect_ratio,
        )

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
            school_info = ""
            if school_pool:
                summary = school_pool.get_summary()
                school_info = f" school_samples={summary['total_samples']} interesting={summary['interesting_episodes']}"
            print(
                f"[{iteration+1}/{cfg.num_iterations}] "
                f"reward={rollout_stats['avg_reward']:.3f} "
                f"pg_loss={update_stats['pg_loss']:.4f} "
                f"vf_loss={update_stats['vf_loss']:.4f} "
                f"entropy={update_stats['entropy']:.3f} "
                f"time={dt:.1f}s{school_info}"
            )

        if (iteration + 1) % cfg.save_interval == 0:
            ckpt_path = output_dir / f"model_{iteration+1}.pt"
            torch.save(model.state_dict(), ckpt_path)

        if rollout_stats["avg_reward"] > best_reward:
            best_reward = rollout_stats["avg_reward"]
            best_path = output_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)

        # Save school pool periodically
        if school_pool and (iteration + 1) % save_pool_interval == 0:
            pool_dir = school_pool.save(version=f"iter_{iteration+1}")
            print(f"  School pool saved: {pool_dir}")

    # Save final training log
    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    # Save final school pool
    if school_pool:
        pool_dir = school_pool.save(version="final")
        print(f"Final school pool saved: {pool_dir}")

        # Save summary
        summary = school_pool.get_summary()
        summary_path = output_dir / "school_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "best_reward": best_reward,
        "iterations": cfg.num_iterations,
        "output_dir": str(output_dir),
        "school_samples": school_pool.get_summary()["total_samples"] if school_pool else 0,
    }
