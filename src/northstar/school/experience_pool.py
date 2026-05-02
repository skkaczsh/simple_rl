"""School experience pool for collecting and managing training samples.

Collects samples during training rollouts, manages priority scoring,
and creates versioned datasets for future training.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from northstar.school.priority import score_event_priority


@dataclass
class TrainingSample:
    """A single training sample collected during rollout."""
    observation: dict[str, Any]
    action: list[float]
    reward: float
    done: bool
    info: dict[str, Any]
    # Episode context
    episode_id: str = ""
    step_index: int = 0
    # Quality metrics
    priority: float = 0.0
    event_type: str = "normal"
    segment_type: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "priority": self.priority,
            "event_type": self.event_type,
            "segment_type": self.segment_type,
        }


@dataclass
class EpisodeRecord:
    """Record of a complete episode."""
    episode_id: str
    total_reward: float = 0.0
    total_steps: int = 0
    terminated: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    samples: list[TrainingSample] = field(default_factory=list)
    # Quality metrics
    avg_priority: float = 0.0
    has_interesting_events: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "terminated": self.terminated,
            "events": self.events,
            "avg_priority": self.avg_priority,
            "has_interesting_events": self.has_interesting_events,
            "sample_count": len(self.samples),
        }


class SchoolExperiencePool:
    """Manages collection and storage of training samples.

    Collects samples during training rollouts, prioritizes them based on
    interesting events (falls, near-falls, tracking errors), and creates
    versioned datasets for future training.
    """

    def __init__(self, output_dir: Path, max_samples: int = 100000) -> None:
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.samples: list[TrainingSample] = []
        self.episodes: list[EpisodeRecord] = []
        self.current_episode: EpisodeRecord | None = None
        self.stats = {
            "total_samples": 0,
            "total_episodes": 0,
            "interesting_episodes": 0,
            "event_counts": {},
        }

    def start_episode(self, episode_id: str) -> None:
        """Start recording a new episode."""
        self.current_episode = EpisodeRecord(episode_id=episode_id)

    def add_step(
        self,
        observation: dict[str, Any],
        action: list[float],
        reward: float,
        done: bool,
        info: dict[str, Any],
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add a step to the current episode."""
        if self.current_episode is None:
            return

        step_idx = self.current_episode.total_steps

        # Determine event type and priority
        event_type = "normal"
        segment_type = "normal"
        priority = 0.1  # Base priority for normal steps

        if events:
            for event in events:
                event_type = event.get("event_type", "unknown")
                self.current_episode.events.append(event)

                # Map event to segment type
                if event_type in ("fall", "near_fall"):
                    segment_type = "near_failure"
                    priority = 0.8
                elif event_type == "action_clip":
                    segment_type = "action_clip"
                    priority = 0.6
                elif event_type == "tracking_error_high":
                    segment_type = "tracking_error_high"
                    priority = 0.5
                elif event_type == "perturbation":
                    segment_type = "perturbation"
                    priority = 0.5
                else:
                    segment_type = "event_injection"
                    priority = 0.4

                # Track event counts
                self.stats["event_counts"][event_type] = \
                    self.stats["event_counts"].get(event_type, 0) + 1

        # Create sample
        sample = TrainingSample(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            info=info,
            episode_id=self.current_episode.episode_id,
            step_index=step_idx,
            priority=priority,
            event_type=event_type,
            segment_type=segment_type,
        )

        self.current_episode.samples.append(sample)
        self.current_episode.total_reward += reward
        self.current_episode.total_steps += 1

        if done:
            self.current_episode.terminated = info.get("terminated", False)

    def end_episode(self) -> EpisodeRecord | None:
        """End the current episode and add it to the pool."""
        if self.current_episode is None:
            return None

        episode = self.current_episode
        self.current_episode = None

        # Compute episode-level metrics
        if episode.samples:
            episode.avg_priority = sum(s.priority for s in episode.samples) / len(episode.samples)

        episode.has_interesting_events = len(episode.events) > 0

        # Add samples to pool (with capacity management)
        for sample in episode.samples:
            if len(self.samples) >= self.max_samples:
                # Remove lowest priority sample
                self.samples.sort(key=lambda s: s.priority)
                if sample.priority > self.samples[0].priority:
                    self.samples[0] = sample
            else:
                self.samples.append(sample)

        self.episodes.append(episode)
        self.stats["total_episodes"] += 1
        self.stats["total_samples"] += len(episode.samples)
        if episode.has_interesting_events:
            self.stats["interesting_episodes"] += 1

        return episode

    def get_top_samples(self, n: int = 1000) -> list[TrainingSample]:
        """Get the top N samples by priority."""
        return sorted(self.samples, key=lambda s: s.priority, reverse=True)[:n]

    def get_interesting_episodes(self) -> list[EpisodeRecord]:
        """Get episodes with interesting events."""
        return [e for e in self.episodes if e.has_interesting_events]

    def save(self, version: str = "v0") -> Path:
        """Save the experience pool to disk."""
        pool_dir = self.output_dir / f"school_pool_{version}"
        pool_dir.mkdir(parents=True, exist_ok=True)

        # Save samples
        samples_path = pool_dir / "samples.jsonl"
        with samples_path.open("w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        # Save episodes
        episodes_path = pool_dir / "episodes.jsonl"
        with episodes_path.open("w") as f:
            for episode in self.episodes:
                f.write(json.dumps(episode.to_dict()) + "\n")

        # Save stats
        stats_path = pool_dir / "stats.json"
        stats_path.write_text(json.dumps(self.stats, indent=2))

        # Save metadata
        metadata = {
            "version": version,
            "timestamp": time.time(),
            "total_samples": len(self.samples),
            "total_episodes": len(self.episodes),
            "interesting_episodes": self.stats["interesting_episodes"],
            "event_counts": self.stats["event_counts"],
        }
        meta_path = pool_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        return pool_dir

    def load(self, version: str = "v0") -> None:
        """Load experience pool from disk."""
        pool_dir = self.output_dir / f"school_pool_{version}"

        # Load samples
        samples_path = pool_dir / "samples.jsonl"
        if samples_path.exists():
            self.samples = []
            with samples_path.open("r") as f:
                for line in f:
                    data = json.loads(line)
                    self.samples.append(TrainingSample(**data))

        # Load episodes
        episodes_path = pool_dir / "episodes.jsonl"
        if episodes_path.exists():
            self.episodes = []
            with episodes_path.open("r") as f:
                for line in f:
                    data = json.loads(line)
                    self.episodes.append(EpisodeRecord(**data))

        # Load stats
        stats_path = pool_dir / "stats.json"
        if stats_path.exists():
            self.stats = json.loads(stats_path.read_text())

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the experience pool."""
        return {
            "total_samples": len(self.samples),
            "total_episodes": len(self.episodes),
            "interesting_episodes": self.stats["interesting_episodes"],
            "event_counts": self.stats["event_counts"],
            "avg_priority": sum(s.priority for s in self.samples) / max(len(self.samples), 1),
            "priority_distribution": {
                "high": len([s for s in self.samples if s.priority >= 0.6]),
                "medium": len([s for s in self.samples if 0.3 <= s.priority < 0.6]),
                "low": len([s for s in self.samples if s.priority < 0.3]),
            },
        }
