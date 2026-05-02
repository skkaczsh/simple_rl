from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from northstar.episode_io.artifact_hash import sha256_file


class EpisodeLogger:
    def __init__(self, run_dir: Path, run_id: str, episode_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.episode_id = episode_id
        self.episode_dir = run_dir / "episodes" / episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.episode_dir / "steps.jsonl"
        self.events_path = self.episode_dir / "events.jsonl"
        self.metrics_path = self.episode_dir / "metrics.json"
        self.manifest_path = self.episode_dir / "episode_manifest.json"
        self._manifest: dict[str, Any] = {}
        self._step_count = 0

    def start(
        self,
        phase: str,
        scenario_id: str,
        seed: int,
        abi_version: str,
        embodiment_id: str,
        env_id: str,
        policy_id: str,
    ) -> None:
        self._manifest = {
            "schema_version": "episode_manifest.v0",
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "phase": phase,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "abi_version": abi_version,
            "embodiment_id": embodiment_id,
            "env_id": env_id,
            "policy_id": policy_id,
            "started_at": "1970-01-01T00:00:00+00:00",
            "ended_at": None,
            "step_count": 0,
            "termination_reason": None,
            "artifact_hashes": {},
        }
        self.steps_path.write_text("", encoding="utf-8")
        self.events_path.write_text("", encoding="utf-8")

    def append_step(
        self,
        step_index: int,
        time_s: float,
        observation: dict[str, Any],
        command: dict[str, Any],
        action: dict[str, Any],
        confidence: dict[str, Any],
        dangerous_signal: dict[str, Any],
        reward_debug: dict[str, float],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        record = {
            "schema_version": "step_record.v0",
            "episode_id": self.episode_id,
            "step_index": int(step_index),
            "time_s": float(time_s),
            "observation": observation,
            "command": command,
            "action": action,
            "confidence": confidence,
            "dangerous_signal": dangerous_signal,
            "reward_debug": reward_debug,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._step_count += 1

    def append_event(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["episode_id"] = self.episode_id
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def finalize(self, metrics: dict[str, Any], termination_reason: str) -> None:
        self.metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8")
        self._manifest["ended_at"] = "1970-01-01T00:00:00+00:00"
        self._manifest["step_count"] = self._step_count
        self._manifest["termination_reason"] = termination_reason
        self._manifest["artifact_hashes"] = {
            "steps_jsonl": sha256_file(self.steps_path),
            "events_jsonl": sha256_file(self.events_path),
            "metrics_json": sha256_file(self.metrics_path),
        }
        self.manifest_path.write_text(json.dumps(self._manifest, sort_keys=True, indent=2), encoding="utf-8")
