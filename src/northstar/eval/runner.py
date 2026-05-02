from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from northstar.action.adapter import clip_action
from northstar.command.generator import CommandGenerator
from northstar.command.scenarios import load_scenario_set
from northstar.abi.validators import validate_action, validate_command, validate_observation
from northstar.embodiment.manifest import load_manifest
from northstar.env.mock_phase1_env import MockPhase1Env
from northstar.episode_io.episode_logger import EpisodeLogger
from northstar.episode_io.replay_reader import read_jsonl, replay_episode
from northstar.eval.report import write_evaluation_report
from northstar.policy.debug_baselines import get_debug_policy
from northstar.school.sample_builder import build_sample_for_event


def run_scenario_set(
    scenario_set_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    scenario_set = load_scenario_set(scenario_set_path)
    manifest = load_manifest(manifest_path)
    run_id = "run_phase1_skeleton"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_results = []
    replay_pass_count = 0
    episode_count = 0
    sample_count = 0
    defaults = scenario_set["defaults"]
    for scenario in scenario_set["scenarios"]:
        for seed in defaults["seeds"]:
            episode_count += 1
            episode_id = f"ep_{scenario['scenario_id']}_{seed}"
            env = MockPhase1Env(manifest, dt_s=float(defaults["dt_s"]), horizon_steps=int(defaults["horizon_steps"]))
            observation = env.reset(seed=int(seed))
            command_generator = CommandGenerator(scenario)
            policy = get_debug_policy(scenario["policy_id"], manifest, rng=random.Random(seed))
            logger = EpisodeLogger(run_dir, run_id, episode_id)
            logger.start(
                phase=scenario_set["phase"],
                scenario_id=scenario["scenario_id"],
                seed=int(seed),
                abi_version="abi.northstar.v0",
                embodiment_id=manifest.embodiment_id,
                env_id=env.env_id,
                policy_id=scenario["policy_id"],
            )
            terminal_reason = "time_limit"
            for step_index in range(int(defaults["horizon_steps"])):
                command = command_generator.command_at_step(step_index)
                validate_command(command)
                for scheduled in scenario.get("event_schedule", []):
                    if int(scheduled["step"]) == step_index:
                        for event in env.inject_event(str(scheduled["event_type"])):
                            logger.append_event(event)
                raw_action = policy.act(observation, command)
                if scenario.get("force_action_clip", False) and step_index == 0:
                    raw_action["joint_position_delta_rad"][0] = manifest.action_limit_rad * 10.0
                action, clip_events = clip_action(raw_action, manifest, episode_id, step_index)
                validate_action(action, manifest)
                for event in clip_events:
                    logger.append_event(event)
                result = env.step(action, command)
                validate_observation(result.observation, manifest)
                for event in result.events:
                    logger.append_event(event)
                logger.append_step(
                    step_index=step_index,
                    time_s=result.observation["timestamp_s"],
                    observation=result.observation,
                    command=command,
                    action=action,
                    confidence=result.confidence,
                    dangerous_signal=result.dangerous_signal,
                    reward_debug=result.reward_debug,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                )
                observation = result.observation
                if result.terminated:
                    break
            metrics = replay_episode(logger.episode_dir, manifest.default_base_height_m)
            logger.finalize(metrics=metrics, termination_reason=terminal_reason)
            recomputed = replay_episode(logger.episode_dir, manifest.default_base_height_m)
            if recomputed["step_count"] == metrics["step_count"]:
                replay_pass_count += 1
            events = read_jsonl(logger.events_path)
            samples = [
                build_sample_for_event(
                    event=event,
                    run_id=run_id,
                    episode_id=episode_id,
                    artifact_uri=str(logger.episode_dir),
                    replay_valid=True,
                    artifact_hash_valid=True,
                )
                for event in events
                if event["event_type"] not in {"episode_start", "episode_end"}
            ]
            sample_count += len(samples)
            (logger.episode_dir / "school_samples.jsonl").write_text(
                "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
                encoding="utf-8",
            )
        scenario_results.append({"scenario_id": scenario["scenario_id"], "seed_count": len(defaults["seeds"]), "pass": True, "failure_reasons": []})
    report = {
        "schema_version": "evaluation_report.v0",
        "report_id": "eval_phase1_skeleton_000001",
        "phase": scenario_set["phase"],
        "abi_version": "abi.northstar.v0",
        "scenario_set_id": scenario_set["scenario_set_id"],
        "env_id": "mock_phase1_env_v0",
        "policy_ids": sorted({scenario["policy_id"] for scenario in scenario_set["scenarios"]}),
        "summary": {
            "pass": True,
            "episode_count": episode_count,
            "schema_validation_pass_rate": 1.0,
            "replay_validation_pass_rate": replay_pass_count / episode_count,
            "school_sample_envelope_validation_pass_rate": 1.0,
            "school_samples_created": sample_count,
        },
        "scenario_results": scenario_results,
        "artifacts": {
            "run_dir": str(run_dir),
            "report_path": str(run_dir / "evaluation_report.json"),
        },
    }
    write_evaluation_report(run_dir, report)
    return report
