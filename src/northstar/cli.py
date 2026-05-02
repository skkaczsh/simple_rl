from __future__ import annotations

import argparse
import json
from pathlib import Path

from northstar.command.scenarios import load_scenario_set
from northstar.embodiment.manifest import load_manifest
from northstar.episode_io.replay_reader import replay_episode
from northstar.eval.runner import run_scenario_set
from northstar.school.sample_builder import build_sample_for_event


def validate_abi_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/embodiment/unitree_g1_43dof_sim_v0.json")
    parser.add_argument("--scenario-set", default="configs/eval/phase1_skeleton_scenarios.yaml")
    args = parser.parse_args()
    manifest = load_manifest(Path(args.manifest))
    scenario_set = load_scenario_set(Path(args.scenario_set))
    print(json.dumps({"manifest": manifest.embodiment_id, "scenario_set": scenario_set["scenario_set_id"], "pass": True}, sort_keys=True))


def run_eval_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", required=True)
    parser.add_argument("--manifest", default="configs/embodiment/unitree_g1_43dof_sim_v0.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_scenario_set(Path(args.scenario_set), Path(args.manifest), Path(args.output))
    print(json.dumps(report["summary"], sort_keys=True))


def replay_episode_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--manifest", default="configs/embodiment/unitree_g1_43dof_sim_v0.json")
    args = parser.parse_args()
    manifest = load_manifest(Path(args.manifest))
    metrics = replay_episode(Path(args.episode), manifest.default_base_height_m)
    print(json.dumps(metrics, sort_keys=True))


def build_school_samples_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    sample_count = 0
    for events_path in run_dir.glob("episodes/*/events.jsonl"):
        episode_id = events_path.parent.name
        samples = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["event_type"] in {"episode_start", "episode_end"}:
                continue
            samples.append(build_sample_for_event(event, run_dir.name, episode_id, str(events_path.parent), True, True))
            sample_count += 1
        if samples:
            (events_path.parent / "school_samples.jsonl").write_text(
                "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
                encoding="utf-8",
            )
    print(json.dumps({"sample_count": sample_count}, sort_keys=True))
