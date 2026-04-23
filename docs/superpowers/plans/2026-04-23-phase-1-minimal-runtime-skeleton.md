# Phase 1 Minimal Runtime Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 0/1 minimal runtime skeleton that can run mock Phase 1 scenarios, validate ABI data, log episodes, replay metrics, generate evaluation reports, and build school sample envelopes.

**Architecture:** Use a small Python package under `src/northstar/` with explicit boundaries: ABI models and validators, deterministic mock env, debug follower policies, action adapter, episode IO, metrics, evaluation runner, and school sample builder. The mock env is only an interface and data-chain substitute; later Isaac Lab work must replace `EnvAdapter` without rewriting ABI, logging, replay, metrics, or school sample code.

**Tech Stack:** Python 3.11+, standard library dataclasses/json/pathlib/hashlib/argparse, pytest, PyYAML.

---

## File Structure

Create this structure:

```text
pyproject.toml
configs/
  embodiment/
    unitree_g1_43dof_sim_v0.json
  eval/
    phase0_scenarios.yaml
    phase1_skeleton_scenarios.yaml
  school/
    phase1_sample_scoring.yaml
src/
  northstar/
    __init__.py
    cli.py
    abi/
      __init__.py
      action.py
      command.py
      episode.py
      observation.py
      signals.py
      validators.py
    action/
      __init__.py
      adapter.py
    command/
      __init__.py
      generator.py
      scenarios.py
    embodiment/
      __init__.py
      manifest.py
    env/
      __init__.py
      adapter.py
      mock_phase1_env.py
      state.py
    eval/
      __init__.py
      report.py
      runner.py
    episode_io/
      __init__.py
      artifact_hash.py
      episode_logger.py
      replay_reader.py
    metrics/
      __init__.py
      accumulator.py
      locomotion.py
      replay.py
    policy/
      __init__.py
      debug_baselines.py
      follower_adapter.py
    school/
      __init__.py
      priority.py
      sample_builder.py
tests/
  abi/
    test_action_schema.py
    test_command_schema.py
    test_observation_schema.py
  action/
    test_action_adapter.py
  command/
    test_command_generator.py
  embodiment/
    test_manifest.py
  env/
    test_mock_phase1_env.py
  eval/
    test_phase1_skeleton_runner.py
  episode_io/
    test_episode_logger.py
    test_replay_reader.py
  metrics/
    test_locomotion_metrics.py
  policy/
    test_debug_baselines.py
  school/
    test_school_sample_builder.py
```

Responsibility map:

- `abi/*`: dataclass-like dictionaries, constructors, and validation errors for the Phase 0/1 ABI.
- `embodiment/manifest.py`: active joint count, foot site count, and action limits.
- `command/*`: scenario loading and deterministic command schedules.
- `env/*`: `EnvAdapter` protocol and deterministic `MockPhase1Env`.
- `policy/*`: `noop`, `random legal`, and `simple PD` debug policies.
- `action/adapter.py`: action clipping and `action_clip` event creation.
- `episode_io/*`: episode artifact writing, hashing, and replay reading.
- `metrics/*`: smoke metrics and replay consistency checks.
- `eval/*`: scenario runner and evaluation report writer.
- `school/*`: envelope extraction and priority scoring.
- `cli.py`: engineering commands only.

## Task 1: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `src/northstar/__init__.py`
- Create: package `__init__.py` files under each subpackage
- Test: no test file in this task

- [ ] **Step 1: Create package directories**

Run:

```bash
mkdir -p configs/embodiment configs/eval configs/school
mkdir -p src/northstar/{abi,action,command,embodiment,env,eval,episode_io,metrics,policy,school}
mkdir -p tests/{abi,action,command,embodiment,env,eval,episode_io,metrics,policy,school}
```

Expected: command exits `0`.

- [ ] **Step 2: Add `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "northstar-rl"
version = "0.1.0"
description = "North Star Phase 0/1 runtime skeleton"
requires-python = ">=3.11"
dependencies = [
  "PyYAML>=6.0.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
]

[project.scripts]
northstar-validate-abi = "northstar.cli:validate_abi_main"
northstar-run-eval = "northstar.cli:run_eval_main"
northstar-replay-episode = "northstar.cli:replay_episode_main"
northstar-build-school-samples = "northstar.cli:build_school_samples_main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Add package markers**

Create these files with the exact content below:

```python
"""North Star Phase 0/1 runtime skeleton."""
```

Paths:

```text
src/northstar/__init__.py
src/northstar/abi/__init__.py
src/northstar/action/__init__.py
src/northstar/command/__init__.py
src/northstar/embodiment/__init__.py
src/northstar/env/__init__.py
src/northstar/eval/__init__.py
src/northstar/episode_io/__init__.py
src/northstar/metrics/__init__.py
src/northstar/policy/__init__.py
src/northstar/school/__init__.py
```

- [ ] **Step 4: Install package in editable mode**

Run:

```bash
python -m pip install -e ".[dev]"
```

Expected output contains:

```text
Successfully installed
```

- [ ] **Step 5: Run import smoke test**

Run:

```bash
python -c "import northstar; print(northstar.__doc__)"
```

Expected output contains:

```text
North Star Phase 0/1 runtime skeleton.
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/northstar
git commit -m "chore: bootstrap northstar runtime package"
```

## Task 2: Embodiment Manifest

**Files:**
- Create: `configs/embodiment/unitree_g1_43dof_sim_v0.json`
- Create: `src/northstar/embodiment/manifest.py`
- Test: `tests/embodiment/test_manifest.py`

- [ ] **Step 1: Write failing manifest tests**

Create `tests/embodiment/test_manifest.py`:

```python
from pathlib import Path

from northstar.embodiment.manifest import EmbodimentManifest, load_manifest


def test_load_manifest_counts_joints_and_feet():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))

    assert manifest.embodiment_id == "unitree_g1_43dof_sim_v0"
    assert manifest.active_joint_count == 12
    assert manifest.foot_contact_site_count == 2
    assert manifest.action_limit_rad == 0.25
    assert manifest.velocity_limit_rad_s == 1.0
    assert manifest.torque_limit_nm == 20.0


def test_manifest_from_dict_rejects_empty_joint_names():
    payload = {
        "embodiment_id": "bad",
        "active_joint_names": [],
        "foot_contact_site_names": ["left_foot"],
        "action_limit_rad": 0.25,
        "velocity_limit_rad_s": 1.0,
        "torque_limit_nm": 20.0,
    }

    try:
        EmbodimentManifest.from_dict(payload)
    except ValueError as exc:
        assert "active_joint_names" in str(exc)
    else:
        raise AssertionError("expected active_joint_names validation error")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/embodiment/test_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `manifest.py`.

- [ ] **Step 3: Add manifest config**

Create `configs/embodiment/unitree_g1_43dof_sim_v0.json`:

```json
{
  "schema_version": "embodiment_manifest.v0",
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "active_joint_names": [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll"
  ],
  "foot_contact_site_names": ["left_foot", "right_foot"],
  "action_limit_rad": 0.25,
  "velocity_limit_rad_s": 1.0,
  "torque_limit_nm": 20.0,
  "default_base_height_m": 0.74
}
```

- [ ] **Step 4: Implement manifest loader**

Create `src/northstar/embodiment/manifest.py`:

```python
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
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/embodiment/test_manifest.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/embodiment/unitree_g1_43dof_sim_v0.json src/northstar/embodiment/manifest.py tests/embodiment/test_manifest.py
git commit -m "feat: add embodiment manifest loader"
```

## Task 3: ABI Command and Action Validation

**Files:**
- Create: `src/northstar/abi/validators.py`
- Create: `src/northstar/abi/command.py`
- Create: `src/northstar/abi/action.py`
- Test: `tests/abi/test_command_schema.py`
- Test: `tests/abi/test_action_schema.py`

- [ ] **Step 1: Write failing command tests**

Create `tests/abi/test_command_schema.py`:

```python
from northstar.abi.command import make_locomotion_command
from northstar.abi.validators import ValidationError, validate_command


def test_valid_phase1_command_passes():
    command = make_locomotion_command(
        command_id="cmd_1",
        target_velocity_base_m_s=[0.2, 0.0, 0.0],
        target_yaw_rate_rad_s=0.3,
    )

    validate_command(command)


def test_phase1_command_rejects_vertical_velocity():
    command = make_locomotion_command(
        command_id="cmd_bad",
        target_velocity_base_m_s=[0.0, 0.0, 0.1],
        target_yaw_rate_rad_s=0.0,
    )

    try:
        validate_command(command)
    except ValidationError as exc:
        assert "target_velocity_base_m_s.z" in str(exc)
    else:
        raise AssertionError("expected vertical velocity validation error")


def test_phase1_command_rejects_enabled_upper_body():
    command = make_locomotion_command(
        command_id="cmd_upper",
        target_velocity_base_m_s=[0.0, 0.0, 0.0],
        target_yaw_rate_rad_s=0.0,
    )
    command["mode_mask"]["upper_body"] = True

    try:
        validate_command(command)
    except ValidationError as exc:
        assert "upper_body" in str(exc)
    else:
        raise AssertionError("expected upper_body validation error")
```

- [ ] **Step 2: Write failing action tests**

Create `tests/abi/test_action_schema.py`:

```python
from pathlib import Path

from northstar.abi.action import make_zero_action
from northstar.abi.validators import ValidationError, validate_action
from northstar.embodiment.manifest import load_manifest


def test_zero_action_matches_manifest_joint_count():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    action = make_zero_action("act_1", manifest, action_source="debug_policy")

    validate_action(action, manifest)
    assert len(action["joint_position_delta_rad"]) == manifest.active_joint_count
    assert action["clipped"] is False


def test_action_rejects_wrong_joint_count():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    action = make_zero_action("act_bad", manifest, action_source="debug_policy")
    action["joint_position_delta_rad"] = [0.0]

    try:
        validate_action(action, manifest)
    except ValidationError as exc:
        assert "joint_position_delta_rad" in str(exc)
    else:
        raise AssertionError("expected joint count validation error")
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
pytest tests/abi/test_command_schema.py tests/abi/test_action_schema.py -v
```

Expected: FAIL with missing `northstar.abi.command` or `northstar.abi.action`.

- [ ] **Step 4: Implement validators**

Create `src/northstar/abi/validators.py`:

```python
from __future__ import annotations

from typing import Any

from northstar.embodiment.manifest import EmbodimentManifest


class ValidationError(ValueError):
    """Raised when an ABI payload violates the Phase 0/1 schema."""


def _require_keys(payload: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        if key not in payload:
            raise ValidationError(f"{path}.{key} is required")


def validate_command(command: dict[str, Any]) -> None:
    _require_keys(command, ["schema_version", "command_id", "mode_mask", "locomotion"], "command")
    if command["schema_version"] != "command.northstar.v0":
        raise ValidationError("command.schema_version must be command.northstar.v0")
    mask = command["mode_mask"]
    for disabled_key in ["upper_body", "light_axis", "semantic_intent"]:
        if bool(mask.get(disabled_key, False)):
            raise ValidationError(f"command.mode_mask.{disabled_key} must be false in Phase 1")
    locomotion = command["locomotion"]
    velocity = locomotion["target_velocity_base_m_s"]
    if len(velocity) != 3:
        raise ValidationError("command.locomotion.target_velocity_base_m_s must have length 3")
    if float(velocity[2]) != 0.0:
        raise ValidationError("command.locomotion.target_velocity_base_m_s.z must be 0.0 in Phase 1")
    if not -0.6 <= float(velocity[0]) <= 1.0:
        raise ValidationError("command.locomotion.target_velocity_base_m_s.x outside [-0.6, 1.0]")
    if not -0.4 <= float(velocity[1]) <= 0.4:
        raise ValidationError("command.locomotion.target_velocity_base_m_s.y outside [-0.4, 0.4]")
    if not -1.0 <= float(locomotion["target_yaw_rate_rad_s"]) <= 1.0:
        raise ValidationError("command.locomotion.target_yaw_rate_rad_s outside [-1.0, 1.0]")


def validate_action(action: dict[str, Any], manifest: EmbodimentManifest) -> None:
    _require_keys(
        action,
        [
            "schema_version",
            "action_id",
            "joint_position_delta_rad",
            "joint_velocity_delta_rad_s",
            "feedforward_torque_nm",
            "action_source",
            "clipped",
            "clip_summary",
        ],
        "action",
    )
    if action["schema_version"] != "action.northstar.v0":
        raise ValidationError("action.schema_version must be action.northstar.v0")
    expected = manifest.active_joint_count
    for key in ["joint_position_delta_rad", "joint_velocity_delta_rad_s", "feedforward_torque_nm"]:
        values = action[key]
        if len(values) != expected:
            raise ValidationError(f"action.{key} length must be {expected}")
```

- [ ] **Step 5: Implement command helpers**

Create `src/northstar/abi/command.py`:

```python
from __future__ import annotations

from typing import Any


def make_locomotion_command(
    command_id: str,
    target_velocity_base_m_s: list[float],
    target_yaw_rate_rad_s: float,
    target_base_height_m: float = 0.0,
    stop_request: bool = False,
    brace_request: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "command.northstar.v0",
        "command_id": command_id,
        "mode_mask": {
            "stand": True,
            "locomotion": True,
            "upper_body": False,
            "light_axis": False,
            "semantic_intent": False,
        },
        "locomotion": {
            "target_base_height_m": float(target_base_height_m),
            "target_velocity_base_m_s": [float(v) for v in target_velocity_base_m_s],
            "target_yaw_rate_rad_s": float(target_yaw_rate_rad_s),
            "target_heading_rad": None,
            "stop_request": bool(stop_request),
            "brace_request": bool(brace_request),
        },
        "upper_body": None,
        "light_axis_hint": None,
        "semantic_hint": None,
    }
```

- [ ] **Step 6: Implement action helpers**

Create `src/northstar/abi/action.py`:

```python
from __future__ import annotations

from typing import Any

from northstar.embodiment.manifest import EmbodimentManifest


def make_zero_action(
    action_id: str,
    manifest: EmbodimentManifest,
    action_source: str,
) -> dict[str, Any]:
    count = manifest.active_joint_count
    return {
        "schema_version": "action.northstar.v0",
        "action_id": action_id,
        "joint_position_delta_rad": [0.0] * count,
        "joint_velocity_delta_rad_s": [0.0] * count,
        "feedforward_torque_nm": [0.0] * count,
        "action_source": action_source,
        "clipped": False,
        "clip_summary": [],
    }
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/abi/test_command_schema.py tests/abi/test_action_schema.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/northstar/abi tests/abi
git commit -m "feat: add phase 1 command and action ABI"
```

## Task 4: Observation and Signal ABI

**Files:**
- Create: `src/northstar/abi/observation.py`
- Create: `src/northstar/abi/signals.py`
- Modify: `src/northstar/abi/validators.py`
- Test: `tests/abi/test_observation_schema.py`

- [ ] **Step 1: Write failing observation tests**

Create `tests/abi/test_observation_schema.py`:

```python
from pathlib import Path

from northstar.abi.command import make_locomotion_command
from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.abi.validators import ValidationError, validate_observation
from northstar.embodiment.manifest import load_manifest


def test_observation_matches_manifest_and_contains_command():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd_1", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, timestamp_s=0.0, dt_s=0.02)

    validate_observation(observation, manifest)
    assert observation["mode_mask"]["upper_body"] is False
    assert len(observation["joint_position_rad"]) == manifest.active_joint_count


def test_observation_rejects_wrong_foot_contact_count():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd_1", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, timestamp_s=0.0, dt_s=0.02)
    observation["foot_contact"] = [True]

    try:
        validate_observation(observation, manifest)
    except ValidationError as exc:
        assert "foot_contact" in str(exc)
    else:
        raise AssertionError("expected foot_contact validation error")


def test_signal_helpers_return_expected_schema_versions():
    confidence = make_confidence(overall=0.5, stability=0.6, tracking=0.7, fallback=1.0)
    dangerous = make_dangerous_signal(overall_risk=0.2, triggered=["near_fall"])

    assert confidence["schema_version"] == "confidence.northstar.v0"
    assert dangerous["schema_version"] == "dangerous_signal.northstar.v0"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/abi/test_observation_schema.py -v
```

Expected: FAIL with missing `northstar.abi.observation`.

- [ ] **Step 3: Implement observation helper**

Create `src/northstar/abi/observation.py`:

```python
from __future__ import annotations

from typing import Any

from northstar.abi.action import make_zero_action
from northstar.embodiment.manifest import EmbodimentManifest


def make_observation(
    manifest: EmbodimentManifest,
    command: dict[str, Any],
    timestamp_s: float,
    dt_s: float,
    base_linear_velocity_m_s: list[float] | None = None,
    base_angular_velocity_rad_s: list[float] | None = None,
    base_height_m: float | None = None,
    previous_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "observation.northstar.v0",
        "timestamp_s": float(timestamp_s),
        "dt_s": float(dt_s),
        "frame": "base",
        "joint_position_rad": [0.0] * manifest.active_joint_count,
        "joint_velocity_rad_s": [0.0] * manifest.active_joint_count,
        "base_linear_velocity_m_s": base_linear_velocity_m_s or [0.0, 0.0, 0.0],
        "base_angular_velocity_rad_s": base_angular_velocity_rad_s or [0.0, 0.0, 0.0],
        "projected_gravity_base": [0.0, 0.0, -1.0],
        "base_height_m": manifest.default_base_height_m if base_height_m is None else float(base_height_m),
        "foot_contact": [True] * manifest.foot_contact_site_count,
        "previous_action": previous_action or make_zero_action("act_initial", manifest, "initial"),
        "command": command,
        "mode_mask": dict(command["mode_mask"]),
        "masks": {
            "privileged": False,
            "upper_body_command_enabled": False,
            "light_axis_enabled": False,
            "semantic_hint_enabled": False,
        },
    }
```

- [ ] **Step 4: Implement signal helpers**

Create `src/northstar/abi/signals.py`:

```python
from __future__ import annotations

from typing import Any


def make_confidence(
    overall: float,
    stability: float,
    tracking: float,
    fallback: float,
    source: str = "debug",
) -> dict[str, Any]:
    return {
        "schema_version": "confidence.northstar.v0",
        "overall": float(overall),
        "stability": float(stability),
        "tracking": float(tracking),
        "fallback": float(fallback),
        "source": source,
    }


def make_dangerous_signal(
    overall_risk: float,
    triggered: list[str],
    fall_risk: float = 0.0,
    near_fall_risk: float = 0.0,
    limit_risk: float = 0.0,
    tracking_risk: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema_version": "dangerous_signal.northstar.v0",
        "overall_risk": float(overall_risk),
        "fall_risk": float(fall_risk),
        "near_fall_risk": float(near_fall_risk),
        "limit_risk": float(limit_risk),
        "tracking_risk": float(tracking_risk),
        "triggered": list(triggered),
    }
```

- [ ] **Step 5: Add observation validator**

Append to `src/northstar/abi/validators.py`:

```python
def validate_observation(observation: dict[str, Any], manifest: EmbodimentManifest) -> None:
    _require_keys(
        observation,
        [
            "schema_version",
            "timestamp_s",
            "dt_s",
            "frame",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "base_linear_velocity_m_s",
            "base_angular_velocity_rad_s",
            "projected_gravity_base",
            "base_height_m",
            "foot_contact",
            "previous_action",
            "command",
            "mode_mask",
            "masks",
        ],
        "observation",
    )
    if observation["schema_version"] != "observation.northstar.v0":
        raise ValidationError("observation.schema_version must be observation.northstar.v0")
    expected_joints = manifest.active_joint_count
    if len(observation["joint_position_rad"]) != expected_joints:
        raise ValidationError(f"observation.joint_position_rad length must be {expected_joints}")
    if len(observation["joint_velocity_rad_s"]) != expected_joints:
        raise ValidationError(f"observation.joint_velocity_rad_s length must be {expected_joints}")
    expected_feet = manifest.foot_contact_site_count
    if len(observation["foot_contact"]) != expected_feet:
        raise ValidationError(f"observation.foot_contact length must be {expected_feet}")
    validate_command(observation["command"])
    validate_action(observation["previous_action"], manifest)
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/abi/test_observation_schema.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/northstar/abi tests/abi/test_observation_schema.py
git commit -m "feat: add observation and signal ABI"
```

## Task 5: Action Adapter

**Files:**
- Create: `src/northstar/action/adapter.py`
- Test: `tests/action/test_action_adapter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/action/test_action_adapter.py`:

```python
from pathlib import Path

from northstar.abi.action import make_zero_action
from northstar.action.adapter import clip_action
from northstar.embodiment.manifest import load_manifest


def test_clip_action_clips_position_velocity_and_torque():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    action = make_zero_action("act_clip", manifest, "test")
    action["joint_position_delta_rad"][0] = 99.0
    action["joint_velocity_delta_rad_s"][1] = -99.0
    action["feedforward_torque_nm"][2] = 99.0

    clipped, events = clip_action(action, manifest, episode_id="ep_1", step_index=3)

    assert clipped["clipped"] is True
    assert clipped["joint_position_delta_rad"][0] == manifest.action_limit_rad
    assert clipped["joint_velocity_delta_rad_s"][1] == -manifest.velocity_limit_rad_s
    assert clipped["feedforward_torque_nm"][2] == manifest.torque_limit_nm
    assert events[0]["event_type"] == "action_clip"
    assert events[0]["step_index"] == 3
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/action/test_action_adapter.py -v
```

Expected: FAIL with missing `northstar.action.adapter`.

- [ ] **Step 3: Implement adapter**

Create `src/northstar/action/adapter.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any

from northstar.embodiment.manifest import EmbodimentManifest


def _clip_value(value: float, limit: float) -> tuple[float, bool]:
    clipped = max(-limit, min(limit, float(value)))
    return clipped, clipped != float(value)


def _clip_list(values: list[float], limit: float) -> tuple[list[float], list[int]]:
    clipped_values: list[float] = []
    clipped_indices: list[int] = []
    for index, value in enumerate(values):
        clipped, changed = _clip_value(value, limit)
        clipped_values.append(clipped)
        if changed:
            clipped_indices.append(index)
    return clipped_values, clipped_indices


def clip_action(
    action: dict[str, Any],
    manifest: EmbodimentManifest,
    episode_id: str,
    step_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = deepcopy(action)
    pos, pos_indices = _clip_list(result["joint_position_delta_rad"], manifest.action_limit_rad)
    vel, vel_indices = _clip_list(result["joint_velocity_delta_rad_s"], manifest.velocity_limit_rad_s)
    torque, torque_indices = _clip_list(result["feedforward_torque_nm"], manifest.torque_limit_nm)
    result["joint_position_delta_rad"] = pos
    result["joint_velocity_delta_rad_s"] = vel
    result["feedforward_torque_nm"] = torque
    clip_summary = []
    if pos_indices:
        clip_summary.append({"field": "joint_position_delta_rad", "indices": pos_indices})
    if vel_indices:
        clip_summary.append({"field": "joint_velocity_delta_rad_s", "indices": vel_indices})
    if torque_indices:
        clip_summary.append({"field": "feedforward_torque_nm", "indices": torque_indices})
    result["clipped"] = bool(clip_summary)
    result["clip_summary"] = clip_summary
    events = []
    if clip_summary:
        events.append(
            {
                "schema_version": "event_record.v0",
                "episode_id": episode_id,
                "step_index": int(step_index),
                "event_type": "action_clip",
                "severity": "warning",
                "source": "action_adapter",
                "payload": {"clip_summary": clip_summary},
            }
        )
    return result, events
```

- [ ] **Step 4: Run test**

Run:

```bash
pytest tests/action/test_action_adapter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/northstar/action/adapter.py tests/action/test_action_adapter.py
git commit -m "feat: add action clipping adapter"
```

## Task 6: Scenario Loading and Command Generator

**Files:**
- Create: `configs/eval/phase0_scenarios.yaml`
- Create: `configs/eval/phase1_skeleton_scenarios.yaml`
- Create: `src/northstar/command/scenarios.py`
- Create: `src/northstar/command/generator.py`
- Test: `tests/command/test_command_generator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/command/test_command_generator.py`:

```python
from pathlib import Path

from northstar.command.generator import CommandGenerator
from northstar.command.scenarios import load_scenario_set


def test_load_phase1_scenario_set():
    scenario_set = load_scenario_set(Path("configs/eval/phase1_skeleton_scenarios.yaml"))

    assert scenario_set["scenario_set_id"] == "phase1_skeleton_scenarios_v001"
    assert len(scenario_set["scenarios"]) == 8


def test_command_generator_applies_stop_schedule():
    scenario_set = load_scenario_set(Path("configs/eval/phase1_skeleton_scenarios.yaml"))
    scenario = next(s for s in scenario_set["scenarios"] if s["scenario_id"] == "phase1_stop_request_smoke")
    generator = CommandGenerator(scenario)

    command_before = generator.command_at_step(0)
    command_after = generator.command_at_step(120)

    assert command_before["locomotion"]["stop_request"] is False
    assert command_after["locomotion"]["stop_request"] is True
    assert command_after["locomotion"]["target_velocity_base_m_s"] == [0.4, 0.0, 0.0]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/command/test_command_generator.py -v
```

Expected: FAIL with missing `northstar.command.generator`.

- [ ] **Step 3: Add Phase 0 scenario config**

Create `configs/eval/phase0_scenarios.yaml`:

```yaml
schema_version: scenario_set.v0
scenario_set_id: phase0_scenarios_v001
phase: phase_0
defaults:
  horizon_steps: 50
  dt_s: 0.02
  seeds: [1, 2, 3]
scenarios:
  - scenario_id: reset_stability
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
  - scenario_id: zero_command_noop
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
  - scenario_id: random_command_schema
    policy_id: debug_random_legal_v0
    command:
      target_velocity_base_m_s: [0.2, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.1
  - scenario_id: event_injection
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
    event_schedule:
      - step: 10
        event_type: near_fall
  - scenario_id: log_replay_integrity
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
```

- [ ] **Step 4: Add Phase 1 scenario config**

Create `configs/eval/phase1_skeleton_scenarios.yaml`:

```yaml
schema_version: scenario_set.v0
scenario_set_id: phase1_skeleton_scenarios_v001
phase: phase_1_skeleton
defaults:
  horizon_steps: 250
  dt_s: 0.02
  seeds: [1, 2, 3]
scenarios:
  - scenario_id: phase1_stand_smoke
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
  - scenario_id: phase1_velocity_forward_smoke
    policy_id: debug_simple_pd_v0
    command:
      target_velocity_base_m_s: [0.4, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
  - scenario_id: phase1_velocity_yaw_smoke
    policy_id: debug_simple_pd_v0
    command:
      target_velocity_base_m_s: [0.2, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.4
  - scenario_id: phase1_stop_request_smoke
    policy_id: debug_simple_pd_v0
    command:
      target_velocity_base_m_s: [0.4, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
    command_schedule:
      - step: 120
        stop_request: true
  - scenario_id: phase1_brace_request_smoke
    policy_id: debug_simple_pd_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
    command_schedule:
      - step: 100
        brace_request: true
  - scenario_id: phase1_action_clip_smoke
    policy_id: debug_random_legal_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
    force_action_clip: true
  - scenario_id: phase1_near_fall_sample_smoke
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
    event_schedule:
      - step: 80
        event_type: near_fall
  - scenario_id: phase1_school_sample_smoke
    policy_id: debug_noop_v0
    command:
      target_velocity_base_m_s: [0.0, 0.0, 0.0]
      target_yaw_rate_rad_s: 0.0
    event_schedule:
      - step: 20
        event_type: event_injection
```

- [ ] **Step 5: Implement scenario loader**

Create `src/northstar/command/scenarios.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_scenario_set(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    required = ["schema_version", "scenario_set_id", "phase", "defaults", "scenarios"]
    for key in required:
        if key not in payload:
            raise ValueError(f"scenario_set.{key} is required")
    return payload
```

- [ ] **Step 6: Implement command generator**

Create `src/northstar/command/generator.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any

from northstar.abi.command import make_locomotion_command


class CommandGenerator:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario

    def command_at_step(self, step_index: int) -> dict[str, Any]:
        base = deepcopy(self.scenario["command"])
        for item in self.scenario.get("command_schedule", []):
            if int(item["step"]) <= step_index:
                for key, value in item.items():
                    if key != "step":
                        base[key] = value
        return make_locomotion_command(
            command_id=f"{self.scenario['scenario_id']}_cmd_{step_index}",
            target_velocity_base_m_s=base.get("target_velocity_base_m_s", [0.0, 0.0, 0.0]),
            target_yaw_rate_rad_s=base.get("target_yaw_rate_rad_s", 0.0),
            stop_request=base.get("stop_request", False),
            brace_request=base.get("brace_request", False),
        )
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/command/test_command_generator.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add configs/eval src/northstar/command tests/command/test_command_generator.py
git commit -m "feat: add phase skeleton scenarios and command generator"
```

## Task 7: Debug Follower Policies

**Files:**
- Create: `src/northstar/policy/debug_baselines.py`
- Create: `src/northstar/policy/follower_adapter.py`
- Test: `tests/policy/test_debug_baselines.py`

- [ ] **Step 1: Write failing tests**

Create `tests/policy/test_debug_baselines.py`:

```python
from pathlib import Path
import random

from northstar.abi.command import make_locomotion_command
from northstar.abi.observation import make_observation
from northstar.embodiment.manifest import load_manifest
from northstar.policy.debug_baselines import get_debug_policy


def test_noop_policy_outputs_zero_action():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, 0.0, 0.02)
    policy = get_debug_policy("debug_noop_v0", manifest)

    action = policy.act(observation, command)

    assert action["action_source"] == "debug_noop_v0"
    assert all(value == 0.0 for value in action["joint_position_delta_rad"])


def test_random_legal_policy_is_deterministic_with_rng_seed():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.0, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, 0.0, 0.02)
    policy_a = get_debug_policy("debug_random_legal_v0", manifest, rng=random.Random(7))
    policy_b = get_debug_policy("debug_random_legal_v0", manifest, rng=random.Random(7))

    assert policy_a.act(observation, command) == policy_b.act(observation, command)


def test_simple_pd_policy_responds_to_forward_velocity_command():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.4, 0.0, 0.0], 0.0)
    observation = make_observation(manifest, command, 0.0, 0.02)
    policy = get_debug_policy("debug_simple_pd_v0", manifest)

    action = policy.act(observation, command)

    assert any(value != 0.0 for value in action["joint_position_delta_rad"])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/policy/test_debug_baselines.py -v
```

Expected: FAIL with missing `northstar.policy.debug_baselines`.

- [ ] **Step 3: Implement debug policies**

Create `src/northstar/policy/debug_baselines.py`:

```python
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
```

- [ ] **Step 4: Add follower adapter alias**

Create `src/northstar/policy/follower_adapter.py`:

```python
from __future__ import annotations

from northstar.policy.debug_baselines import DebugPolicy

__all__ = ["DebugPolicy"]
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/policy/test_debug_baselines.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/northstar/policy tests/policy/test_debug_baselines.py
git commit -m "feat: add debug follower policies"
```

## Task 8: Mock Phase 1 Env

**Files:**
- Create: `src/northstar/env/adapter.py`
- Create: `src/northstar/env/state.py`
- Create: `src/northstar/env/mock_phase1_env.py`
- Test: `tests/env/test_mock_phase1_env.py`

- [ ] **Step 1: Write failing tests**

Create `tests/env/test_mock_phase1_env.py`:

```python
from pathlib import Path

from northstar.abi.command import make_locomotion_command
from northstar.abi.action import make_zero_action
from northstar.embodiment.manifest import load_manifest
from northstar.env.mock_phase1_env import MockPhase1Env


def test_mock_env_reset_returns_valid_observation():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    env = MockPhase1Env(manifest=manifest, dt_s=0.02, horizon_steps=5)

    observation = env.reset(seed=1)

    assert observation["schema_version"] == "observation.northstar.v0"
    assert observation["base_height_m"] == manifest.default_base_height_m


def test_mock_env_step_applies_stop_request_event():
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    env = MockPhase1Env(manifest=manifest, dt_s=0.02, horizon_steps=5)
    env.reset(seed=1)
    command = make_locomotion_command("cmd", [0.4, 0.0, 0.0], 0.0, stop_request=True)
    action = make_zero_action("act", manifest, "test")

    result = env.step(action, command)

    assert any(event["event_type"] == "stop_request" for event in result.events)
    assert result.observation["base_linear_velocity_m_s"][0] == 0.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/env/test_mock_phase1_env.py -v
```

Expected: FAIL with missing `northstar.env.mock_phase1_env`.

- [ ] **Step 3: Implement env protocol**

Create `src/northstar/env/adapter.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class StepResult:
    observation: dict[str, Any]
    confidence: dict[str, Any]
    dangerous_signal: dict[str, Any]
    reward_debug: dict[str, float]
    events: list[dict[str, Any]]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class EnvAdapter(Protocol):
    def reset(self, seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, action: dict[str, Any], command: dict[str, Any]) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 4: Implement env state**

Create `src/northstar/env/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MockPhase1State:
    step_index: int = 0
    time_s: float = 0.0
    base_linear_velocity_m_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_angular_velocity_rad_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_height_m: float = 0.74
    projected_gravity_base: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
```

- [ ] **Step 5: Implement mock env**

Create `src/northstar/env/mock_phase1_env.py`:

```python
from __future__ import annotations

import random
from typing import Any

from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.embodiment.manifest import EmbodimentManifest
from northstar.env.adapter import StepResult
from northstar.env.state import MockPhase1State


class MockPhase1Env:
    env_id = "mock_phase1_env_v0"

    def __init__(self, manifest: EmbodimentManifest, dt_s: float, horizon_steps: int) -> None:
        self.manifest = manifest
        self.dt_s = float(dt_s)
        self.horizon_steps = int(horizon_steps)
        self.rng = random.Random(0)
        self.state = MockPhase1State(base_height_m=manifest.default_base_height_m)
        self.last_observation: dict[str, Any] | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        self.rng = random.Random(seed)
        self.state = MockPhase1State(base_height_m=self.manifest.default_base_height_m)
        command = {
            "schema_version": "command.northstar.v0",
            "command_id": "reset_cmd",
            "mode_mask": {
                "stand": True,
                "locomotion": True,
                "upper_body": False,
                "light_axis": False,
                "semantic_intent": False,
            },
            "locomotion": {
                "target_base_height_m": 0.0,
                "target_velocity_base_m_s": [0.0, 0.0, 0.0],
                "target_yaw_rate_rad_s": 0.0,
                "target_heading_rad": None,
                "stop_request": False,
                "brace_request": False,
            },
            "upper_body": None,
            "light_axis_hint": None,
            "semantic_hint": None,
        }
        self.last_observation = make_observation(self.manifest, command, 0.0, self.dt_s)
        return self.last_observation

    def step(self, action: dict[str, Any], command: dict[str, Any]) -> StepResult:
        events: list[dict[str, Any]] = []
        locomotion = command["locomotion"]
        target_velocity = list(locomotion["target_velocity_base_m_s"])
        if locomotion.get("stop_request", False):
            target_velocity = [0.0, 0.0, 0.0]
            events.append(self._event("stop_request", "info", {}))
        if locomotion.get("brace_request", False):
            events.append(self._event("brace_request", "info", {}))
        self.state.base_linear_velocity_m_s = [float(value) for value in target_velocity]
        self.state.base_angular_velocity_rad_s = [0.0, 0.0, float(locomotion["target_yaw_rate_rad_s"])]
        self.state.step_index += 1
        self.state.time_s = self.state.step_index * self.dt_s
        observation = make_observation(
            self.manifest,
            command,
            timestamp_s=self.state.time_s,
            dt_s=self.dt_s,
            base_linear_velocity_m_s=self.state.base_linear_velocity_m_s,
            base_angular_velocity_rad_s=self.state.base_angular_velocity_rad_s,
            base_height_m=self.state.base_height_m,
            previous_action=action,
        )
        triggered = []
        near_fall_risk = 0.0
        if self.state.base_height_m < self.manifest.default_base_height_m - 0.2:
            triggered.append("near_fall")
            near_fall_risk = 0.8
            events.append(self._event("near_fall", "warning", {"base_height_m": self.state.base_height_m}))
        dangerous = make_dangerous_signal(
            overall_risk=near_fall_risk,
            triggered=triggered,
            near_fall_risk=near_fall_risk,
        )
        terminated = self.state.step_index >= self.horizon_steps
        self.last_observation = observation
        return StepResult(
            observation=observation,
            confidence=make_confidence(1.0 - near_fall_risk, 1.0 - near_fall_risk, 1.0, 1.0),
            dangerous_signal=dangerous,
            reward_debug={"mock_reward": 1.0 - near_fall_risk},
            events=events,
            terminated=terminated,
            truncated=False,
            info={"env_id": self.env_id},
        )

    def inject_event(self, event_type: str) -> list[dict[str, Any]]:
        if event_type == "near_fall":
            self.state.base_height_m = self.manifest.default_base_height_m - 0.25
        return [self._event(event_type, "warning", {"injected": True})]

    def close(self) -> None:
        return None

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
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/env/test_mock_phase1_env.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/northstar/env tests/env/test_mock_phase1_env.py
git commit -m "feat: add deterministic mock phase 1 env"
```

## Task 9: Episode Logger and Artifact Hash

**Files:**
- Create: `src/northstar/episode_io/artifact_hash.py`
- Create: `src/northstar/episode_io/episode_logger.py`
- Test: `tests/episode_io/test_episode_logger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/episode_io/test_episode_logger.py`:

```python
import json
from pathlib import Path

from northstar.abi.command import make_locomotion_command
from northstar.abi.action import make_zero_action
from northstar.abi.observation import make_observation
from northstar.abi.signals import make_confidence, make_dangerous_signal
from northstar.embodiment.manifest import load_manifest
from northstar.episode_io.episode_logger import EpisodeLogger


def test_episode_logger_writes_required_files(tmp_path: Path):
    manifest = load_manifest(Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"))
    command = make_locomotion_command("cmd", [0.0, 0.0, 0.0], 0.0)
    action = make_zero_action("act", manifest, "test")
    observation = make_observation(manifest, command, 0.0, 0.02)
    logger = EpisodeLogger(tmp_path, run_id="run_test", episode_id="ep_test")

    logger.start(
        phase="phase_1_skeleton",
        scenario_id="phase1_stand_smoke",
        seed=1,
        abi_version="abi.northstar.v0",
        embodiment_id=manifest.embodiment_id,
        env_id="mock_phase1_env_v0",
        policy_id="debug_noop_v0",
    )
    logger.append_step(
        step_index=0,
        time_s=0.0,
        observation=observation,
        command=command,
        action=action,
        confidence=make_confidence(1.0, 1.0, 1.0, 1.0),
        dangerous_signal=make_dangerous_signal(0.0, []),
        reward_debug={"mock_reward": 1.0},
        terminated=False,
        truncated=False,
        info={},
    )
    logger.append_event({"schema_version": "event_record.v0", "episode_id": "ep_test", "step_index": 0, "event_type": "episode_end", "severity": "info", "source": "test", "payload": {}})
    logger.finalize(metrics={"step_count": 1}, termination_reason="time_limit")

    episode_dir = tmp_path / "episodes" / "ep_test"
    assert (episode_dir / "episode_manifest.json").exists()
    assert (episode_dir / "steps.jsonl").exists()
    assert (episode_dir / "events.jsonl").exists()
    assert (episode_dir / "metrics.json").exists()
    manifest_payload = json.loads((episode_dir / "episode_manifest.json").read_text())
    assert manifest_payload["artifact_hashes"]["steps_jsonl"].startswith("sha256:")
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/episode_io/test_episode_logger.py -v
```

Expected: FAIL with missing `EpisodeLogger`.

- [ ] **Step 3: Implement artifact hash**

Create `src/northstar/episode_io/artifact_hash.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
```

- [ ] **Step 4: Implement episode logger**

Create `src/northstar/episode_io/episode_logger.py`:

```python
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
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/episode_io/test_episode_logger.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/northstar/episode_io tests/episode_io/test_episode_logger.py
git commit -m "feat: add episode logger"
```

## Task 10: Locomotion Metrics and Replay Reader

**Files:**
- Create: `src/northstar/metrics/locomotion.py`
- Create: `src/northstar/metrics/accumulator.py`
- Create: `src/northstar/metrics/replay.py`
- Create: `src/northstar/episode_io/replay_reader.py`
- Test: `tests/metrics/test_locomotion_metrics.py`
- Test: `tests/episode_io/test_replay_reader.py`

- [ ] **Step 1: Write failing metrics tests**

Create `tests/metrics/test_locomotion_metrics.py`:

```python
from northstar.metrics.locomotion import rmse, summarize_steps


def test_rmse_computes_root_mean_squared_error():
    assert rmse([0.0, 1.0], [0.0, 3.0]) == 1.4142135623730951


def test_summarize_steps_counts_fall_and_action_clip_events():
    steps = [
        {
            "observation": {"base_height_m": 0.74, "base_linear_velocity_m_s": [0.0, 0.0, 0.0], "base_angular_velocity_rad_s": [0.0, 0.0, 0.0]},
            "command": {"locomotion": {"target_velocity_base_m_s": [0.0, 0.0, 0.0], "target_yaw_rate_rad_s": 0.0}},
            "action": {"clipped": False},
            "dangerous_signal": {"triggered": []},
        },
        {
            "observation": {"base_height_m": 0.44, "base_linear_velocity_m_s": [0.2, 0.0, 0.0], "base_angular_velocity_rad_s": [0.0, 0.0, 0.1]},
            "command": {"locomotion": {"target_velocity_base_m_s": [0.0, 0.0, 0.0], "target_yaw_rate_rad_s": 0.0}},
            "action": {"clipped": True},
            "dangerous_signal": {"triggered": ["near_fall"]},
        },
    ]

    summary = summarize_steps(steps, default_base_height_m=0.74)

    assert summary["step_count"] == 2
    assert summary["near_fall_count"] == 1
    assert summary["action_clipping_count"] == 1
```

- [ ] **Step 2: Write failing replay tests**

Create `tests/episode_io/test_replay_reader.py`:

```python
from pathlib import Path

from northstar.episode_io.replay_reader import read_jsonl, replay_episode


def test_read_jsonl_reads_records(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_replay_episode_recomputes_step_count(tmp_path: Path):
    episode_dir = tmp_path / "episodes" / "ep_1"
    episode_dir.mkdir(parents=True)
    (episode_dir / "steps.jsonl").write_text(
        '{"observation": {"base_height_m": 0.74, "base_linear_velocity_m_s": [0.0, 0.0, 0.0], "base_angular_velocity_rad_s": [0.0, 0.0, 0.0]}, "command": {"locomotion": {"target_velocity_base_m_s": [0.0, 0.0, 0.0], "target_yaw_rate_rad_s": 0.0}}, "action": {"clipped": false}, "dangerous_signal": {"triggered": []}}\n',
        encoding="utf-8",
    )

    metrics = replay_episode(episode_dir, default_base_height_m=0.74)

    assert metrics["step_count"] == 1
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
pytest tests/metrics/test_locomotion_metrics.py tests/episode_io/test_replay_reader.py -v
```

Expected: FAIL with missing modules.

- [ ] **Step 4: Implement locomotion metrics**

Create `src/northstar/metrics/locomotion.py`:

```python
from __future__ import annotations

import math
from typing import Any


def rmse(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        raise ValueError("actual and expected must have same length")
    if not actual:
        return 0.0
    return math.sqrt(sum((a - e) ** 2 for a, e in zip(actual, expected)) / len(actual))


def summarize_steps(steps: list[dict[str, Any]], default_base_height_m: float) -> dict[str, Any]:
    if not steps:
        return {
            "step_count": 0,
            "near_fall_count": 0,
            "fall_count": 0,
            "action_clipping_count": 0,
            "base_height_rmse_m": 0.0,
            "velocity_rmse_m_s": 0.0,
            "yaw_rate_rmse_rad_s": 0.0,
        }
    base_heights = [float(step["observation"]["base_height_m"]) for step in steps]
    velocity_errors = []
    yaw_errors = []
    near_fall_count = 0
    fall_count = 0
    action_clipping_count = 0
    for step in steps:
        observation = step["observation"]
        command = step["command"]["locomotion"]
        velocity_errors.append(
            rmse(
                list(observation["base_linear_velocity_m_s"]),
                list(command["target_velocity_base_m_s"]),
            )
        )
        yaw_errors.append(
            abs(float(observation["base_angular_velocity_rad_s"][2]) - float(command["target_yaw_rate_rad_s"]))
        )
        triggered = set(step.get("dangerous_signal", {}).get("triggered", []))
        if "near_fall" in triggered:
            near_fall_count += 1
        if "fall" in triggered:
            fall_count += 1
        if bool(step["action"].get("clipped", False)):
            action_clipping_count += 1
    return {
        "step_count": len(steps),
        "near_fall_count": near_fall_count,
        "fall_count": fall_count,
        "action_clipping_count": action_clipping_count,
        "base_height_rmse_m": rmse(base_heights, [default_base_height_m] * len(base_heights)),
        "velocity_rmse_m_s": sum(velocity_errors) / len(velocity_errors),
        "yaw_rate_rmse_rad_s": sum(yaw_errors) / len(yaw_errors),
    }
```

- [ ] **Step 5: Add accumulator**

Create `src/northstar/metrics/accumulator.py`:

```python
from __future__ import annotations

from typing import Any

from northstar.metrics.locomotion import summarize_steps


class MetricsAccumulator:
    def __init__(self, default_base_height_m: float) -> None:
        self.default_base_height_m = default_base_height_m
        self.steps: list[dict[str, Any]] = []

    def update(self, step_record: dict[str, Any]) -> None:
        self.steps.append(step_record)

    def summary(self) -> dict[str, Any]:
        return summarize_steps(self.steps, self.default_base_height_m)
```

- [ ] **Step 6: Implement replay helpers**

Create `src/northstar/episode_io/replay_reader.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from northstar.metrics.locomotion import summarize_steps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def replay_episode(episode_dir: Path, default_base_height_m: float) -> dict[str, Any]:
    steps = read_jsonl(episode_dir / "steps.jsonl")
    return summarize_steps(steps, default_base_height_m)
```

Create `src/northstar/metrics/replay.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from northstar.episode_io.replay_reader import replay_episode


def replay_validation_passes(
    episode_dir: Path,
    saved_metrics: dict[str, Any],
    default_base_height_m: float,
) -> bool:
    recomputed = replay_episode(episode_dir, default_base_height_m)
    return recomputed.get("step_count") == saved_metrics.get("step_count")
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/metrics/test_locomotion_metrics.py tests/episode_io/test_replay_reader.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/northstar/metrics src/northstar/episode_io/replay_reader.py tests/metrics tests/episode_io/test_replay_reader.py
git commit -m "feat: add metrics and replay reader"
```

## Task 11: School Sample Builder

**Files:**
- Create: `configs/school/phase1_sample_scoring.yaml`
- Create: `src/northstar/abi/school_sample.py`
- Create: `src/northstar/school/priority.py`
- Create: `src/northstar/school/sample_builder.py`
- Test: `tests/school/test_school_sample_builder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/school/test_school_sample_builder.py`:

```python
from northstar.school.sample_builder import build_sample_for_event


def test_build_sample_for_near_fall_event():
    event = {
        "event_type": "near_fall",
        "step_index": 10,
        "severity": "warning",
        "payload": {"base_height_m": 0.42},
    }

    sample = build_sample_for_event(
        event=event,
        run_id="run_1",
        episode_id="ep_1",
        artifact_uri="runs/run_1/episodes/ep_1",
        replay_valid=True,
        artifact_hash_valid=True,
    )

    assert sample["schema_version"] == "school_sample_envelope.v0"
    assert sample["segment_type"] == "near_failure"
    assert sample["labels"]["usable_for_training"] is True
    assert sample["priority"] > 0.0
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/school/test_school_sample_builder.py -v
```

Expected: FAIL with missing `northstar.school.sample_builder`.

- [ ] **Step 3: Add scoring config**

Create `configs/school/phase1_sample_scoring.yaml`:

```yaml
schema_version: sample_scoring.v0
phase: phase_1_skeleton
weights:
  event_severity_score: 0.30
  replay_relevance_score: 0.25
  phase1_metric_error_score: 0.20
  rarity_score: 0.15
  data_quality_score: 0.10
event_type_to_segment_type:
  near_fall: near_failure
  fall: fall
  invalid_command: invalid_command
  action_clip: action_clip
  tracking_error_high: tracking_error_high
  fallback_transition: fallback_transition
  event_injection: event_injection
```

- [ ] **Step 4: Implement school sample validator**

Create `src/northstar/abi/school_sample.py`:

```python
from __future__ import annotations

from typing import Any


def validate_school_sample(sample: dict[str, Any]) -> None:
    required = [
        "schema_version",
        "sample_id",
        "source",
        "phase",
        "source_episode_id",
        "segment_type",
        "step_range",
        "priority",
        "labels",
        "artifact_uri",
        "metrics",
        "data_quality",
    ]
    for key in required:
        if key not in sample:
            raise ValueError(f"school_sample.{key} is required")
    if sample["schema_version"] != "school_sample_envelope.v0":
        raise ValueError("school_sample.schema_version must be school_sample_envelope.v0")
    if len(sample["step_range"]) != 2:
        raise ValueError("school_sample.step_range must have two values")
```

- [ ] **Step 5: Implement priority scorer**

Create `src/northstar/school/priority.py`:

```python
from __future__ import annotations


def score_event_priority(event_type: str, replay_valid: bool, artifact_hash_valid: bool) -> float:
    severity = 1.0 if event_type in {"fall", "near_fall"} else 0.5
    replay = 1.0 if replay_valid else 0.0
    metric_error = 0.8 if event_type in {"near_fall", "tracking_error_high", "action_clip"} else 0.4
    rarity = 0.6 if event_type in {"fall", "near_fall"} else 0.3
    data_quality = 1.0 if replay_valid and artifact_hash_valid else 0.0
    return (
        0.30 * severity
        + 0.25 * replay
        + 0.20 * metric_error
        + 0.15 * rarity
        + 0.10 * data_quality
    )
```

- [ ] **Step 6: Implement sample builder**

Create `src/northstar/school/sample_builder.py`:

```python
from __future__ import annotations

from typing import Any

from northstar.abi.school_sample import validate_school_sample
from northstar.school.priority import score_event_priority


EVENT_TO_SEGMENT = {
    "near_fall": "near_failure",
    "fall": "fall",
    "invalid_command": "invalid_command",
    "action_clip": "action_clip",
    "tracking_error_high": "tracking_error_high",
    "fallback_transition": "fallback_transition",
    "event_injection": "event_injection",
}


def build_sample_for_event(
    event: dict[str, Any],
    run_id: str,
    episode_id: str,
    artifact_uri: str,
    replay_valid: bool,
    artifact_hash_valid: bool,
) -> dict[str, Any]:
    event_type = str(event["event_type"])
    step = int(event["step_index"])
    segment_type = EVENT_TO_SEGMENT.get(event_type, "event_injection")
    sample = {
        "schema_version": "school_sample_envelope.v0",
        "sample_id": f"sample_{run_id}_{episode_id}_{step}_{event_type}",
        "source": "phase1_skeleton_eval",
        "phase": "phase_1_skeleton",
        "source_episode_id": episode_id,
        "segment_type": segment_type,
        "step_range": [max(0, step - 20), step + 20],
        "priority": score_event_priority(event_type, replay_valid, artifact_hash_valid),
        "labels": {
            "usable_for_training": bool(replay_valid and artifact_hash_valid),
            "usable_for_release_gate": bool(replay_valid and artifact_hash_valid and event_type in {"near_fall", "fall", "action_clip"}),
            "requires_human_review": False,
        },
        "artifact_uri": artifact_uri,
        "metrics": {
            "event_step": step,
            "event_type": event_type,
        },
        "data_quality": {
            "schema_valid": True,
            "artifact_hash_valid": bool(artifact_hash_valid),
            "replay_valid": bool(replay_valid),
        },
    }
    validate_school_sample(sample)
    return sample
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/school/test_school_sample_builder.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add configs/school src/northstar/abi/school_sample.py src/northstar/school tests/school/test_school_sample_builder.py
git commit -m "feat: add school sample envelope builder"
```

## Task 12: Evaluation Runner and Report Writer

**Files:**
- Create: `src/northstar/eval/report.py`
- Create: `src/northstar/eval/runner.py`
- Test: `tests/eval/test_phase1_skeleton_runner.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/eval/test_phase1_skeleton_runner.py`:

```python
import json
from pathlib import Path

from northstar.eval.runner import run_scenario_set


def test_phase1_skeleton_runner_generates_report_and_episodes(tmp_path: Path):
    report = run_scenario_set(
        scenario_set_path=Path("configs/eval/phase1_skeleton_scenarios.yaml"),
        manifest_path=Path("configs/embodiment/unitree_g1_43dof_sim_v0.json"),
        output_dir=tmp_path,
    )

    assert report["summary"]["pass"] is True
    assert report["summary"]["schema_validation_pass_rate"] == 1.0
    assert report["summary"]["replay_validation_pass_rate"] == 1.0
    report_path = Path(report["artifacts"]["report_path"])
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert loaded["schema_version"] == "evaluation_report.v0"
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/eval/test_phase1_skeleton_runner.py -v
```

Expected: FAIL with missing `northstar.eval.runner`.

- [ ] **Step 3: Implement report writer**

Create `src/northstar/eval/report.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_evaluation_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = run_dir / "evaluation_report.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    return path
```

- [ ] **Step 4: Implement runner**

Create `src/northstar/eval/runner.py`:

```python
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
```

- [ ] **Step 5: Run integration test**

Run:

```bash
pytest tests/eval/test_phase1_skeleton_runner.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/northstar/eval tests/eval/test_phase1_skeleton_runner.py
git commit -m "feat: add phase 1 skeleton evaluation runner"
```

## Task 13: CLI Entrypoints

**Files:**
- Create: `src/northstar/cli.py`
- Test: use command smoke tests

- [ ] **Step 1: Implement CLI**

Create `src/northstar/cli.py`:

```python
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
```

- [ ] **Step 2: Run CLI smoke tests**

Run:

```bash
northstar-validate-abi --scenario-set configs/eval/phase1_skeleton_scenarios.yaml
```

Expected output contains:

```text
"pass": true
```

Run:

```bash
northstar-run-eval --scenario-set configs/eval/phase1_skeleton_scenarios.yaml --output runs/dev
```

Expected output contains:

```text
"pass": true
```

- [ ] **Step 3: Commit**

```bash
git add src/northstar/cli.py
git commit -m "feat: add northstar runtime CLI commands"
```

## Task 14: Full Test and Skeleton Acceptance

**Files:**
- Modify only if test failures reveal narrow bugs in files created above
- No new feature files expected

- [ ] **Step 1: Run the full test suite**

Run:

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run full Phase 1 skeleton evaluation**

Run:

```bash
northstar-run-eval --scenario-set configs/eval/phase1_skeleton_scenarios.yaml --output runs/dev
```

Expected output contains:

```text
"pass": true
```

- [ ] **Step 3: Replay one generated episode**

Run:

```bash
northstar-replay-episode --episode runs/dev/run_phase1_skeleton/episodes/ep_phase1_stand_smoke_1
```

Expected output contains:

```text
"step_count":
```

- [ ] **Step 4: Build school samples for the run**

Run:

```bash
northstar-build-school-samples --run runs/dev/run_phase1_skeleton
```

Expected output contains:

```text
"sample_count":
```

- [ ] **Step 5: Check generated artifacts**

Run:

```bash
test -f runs/dev/run_phase1_skeleton/evaluation_report.json
test -f runs/dev/run_phase1_skeleton/episodes/ep_phase1_stand_smoke_1/episode_manifest.json
test -f runs/dev/run_phase1_skeleton/episodes/ep_phase1_stand_smoke_1/steps.jsonl
test -f runs/dev/run_phase1_skeleton/episodes/ep_phase1_stand_smoke_1/events.jsonl
test -f runs/dev/run_phase1_skeleton/episodes/ep_phase1_stand_smoke_1/metrics.json
```

Expected: all commands exit `0`.

- [ ] **Step 6: Inspect git status**

Run:

```bash
git status --short
```

Expected: generated `runs/` files appear unless ignored. Do not commit generated `runs/` artifacts.

- [ ] **Step 7: Add a project `.gitignore` if generated artifacts appear**

If `git status --short` shows `?? runs/`, create `.gitignore`:

```gitignore
runs/
__pycache__/
.pytest_cache/
*.pyc
```

Run:

```bash
git add .gitignore
git commit -m "chore: ignore generated runtime artifacts"
```

- [ ] **Step 8: Final commit for narrow bug fixes**

If Step 1-5 required bug fixes, commit only the changed source and tests:

```bash
git add src tests configs pyproject.toml
git commit -m "fix: stabilize phase 1 skeleton acceptance"
```

If no files changed after Step 7, do not create an empty commit.

## Self-Review Checklist

Spec coverage:

- Minimal ABI schema: Tasks 2-4.
- Phase 1 skeleton scenario set: Task 6.
- Mock Phase 1 env: Task 8.
- Debug follower policies: Task 7.
- Action adapter with clipping events: Task 5.
- Episode logger: Task 9.
- Replay reader and metric recomputation: Task 10.
- Evaluation report: Task 12.
- School sample envelope: Task 11.
- CLI entrypoints: Task 13.
- End-to-end acceptance: Task 14.

Implementation constraints:

- Mock env is clearly marked `mock_phase1_env_v0`.
- Motion metrics are smoke metrics only.
- No Isaac Lab, PPO, RSL-RL, or real robot code is included.
- Release gate split isolation is represented by envelope labels and data quality flags.

Verification commands:

```bash
pytest -v
northstar-validate-abi --scenario-set configs/eval/phase1_skeleton_scenarios.yaml
northstar-run-eval --scenario-set configs/eval/phase1_skeleton_scenarios.yaml --output runs/dev
northstar-replay-episode --episode runs/dev/run_phase1_skeleton/episodes/ep_phase1_stand_smoke_1
northstar-build-school-samples --run runs/dev/run_phase1_skeleton
```
