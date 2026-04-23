# Phase 1 最小可运行骨架设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [Phase 1 基础运动底座与学校最小闭环设计](./2026-04-23-phase-1-locomotion-school-loop-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [线上/线下验证树与指标设计](./2026-04-23-validation-tree-and-metrics-design.md)

## 1. 目标

本文定义 Phase 0/1 的第一刀落地范围：最小可运行 runtime skeleton。

它的目标不是训练出可用 locomotion policy，而是先跑通以下闭环：

```text
phase1 scenario config
  -> mock/env adapter
  -> command generator
  -> debug follower policy
  -> action adapter
  -> episode logger
  -> replay reader
  -> metrics/eval report
  -> school sample envelope
```

该骨架必须证明：

1. Phase 0 ABI 可以落成代码并被校验。
2. Phase 1 最小场景可以通过统一 runner 执行。
3. episode log 可以被写入、读取、回放和复算指标。
4. debug follower policy 可以通过 follower/action adapter 产生合法 action。
5. evaluation report 可以稳定生成。
6. school sample envelope 可以从 episode 片段生成。
7. 后续 Isaac Lab、RSL-RL、PPO 和真实训练可以替换 mock/env 与 debug policy，而不重写数据链路。

## 2. 范围

本文覆盖：

- 最小仓库结构。
- 最小 ABI schema。
- mock Phase 1 environment。
- command generator。
- debug follower policy。
- action adapter。
- episode logger。
- replay reader。
- metrics 与 evaluation runner。
- school sample envelope builder。
- 最小测试集。
- skeleton 的验收标准。

本文不覆盖：

- PPO/RSL-RL 训练配方。
- Isaac Lab 具体接入代码。
- Unitree G1 真实资产调试。
- 高性能 locomotion reward。
- 小脑 light-axis。
- 本地大脑。
- 联邦训练。
- 真实机器人验收。
- 完整 model gate blending。

这些内容应在后续子规格中展开，例如：

- `phase-1-training-recipe-design.md`
- `phase-1-isaac-lab-adapter-design.md`
- `phase-1-school-release-implementation-design.md`

## 3. 设计原则

### 3.1 先证明链路，不证明性能

第一版 skeleton 的验收目标是“能跑、能记、能评、能回放、能生成学校样本”。任何 locomotion 性能指标只做 smoke check，不作为训练成功声明。

### 3.2 Mock Env 是接口替身，不是物理替身

mock env 只负责验证 runtime 合约：

- reset/step 生命周期。
- observation shape 与 schema。
- command 注入。
- action clipping。
- event injection。
- episode log。
- replay metrics。

它不承担真实动力学可信度。

### 3.3 后续仿真器只替换 Env Adapter

Isaac Lab 或其他目标仿真器接入后，应替换 `EnvAdapter` 实现，而不是重写 evaluation runner、logger、replay、metrics、school sample builder。

### 3.4 数据格式从第一天可回放

每个 episode 必须可通过 metadata + step records + events 重建执行序列。没有 replay，就没有可信 release gate。

### 3.5 学校样本从第一天生成

Phase 1 的学校系统不等待训练成熟。即使是 mock env 和 debug policy，也必须能生成 envelope，用于验证 sample ingress、metadata index 和 release gate split isolation。

## 4. 第一条可运行链路

最小链路如下：

```text
configs/eval/phase1_skeleton_scenarios.yaml
  -> ScenarioLoader
  -> MockPhase1Env.reset(seed, scenario)
  -> CommandGenerator.next(step)
  -> ObservationBuilder.build(state, command)
  -> DebugFollowerPolicy.act(observation, command)
  -> ActionAdapter.clip_and_format(action)
  -> MockPhase1Env.step(action)
  -> EpisodeLogger.append_step(...)
  -> MetricsAccumulator.update(...)
  -> EpisodeLogger.finalize()
  -> ReplayReader.load(episode_dir)
  -> MetricsRecalculator.recompute()
  -> EvaluationReportWriter.write()
  -> SchoolSampleBuilder.extract_segments()
```

该链路的第一版可以完全本地运行，不依赖 GPU、Isaac Lab 或外部服务。

## 5. 模块边界

### 5.1 Config Layer

职责：

- 定义 embodiment manifest。
- 定义 ABI 版本。
- 定义 scenario set。
- 定义 metric threshold。
- 定义 school sample scoring 初版。

不负责：

- 训练策略。
- 修改 runtime 行为。
- 保存 episode 数据。

### 5.2 ABI Layer

职责：

- 定义最小 schema。
- 提供 validator。
- 提供 schema version。
- 提供 shape/dtype/range 检查。

不负责：

- 业务逻辑。
- 仿真器状态更新。
- reward 计算。

### 5.3 Env Adapter

职责：

- 提供 `reset()`、`step()`、`close()`。
- 将底层环境状态转换为 ABI observation。
- 接受 ABI action。
- 产生 step result、events、termination。

不负责：

- 决定 command 采样策略。
- 决定 policy 行为。
- 写 episode log。

### 5.4 Command Generator

职责：

- 根据 scenario 生成 Phase 1 locomotion command。
- 支持 fixed command、random legal command、event command。
- 支持 stop/brace 事件注入。

不负责：

- 直接修改 policy。
- 直接修改 env state。
- 生成上肢、小脑或语义 command。

### 5.5 Debug Follower Policy

职责：

- 在不训练的情况下输出合法 action。
- 提供 `noop`、`random`、`simple_pd` baseline。
- 用于连通性测试和日志验证。

不负责：

- 声明 locomotion 能力。
- 作为 Phase 1 最终 follower。
- 覆盖 action adapter。

### 5.6 Action Adapter

职责：

- 将 policy output 规范化为 ABI action。
- 执行 clipping。
- 记录 action clip event。
- 保证输出不越过 embodiment limit。

不负责：

- 优化策略。
- 改写 command。
- 隐藏非法 action。

### 5.7 Episode Logger

职责：

- 写入 episode manifest。
- 写入 step records。
- 写入 events。
- 写入 episode metrics。
- 写入 artifact hash。

不负责：

- 训练数据采样。
- 删除 release gate 样本。
- 修改 replay 指标。

### 5.8 Replay Reader

职责：

- 读取 episode 目录。
- 重建 step 序列。
- 复算 metrics。
- 校验 replay result 与原始 metrics。

不负责：

- 重跑仿真。
- 修改原始 episode。

### 5.9 Evaluation Runner

职责：

- 执行 scenario set。
- 固定 seeds。
- 调用 env/policy/logger/metrics。
- 汇总 evaluation report。
- 返回 pass/fail。

不负责：

- 训练模型。
- 自动调参。
- 发布 candidate。

### 5.10 School Sample Builder

职责：

- 从 episode 和 events 中抽取高价值片段。
- 生成 school sample envelope。
- 标注 segment type、priority、data quality。
- 区分 training / validation / release_gate candidate。

不负责：

- 上传云端。
- 聚合联邦模型。
- 发布模型。

## 6. 推荐仓库结构

第一版实现建议使用 Python 包结构：

```text
configs/
  embodiment/
    unitree_g1_43dof_sim_v0.json
  abi/
    abi_northstar_v0.json
  eval/
    phase0_scenarios.yaml
    phase1_skeleton_scenarios.yaml
  school/
    phase1_sample_scoring.yaml

src/
  northstar/
    abi/
      __init__.py
      observation.py
      command.py
      action.py
      confidence.py
      dangerous_signal.py
      episode.py
      model_manifest.py
      school_sample.py
      validators.py
    embodiment/
      __init__.py
      manifest.py
      loader.py
    env/
      __init__.py
      adapter.py
      mock_phase1_env.py
      state.py
    command/
      __init__.py
      generator.py
      scenarios.py
    policy/
      __init__.py
      follower_adapter.py
      debug_baselines.py
    action/
      __init__.py
      adapter.py
    logging/
      __init__.py
      episode_logger.py
      replay_reader.py
      artifact_hash.py
    metrics/
      __init__.py
      locomotion.py
      replay.py
      accumulator.py
    eval/
      __init__.py
      runner.py
      report.py
    school/
      __init__.py
      sample_builder.py
      priority.py

tests/
  abi/
    test_observation_schema.py
    test_command_schema.py
    test_action_schema.py
    test_episode_schema.py
  env/
    test_mock_phase1_env.py
  command/
    test_command_generator.py
  policy/
    test_debug_baselines.py
  logging/
    test_episode_logger.py
    test_replay_reader.py
  metrics/
    test_locomotion_metrics.py
  eval/
    test_phase1_skeleton_runner.py
  school/
    test_school_sample_builder.py
```

## 7. 最小 ABI

第一版 ABI 只实现 Phase 0/1 必需字段，并保留后续扩展位。

### 7.1 Observation

```json
{
  "schema_version": "observation.northstar.v0",
  "timestamp_s": 0.0,
  "dt_s": 0.02,
  "frame": "base",
  "joint_position_rad": [],
  "joint_velocity_rad_s": [],
  "base_linear_velocity_m_s": [0.0, 0.0, 0.0],
  "base_angular_velocity_rad_s": [0.0, 0.0, 0.0],
  "projected_gravity_base": [0.0, 0.0, -1.0],
  "base_height_m": 0.0,
  "foot_contact": [],
  "previous_action": {},
  "command": {},
  "mode_mask": {
    "stand": true,
    "locomotion": true,
    "upper_body": false,
    "light_axis": false,
    "semantic_intent": false
  },
  "masks": {
    "privileged": false,
    "upper_body_command_enabled": false,
    "light_axis_enabled": false,
    "semantic_hint_enabled": false
  }
}
```

要求：

- `joint_position_rad` 长度必须等于 embodiment manifest 中 active joint count。
- `joint_velocity_rad_s` 同上。
- `foot_contact` 长度必须等于 foot contact site count。
- `command` 必须通过 command schema。
- `previous_action` 必须通过 action schema 或为空初始值。

### 7.2 Command

```json
{
  "schema_version": "command.northstar.v0",
  "command_id": "cmd_000001",
  "mode_mask": {
    "stand": true,
    "locomotion": true,
    "upper_body": false,
    "light_axis": false,
    "semantic_intent": false
  },
  "locomotion": {
    "target_base_height_m": 0.0,
    "target_velocity_base_m_s": [0.0, 0.0, 0.0],
    "target_yaw_rate_rad_s": 0.0,
    "target_heading_rad": null,
    "stop_request": false,
    "brace_request": false
  },
  "upper_body": null,
  "light_axis_hint": null,
  "semantic_hint": null
}
```

Phase 1 validator 必须拒绝：

- `target_velocity_base_m_s.z != 0.0`。
- `upper_body` enabled。
- `light_axis_hint` enabled。
- `semantic_hint` enabled。
- 超出 Phase 1 command range 的速度或 yaw。

### 7.3 Action

```json
{
  "schema_version": "action.northstar.v0",
  "action_id": "act_000001",
  "joint_position_delta_rad": [],
  "joint_velocity_delta_rad_s": [],
  "feedforward_torque_nm": [],
  "action_source": "debug_policy",
  "clipped": false,
  "clip_summary": []
}
```

要求：

- 所有关节数组长度必须匹配 active joint count。
- action adapter 必须在进入 env 前执行 range check。
- 如果发生 clipping，必须写入 `action_clip` event。

### 7.4 Confidence

```json
{
  "schema_version": "confidence.northstar.v0",
  "overall": 1.0,
  "stability": 1.0,
  "tracking": 1.0,
  "fallback": 1.0,
  "source": "debug"
}
```

Phase 1 skeleton 可以用规则生成 confidence，不要求模型预测。

### 7.5 Dangerous Signal

```json
{
  "schema_version": "dangerous_signal.northstar.v0",
  "overall_risk": 0.0,
  "fall_risk": 0.0,
  "near_fall_risk": 0.0,
  "limit_risk": 0.0,
  "tracking_risk": 0.0,
  "triggered": []
}
```

Phase 1 skeleton 的 dangerous signal 使用规则：

- base height 低于阈值触发 fall/near-fall。
- projected gravity 偏离阈值触发 tilt risk。
- action clipping 触发 limit risk。
- tracking error 超阈值触发 tracking risk。

## 8. Episode 数据格式

每个 run 输出目录：

```text
runs/
  <run_id>/
    run_manifest.json
    evaluation_report.json
    episodes/
      <episode_id>/
        episode_manifest.json
        steps.jsonl
        events.jsonl
        metrics.json
        school_samples.jsonl
```

### 8.1 Episode Manifest

```json
{
  "schema_version": "episode_manifest.v0",
  "episode_id": "ep_000001",
  "run_id": "run_20260423_000001",
  "phase": "phase_1_skeleton",
  "scenario_id": "zero_command_noop",
  "seed": 1,
  "abi_version": "abi.northstar.v0",
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "env_id": "mock_phase1_env_v0",
  "policy_id": "debug_noop_v0",
  "started_at": "2026-04-23T10:00:00+08:00",
  "ended_at": "2026-04-23T10:00:05+08:00",
  "step_count": 250,
  "termination_reason": "time_limit",
  "artifact_hashes": {
    "steps_jsonl": "sha256:...",
    "events_jsonl": "sha256:...",
    "metrics_json": "sha256:..."
  }
}
```

### 8.2 Step Record

```json
{
  "schema_version": "step_record.v0",
  "episode_id": "ep_000001",
  "step_index": 0,
  "time_s": 0.0,
  "observation": {},
  "command": {},
  "action": {},
  "confidence": {},
  "dangerous_signal": {},
  "reward_debug": {},
  "terminated": false,
  "truncated": false,
  "info": {}
}
```

### 8.3 Event Record

```json
{
  "schema_version": "event_record.v0",
  "episode_id": "ep_000001",
  "step_index": 120,
  "event_type": "near_fall",
  "severity": "warning",
  "source": "mock_phase1_env",
  "payload": {
    "base_height_m": 0.42,
    "threshold_m": 0.45
  }
}
```

必需 event types：

```text
episode_start
episode_end
invalid_command
action_clip
near_fall
fall
tracking_error_high
stop_request
brace_request
fallback_transition
event_injection
```

## 9. Mock Phase 1 Env

### 9.1 Env State

mock env state 至少包含：

- time。
- joint position。
- joint velocity。
- base linear velocity。
- base angular velocity。
- base height。
- projected gravity。
- foot contacts。
- previous action。
- event flags。

### 9.2 Step 规则

第一版使用确定性、简单、可复现规则：

1. command 中目标速度影响 mock base velocity。
2. debug action 影响 joint position/velocity。
3. action clipping 增加 limit risk。
4. event injection 可以直接制造 near-fall、fall、invalid command。
5. stop_request 将目标速度衰减到 0。
6. brace_request 降低 near-fall risk。

这些规则只用于验证指标和事件链路，不声明物理可信。

### 9.3 Termination

termination reason：

```text
time_limit
fall
invalid_command
nan_state
schema_error
manual_abort
```

mock env 必须支持：

- fixed horizon。
- fall termination。
- invalid command termination。
- deterministic replay by seed。

## 10. Debug Policies

### 10.1 Noop Policy

输出全零 action。

用途：

- zero command。
- reset stability。
- logger/replay smoke test。

### 10.2 Random Legal Policy

在合法 action range 内采样。

用途：

- action schema validation。
- action clipping 边界测试。
- random command smoke test。

### 10.3 Simple PD Policy

根据目标速度和当前速度生成简单 joint delta 或 mock control signal。

用途：

- phase1 velocity command smoke。
- stop/brace event smoke。
- metrics non-zero behavior validation。

Simple PD 不作为真实 locomotion policy。

## 11. Scenario Set

### 11.1 Phase 0 Skeleton Scenarios

```text
reset_stability
zero_command_noop
random_command_schema
event_injection
log_replay_integrity
```

### 11.2 Phase 1 Skeleton Scenarios

```text
phase1_stand_smoke
phase1_velocity_forward_smoke
phase1_velocity_yaw_smoke
phase1_stop_request_smoke
phase1_brace_request_smoke
phase1_action_clip_smoke
phase1_near_fall_sample_smoke
phase1_school_sample_smoke
```

### 11.3 Scenario Config

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
    expected:
      schema_validation_pass: true
      replay_validation_pass: true
      school_sample_generation_min_count: 1
  - scenario_id: phase1_stop_request_smoke
    policy_id: debug_simple_pd_v0
    command_schedule:
      - step: 0
        target_velocity_base_m_s: [0.4, 0.0, 0.0]
      - step: 120
        stop_request: true
    expected:
      stop_request_event_count_min: 1
```

## 12. Metrics

### 12.1 Infrastructure Metrics

必需：

- schema validation pass rate。
- replay validation pass rate。
- episode count。
- step count。
- event count。
- artifact hash validation pass rate。
- school sample generation rate。

### 12.2 Phase 1 Smoke Metrics

必需：

- fall rate。
- near-fall rate。
- base height RMSE。
- velocity RMSE。
- yaw rate RMSE。
- action clipping count。
- invalid command count。
- stop request event count。
- brace request event count。

### 12.3 Skeleton 通过门槛

| 指标 | 门槛 |
| --- | --- |
| `schema_validation_pass_rate` | `100%` |
| `replay_validation_pass_rate` | `100%` |
| `artifact_hash_validation_pass_rate` | `100%` |
| `required_scenario_pass_rate` | `100%` |
| `episode_log_readable_rate` | `100%` |
| `school_sample_envelope_validation_pass_rate` | `100%` |
| `evaluation_report_generated` | `true` |
| `model_manifest_generated` | `true` |

fall rate、velocity RMSE 等运动指标在 skeleton 阶段只记录，不用于声明 locomotion 成功。

## 13. Evaluation Report

```json
{
  "schema_version": "evaluation_report.v0",
  "report_id": "eval_phase1_skeleton_000001",
  "phase": "phase_1_skeleton",
  "abi_version": "abi.northstar.v0",
  "scenario_set_id": "phase1_skeleton_scenarios_v001",
  "env_id": "mock_phase1_env_v0",
  "policy_ids": [
    "debug_noop_v0",
    "debug_random_legal_v0",
    "debug_simple_pd_v0"
  ],
  "summary": {
    "pass": true,
    "episode_count": 24,
    "schema_validation_pass_rate": 1.0,
    "replay_validation_pass_rate": 1.0,
    "school_sample_envelope_validation_pass_rate": 1.0
  },
  "scenario_results": [
    {
      "scenario_id": "phase1_stand_smoke",
      "seed_count": 3,
      "pass": true,
      "failure_reasons": []
    }
  ],
  "artifacts": {
    "run_dir": "runs/run_20260423_000001",
    "report_path": "runs/run_20260423_000001/evaluation_report.json"
  }
}
```

## 14. School Sample Envelope

### 14.1 Segment Types

Skeleton 必须支持：

```text
clean_success_reference
near_failure
fall
invalid_command
action_clip
tracking_error_high
fallback_transition
recovery_success
event_injection
```

### 14.2 Envelope

```json
{
  "schema_version": "school_sample_envelope.v0",
  "sample_id": "sample_000001",
  "source": "phase1_skeleton_eval",
  "phase": "phase_1_skeleton",
  "source_episode_id": "ep_000001",
  "segment_type": "near_failure",
  "step_range": [100, 140],
  "priority": 0.75,
  "labels": {
    "usable_for_training": true,
    "usable_for_release_gate": true,
    "requires_human_review": false
  },
  "artifact_uri": "runs/run_20260423_000001/episodes/ep_000001",
  "metrics": {
    "dangerous_signal_peak": 0.62,
    "base_height_min_m": 0.42
  },
  "data_quality": {
    "schema_valid": true,
    "artifact_hash_valid": true,
    "replay_valid": true
  }
}
```

### 14.3 Priority 初版

```text
priority =
  0.30 * event_severity_score +
  0.25 * replay_relevance_score +
  0.20 * phase1_metric_error_score +
  0.15 * rarity_score +
  0.10 * data_quality_score
```

第一版 priority 不追求完美，只要求可解释、可复算、可测试。

## 15. Model Manifest

debug policy 也必须注册 manifest，避免后续 release package 没有来源。

```json
{
  "schema_version": "model_manifest.v0",
  "model_id": "debug_noop_v0",
  "model_family": "debug_follower_policy",
  "phase": "phase_1_skeleton",
  "abi_version": "abi.northstar.v0",
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "artifact_uri": null,
  "created_at": "2026-04-23T10:00:00+08:00",
  "inputs": ["observation.northstar.v0", "command.northstar.v0"],
  "outputs": ["action.northstar.v0"],
  "status": "debug_only"
}
```

## 16. Error Handling

### 16.1 Schema Error

处理规则：

- validator 返回明确字段路径。
- current episode 标记 `schema_error` termination。
- evaluation report 记录 blocking failure。
- 不生成 usable training sample。

### 16.2 Invalid Command

处理规则：

- command validator 拒绝。
- 写 `invalid_command` event。
- scenario 标记 failed，除非该 scenario 专门测试 invalid command。

### 16.3 Action Clip

处理规则：

- action adapter clip。
- 写 `action_clip` event。
- dangerous signal limit risk 增加。
- school sample builder 可生成 `action_clip` sample。

### 16.4 Replay Mismatch

处理规则：

- replay validation failed。
- evaluation report blocking failure。
- run 不允许作为 release gate 输入。

### 16.5 Artifact Hash Mismatch

处理规则：

- data quality false。
- sample 不允许进入 training 或 release gate。
- 需要重新生成或标记损坏。

## 17. 测试策略

### 17.1 Unit Tests

必须覆盖：

- ABI validators。
- command range validator。
- action adapter clipping。
- dangerous signal rules。
- episode logger writes expected files。
- replay reader reconstructs steps。
- metrics recalculation matches saved metrics。
- school sample envelope validation。

### 17.2 Integration Tests

必须覆盖：

1. 跑 `phase1_stand_smoke`，生成 episode 和 evaluation report。
2. 跑 `event_injection`，生成 near-fall/fall/action_clip 样本。
3. 跑 `log_replay_integrity`，replay metrics 与原始 metrics 一致。
4. 跑 `phase1_school_sample_smoke`，生成至少一个 valid envelope。

### 17.3 Golden Artifacts

建议保留小型 golden run：

```text
tests/fixtures/runs/phase1_skeleton_golden/
```

用途：

- replay reader 回归。
- schema migration 回归。
- school sample builder 回归。

Golden artifact 必须小，不应包含大规模 rollout。

## 18. CLI 入口

第一版建议提供：

```text
northstar-validate-abi
northstar-run-eval
northstar-replay-episode
northstar-build-school-samples
```

示例：

```text
northstar-run-eval --scenario-set configs/eval/phase1_skeleton_scenarios.yaml --output runs/dev
northstar-replay-episode --episode runs/dev/episodes/ep_000001
northstar-build-school-samples --run runs/dev
```

CLI 不是产品接口，只是工程验证入口。

## 19. 与真实 Phase 1 训练的交接

Skeleton 完成后，真实 Phase 1 训练只应替换或扩展：

- `MockPhase1Env` -> Isaac Lab env adapter。
- `DebugFollowerPolicy` -> trainable follower policy。
- smoke metrics -> training/eval metrics。
- mock events -> simulator/physics events。

不应重写：

- ABI validators。
- episode logger。
- replay reader。
- evaluation report。
- school sample envelope。
- model manifest。
- release gate split isolation。

## 20. 实施里程碑

### 20.1 Milestone 1：ABI 与 Config

输出：

- embodiment manifest。
- ABI schema。
- validators。
- scenario config。

验收：

- schema validation tests pass。
- invalid command/action 能被拒绝。

### 20.2 Milestone 2：Mock Env 与 Debug Policy

输出：

- mock env。
- noop/random/simple_pd policy。
- action adapter。

验收：

- reset/step smoke test pass。
- action clipping event pass。

### 20.3 Milestone 3：Logger 与 Replay

输出：

- episode logger。
- replay reader。
- artifact hash。

验收：

- episode 写入完整。
- replay 复算 metrics 一致。

### 20.4 Milestone 4：Evaluation Runner

输出：

- scenario runner。
- metrics accumulator。
- evaluation report。

验收：

- Phase 0/1 skeleton scenario set 100% pass。

### 20.5 Milestone 5：School Sample Builder

输出：

- sample extractor。
- priority scorer。
- envelope writer。

验收：

- failure/success/event segments 能生成 valid school sample envelope。

## 21. 风险与缓解

### 21.1 Mock Env 被误认为训练环境

风险：团队基于 mock env 指标判断 locomotion 成功。

缓解：文档、report 和 model manifest 明确标记 `phase_1_skeleton` 与 `mock_phase1_env_v0`；运动指标只做 smoke，不作为 locomotion gate。

### 21.2 ABI 过早复杂

风险：第一版 schema 试图覆盖所有后续模块，导致实现变慢。

缓解：只实现 Phase 0/1 必需字段；后续字段以 nullable/mask 形式预留。

### 21.3 Logger 格式频繁变化

风险：replay 和 school sample builder 无法稳定。

缓解：所有 artifact 带 schema version；破坏性变化必须通过 schema migration。

### 21.4 Evaluation Runner 与训练框架耦合

风险：后续接入 RSL-RL 后必须重写 runner。

缓解：runner 只依赖 EnvAdapter 和 PolicyAdapter 接口。

### 21.5 School 样本没有数据质量门槛

风险：损坏或不可回放样本进入训练。

缓解：envelope 必须包含 schema/replay/hash data quality；不合格样本不可训练。

## 22. 完成定义

Phase 1 最小可运行骨架完成，需要满足：

1. 定义并实现最小 ABI schema。
2. 定义并实现 Phase 1 skeleton scenario set。
3. Mock Phase 1 env 支持 deterministic reset/step。
4. Debug follower policies 输出合法 action。
5. Action adapter 能 clip 并记录 event。
6. Episode logger 写入 manifest、steps、events、metrics。
7. Replay reader 能读取 episode 并复算 metrics。
8. Evaluation runner 能生成 `evaluation_report.v0`。
9. School sample builder 能生成 valid envelope。
10. `schema_validation_pass_rate = 100%`。
11. `replay_validation_pass_rate = 100%`。
12. `school_sample_envelope_validation_pass_rate = 100%`。
13. 测试覆盖 ABI、mock env、logger、replay、eval runner、school sample builder。
14. 文档明确 skeleton 不声明 locomotion 性能成功。
