# Model Gate、Fallback 与版本切换设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [Phase 1 基础运动底座与学校最小闭环设计](./2026-04-23-phase-1-locomotion-school-loop-design.md)
- [Whole-Body Follower 统一策略设计](./2026-04-23-whole-body-follower-unified-policy-design.md)
- [小脑光轴学习与消融设计](./2026-04-23-cerebellum-light-axis-learning-design.md)

## 1. 目标

本文定义本地模型版本切换、gate blending、fallback、adapter 迁移和回滚反馈的设计。

该规格要解决的问题是：

1. 学校发布新模型后，本地如何在稳定模型、候选模型、adapter、MoE expert 和 fallback policy 之间做连续切换。
2. gate 如何把 `confidence`、`dangerous_sig`、模型分歧、任务风险和能力边界转成可执行权重。
3. fallback 如何保持丝滑，而不是硬切换导致动作突变。
4. 本地 adapter 在学校主模型替换后应保留、迁移、重置还是重新校准。
5. 本地切换、回滚、失败和模型分歧如何回流学校。

该规格适用于 Phase 1 之后的所有阶段：

- Phase 1：stable/candidate follower gate。
- Phase 2/2.5：whole-body follower、MoE 和 stitched baseline 对照中的 gate。
- Phase 3：小脑 generator/selector、light-axis、expert 和 follower 的联合 gate。
- Phase 4：本地大脑语义意图通过小脑时的能力边界与降级。

## 2. 范围

本文覆盖：

- 模型版本关系。
- release state 与本地运行状态。
- gate 输入、输出和默认 blending 公式。
- fallback mode。
- stable/candidate shadow inference。
- adapter 保留、迁移、重置和重新校准策略。
- MoE expert gate。
- dangerous signal 触发条件。
- 本地 gate feedback。
- rollback 与 school release state 回流。

本文不覆盖：

- 学校系统完整训练算法。
- 小脑 generator/selector 网络结构。
- 本地大脑 VLM/VLA/VLN 模型。
- 开放环境安全认证。
- 真实机器人强验收。
- 商业化 OTA 平台。

## 3. 设计原则

### 3.1 稳定模型永远可回退

本地执行路径必须始终保留至少一个 stable model。候选模型、adapter 或 expert 失败时，系统必须能连续退回 stable model 或保守 fallback policy。

### 3.2 Gate 是连续权重，不是模式硬切换

除非触发紧急 stop，gate 应输出连续权重。权重变化必须有平滑约束，避免动作突变。

### 3.3 Candidate 默认不接管

候选模型初始接管比例为 0。必须先通过影子推理、低风险场景和低比例接管，才能逐步扩大权重。

### 3.4 Adapter 不默认跨主模型复用

本地 adapter 绑定 base model 和 ABI。学校主模型替换后，adapter 是否保留必须由兼容性检查和短回放校准决定。

### 3.5 Gate 决策必须可审计

每次权重变化、fallback、回滚和 blocked condition 触发，都必须记录原因、输入信号和受影响模型版本。

## 4. 模型实体

### 4.1 Shared Main Model

学校发布的共享主模型，包括：

- `follower`
- `cerebellum_generator`
- `cerebellum_selector_gate`
- `dangerous_signal_predictor`
- `confidence_calibrator`

共享主模型本地默认冻结，不做完整在线更新。

### 4.2 Local Adapter

本地轻量适配组件：

- LoRA。
- FiLM。
- Adapter。
- task token。
- calibration parameters。

adapter 只用于本地个性化，不直接作为学校 stable main model。

### 4.3 MoE Expert

expert 是可被 gate 调用的 whole-body 或 light-axis 模型。

规则：

- expert 必须声明适用能力边界。
- whole-body expert 必须输出全身 action 或全身 light-axis。
- expert 不能只控制孤立肢体。

### 4.4 Fallback Policy

fallback policy 是风险升高时的保守执行路径。

最小集合：

- `stable_model`
- `brace`
- `stop`
- `conservative_posture`
- `hold_current_safe_pose`

### 4.5 Gate Policy

gate policy 不是单独的高层任务规划器，而是本地执行路径的一部分。它可以由规则、学习模型或小脑 selector 输出，但必须遵守本文的输出契约。

## 5. 版本关系

### 5.1 Version Graph

每个运行模型都属于版本图：

```json
{
  "schema_version": "local_model_graph.v0",
  "active_set_id": "active_set_000001",
  "abi_version": "abi.northstar.v0",
  "stable": {
    "follower_model_id": "follower_stable_phase2_v001",
    "cerebellum_model_id": "cerebellum_stable_phase3_v000"
  },
  "candidate": {
    "follower_model_id": "follower_candidate_phase2_v002",
    "cerebellum_model_id": "cerebellum_candidate_phase3_v001"
  },
  "adapters": [
    {
      "adapter_id": "adapter_local_001",
      "base_model_id": "follower_stable_phase2_v001",
      "status": "active"
    }
  ],
  "experts": [
    {
      "expert_id": "expert_locomotion_v001",
      "model_id": "follower_stable_phase1_v003",
      "role": "locomotion_expert"
    }
  ],
  "fallback": {
    "primary": "stable_model",
    "secondary": ["brace", "stop", "conservative_posture"]
  }
}
```

规则：

- `stable` 不能为空。
- `candidate` 可以为空。
- 每个 adapter 必须绑定 `base_model_id`。
- 每个 expert 必须声明 role 和 capability boundary。
- active set 的 ABI version 必须与 release package 兼容。

### 5.2 Release State 与 Local Runtime State

学校 release state：

```text
draft
candidate
staged
stable
rejected
rolled_back
archived
```

本地 runtime state：

```text
downloaded
verified
shadow
limited_active
active
paused
rolled_back
removed
```

映射规则：

- 学校 `candidate` 到本地后只能进入 `downloaded -> verified -> shadow`。
- 学校 `staged` 允许本地进入 `limited_active`。
- 学校 `stable` 允许本地进入 `active`，但仍保留上一 stable 作为短期 fallback。
- 学校 `rejected` 不能被本地自动拉取。
- 学校 `rolled_back` 触发本地暂停或回滚。

## 6. Gate 输入

Gate 每个 control step 或 gate step 读取以下输入。

### 6.1 状态与命令

- `obs_public`
- `active_command`
- `command_mask`
- 当前任务族。
- 当前 phase。
- morphology id。

### 6.2 模型输出

- stable model output。
- candidate model output。
- active adapter output 或 adapter delta。
- expert outputs。
- cerebellum selected light-axis。
- follower action。

### 6.3 风险信号

- `confidence.overall`
- `confidence.command_feasibility`
- `confidence.fallback_readiness`
- `dangerous_sig.overall_risk`
- `dangerous_sig.fall_risk`
- `dangerous_sig.collision_risk`
- `dangerous_sig.overload_risk`
- `dangerous_sig.unreachable_risk`
- `dangerous_sig.model_disagreement`
- `dangerous_sig.low_confidence`

### 6.4 能力边界

- release package blocked conditions。
- capability summary。
- known failure modes。
- task family allowlist。
- command range allowlist。
- phase-specific release gate status。

## 7. Gate 输出

Gate 输出必须统一成以下结构：

```json
{
  "schema_version": "gate_decision.v0",
  "decision_id": "gate_000001",
  "step_index": 123,
  "weights": {
    "stable_model": 1.0,
    "candidate_model": 0.0,
    "adapter": 1.0,
    "experts": []
  },
  "fallback": {
    "mode": "none",
    "intensity": 0.0,
    "target": null
  },
  "blocked": {
    "candidate_blocked": false,
    "adapter_blocked": false,
    "expert_blocked": []
  },
  "reasons": [],
  "signals": {
    "confidence": 1.0,
    "overall_risk": 0.0,
    "model_disagreement": 0.0
  }
}
```

规则：

- 权重必须在 `[0.0, 1.0]`。
- stable/candidate 权重相加不一定等于 1，因为 adapter 和 expert 可以作为附加调制，但最终 action 或 light-axis 混合必须归一化。
- `fallback.mode != none` 时，candidate weight 应下降。
- `reasons` 必须记录触发条件，例如 `low_confidence`、`fall_risk_high`、`candidate_shadow_only`。

## 8. 默认 Gate Blending

### 8.1 Candidate 权重上限

candidate 的理论上限由 release package 给出：

```text
w_candidate_max = release.gate_recommendation.max_takeover_ratio
```

Phase 默认：

- Phase 1：`0.25`
- Phase 2/2.5：`0.25`
- Phase 3：`0.25`
- stable 后短期 active：`0.75`
- 成为新 stable 后：`1.0`

### 8.2 风险门控

风险缩放：

```text
risk_block =
  max(
    fall_risk,
    collision_risk,
    overload_risk,
    unreachable_risk,
    low_confidence,
    model_disagreement
  )

risk_scale = clamp(1.0 - risk_block, 0.0, 1.0)
```

### 8.3 置信度缩放

```text
confidence_scale = clamp((confidence_overall - c_min) / (c_good - c_min), 0.0, 1.0)

c_min = 0.4
c_good = 0.8
```

### 8.4 Shadow/Limited Active 进度

```text
stage_scale =
  0.0   if runtime_state == "shadow"
  0.25  if runtime_state == "limited_active"
  0.75  if runtime_state == "active" and release_state == "staged"
  1.0   if release_state == "stable"
```

### 8.5 Candidate 权重

```text
w_candidate_raw =
  w_candidate_max
  * risk_scale
  * confidence_scale
  * stage_scale
  * task_allow_scale

w_candidate_target = clamp(w_candidate_raw, 0.0, w_candidate_max)
```

`task_allow_scale`：

- allowed task family：`1.0`
- risky but allowed：`0.5`
- blocked：`0.0`

### 8.6 时间平滑

```text
w_candidate_t =
  w_candidate_{t-1}
  + clamp(
      w_candidate_target - w_candidate_{t-1},
      -delta_down_max,
      delta_up_max
    )
```

推荐：

- `delta_up_max = 0.02 per gate step`
- `delta_down_max = 0.10 per gate step`

下降可以比上升快，但不能瞬间硬切，除非 emergency stop。

### 8.7 Stable 权重

```text
w_stable = 1.0 - w_candidate_t
```

如果 MoE experts 参与 action/light-axis 混合，则 stable/candidate 先形成 base output，再与 experts 经 expert gate 混合。

## 9. Output Blending

### 9.1 Action Blending

对 follower action：

```text
action_blend =
  w_stable    * action_stable
+ w_candidate * action_candidate
```

然后执行：

- action clipping。
- PD/servo mapping。
- event logging。

规则：

- blending 只允许在同一 action schema 和同一 joint order 下进行。
- 如果 schema hash 不一致，candidate weight 必须为 0。
- clipping 后必须记录 candidate/stable 各自是否越界。

### 9.2 Light-Axis Blending

对小脑 light-axis：

```text
axis_blend.pose =
  blend_pose(axis_stable.pose, axis_candidate.pose, w_candidate)

axis_blend.velocity =
  w_stable * axis_stable.velocity
+ w_candidate * axis_candidate.velocity

axis_blend.torque_pose =
  blend_torque_pose(axis_stable.torque_pose, axis_candidate.torque_pose, w_candidate)
```

confidence 与 risk：

```text
confidence_blend = min(confidence_stable, weighted_confidence)
risk_blend = max(
  risk_stable,
  risk_candidate,
  model_disagreement,
  fallback_trigger_risk
)
```

规则：

- risk 不能被权重稀释或平均掩盖，应取偏保守组合。
- rotation/pose blending 必须使用连续表示，避免四元数符号跳变。
- light-axis blending 后仍需 follower feasibility check。

### 9.3 Expert Blending

MoE experts 输出全身 action 或 light-axis。

```text
expert_weight_i =
  softmax(score_i / temperature)
```

约束：

- `max_delta_expert_weight_per_step` 限制 expert 权重变化。
- expert 权重低于 `0.05` 可以视为 inactive。
- `expert_switch_instability` 高时，降低 temperature 或进入 stable fallback。

默认：

- `temperature = 1.0`
- `max_delta_expert_weight_per_step = 0.05`

## 10. Fallback Mode

### 10.1 Fallback 层级

Fallback 分成五级：

| 级别 | 模式 | 说明 |
| --- | --- | --- |
| 0 | `none` | 正常执行 |
| 1 | `stable_model` | 退回稳定模型 |
| 2 | `conservative_posture` | 降低任务激进度，维持保守姿态 |
| 3 | `brace` | 抗扰或高风险姿态 |
| 4 | `stop` | 安全停止 |
| 5 | `emergency_terminate` | 仿真终止或真实系统急停接口 |

Phase 1/2/3 默认只自动使用 0-4。`emergency_terminate` 由仿真 termination 或真实安全系统处理。

### 10.2 Fallback 触发

默认触发：

```text
if fall_risk > 0.7:
  fallback = brace
elif collision_risk > 0.7:
  fallback = conservative_posture
elif overload_risk > 0.7:
  fallback = stable_model
elif unreachable_risk > 0.6:
  fallback = stable_model
elif confidence < 0.4:
  fallback = stable_model
elif model_disagreement > 0.6:
  fallback = stable_model
else:
  fallback = none
```

### 10.3 Fallback 退出

退出 fallback 必须满足 hysteresis：

```text
exit_allowed =
  risk_below_exit_threshold for N consecutive steps
  and confidence_above_exit_threshold
  and command_feasibility_above_exit_threshold
```

默认：

- `N = 20 gate steps`
- `risk_exit_threshold = 0.3`
- `confidence_exit_threshold = 0.65`
- `command_feasibility_exit_threshold = 0.6`

### 10.4 Fallback 平滑

Fallback intensity：

```text
fallback_intensity_target = risk_to_intensity(overall_risk)
fallback_intensity_t =
  fallback_intensity_{t-1}
  + clamp(
      fallback_intensity_target - fallback_intensity_{t-1},
      -0.05,
      0.10
    )
```

规则：

- 进入 fallback 可以快于退出。
- brace/stop 可以覆盖 candidate weight。
- fallback 退出必须记录事件。

## 11. Adapter 兼容与迁移

### 11.1 Adapter Metadata

每个 adapter 必须记录：

```json
{
  "schema_version": "adapter_manifest.v0",
  "adapter_id": "adapter_local_001",
  "adapter_type": "lora",
  "base_model_id": "follower_stable_phase2_v001",
  "abi_version": "abi.northstar.v0",
  "input_schema_hash": "sha256:...",
  "output_schema_hash": "sha256:...",
  "trained_on": {
    "client_id": "robot_or_sim_001",
    "sample_ids": []
  },
  "status": "active"
}
```

### 11.2 主模型替换后的策略

学校主模型替换后，adapter 有四种策略：

```text
retain
migrate
reset
recalibrate
```

选择规则：

- `retain`：base model id 相同，schema hash 相同，评测通过。
- `migrate`：base model 变更但架构兼容，有 adapter migration map。
- `reset`：schema 或层结构不兼容，或回放评测失败。
- `recalibrate`：结构兼容但性能不确定，需要短回放和少量本地 warmup。

默认策略：

```text
if base_model_id unchanged and schema_hash unchanged:
  retain
elif migration_map exists and replay_check passes:
  migrate
elif schema_hash compatible and replay_check uncertain:
  recalibrate
else:
  reset
```

### 11.3 Adapter Gate

adapter 也需要 gate：

```text
w_adapter =
  1.0 if adapter_status == active and adapter_confidence >= 0.7
  0.5 if adapter_status == recalibrating
  0.0 if adapter_status in [reset, blocked]
```

adapter 输出必须可关闭。本地不能因为 adapter 失败而破坏 stable fallback。

## 12. Shadow Inference

### 12.1 Shadow 模式

shadow 模式下：

- stable 控制环境。
- candidate 只推理。
- adapter 可记录预测但不生效。
- gate 计算假设权重但不执行 candidate action。

记录：

- action disagreement。
- light-axis disagreement。
- confidence difference。
- dangerous signal difference。
- candidate would-have-fallback events。

### 12.2 Disagreement 指标

Action disagreement：

```text
action_disagreement =
  mean_j abs(action_candidate_j - action_stable_j) / action_scale_j
```

Light-axis disagreement：

```text
axis_disagreement =
  w_pose * pose_distance
+ w_vel  * velocity_distance
+ w_tp   * torque_pose_distance
+ w_risk * risk_delta
```

模型分歧：

```text
model_disagreement =
  clamp(max(action_disagreement, axis_disagreement), 0.0, 1.0)
```

### 12.3 Shadow 通过条件

candidate 从 `shadow` 进入 `limited_active` 需要：

- schema validation pass。
- no NaN/Inf。
- model disagreement 低于门槛。
- candidate dangerous signal 不显著高于 stable。
- release gate replay pass。
- no blocked condition violation。

默认：

- `mean_model_disagreement < 0.25`
- `p95_model_disagreement < 0.5`
- `candidate_dangerous_peak <= stable_dangerous_peak + 0.1`

## 13. Runtime State Machine

本地 runtime 状态机：

```text
downloaded
  -> verified
  -> shadow
  -> limited_active
  -> active
  -> paused
  -> rolled_back
```

状态转换：

- `downloaded -> verified`：hash、ABI、schema、manifest 通过。
- `verified -> shadow`：release package 允许 shadow。
- `shadow -> limited_active`：shadow 通过。
- `limited_active -> active`：低比例接管通过，学校或本地策略允许升级。
- `active -> paused`：风险升高但未触发正式回滚。
- `paused -> shadow`：重新验证。
- `paused -> rolled_back`：回滚条件满足。
- `limited_active/active -> rolled_back`：严重风险、回归或学校 rolled_back。

## 14. Rollback

### 14.1 本地回滚触发

本地触发 rollback：

- fall rate 超过 stable。
- near-fall rate 超过门槛。
- dangerous_sig peak 超过门槛。
- fallback_count 超过门槛。
- hard switch count 超过门槛。
- model disagreement 持续高。
- adapter migration 导致回放退化。
- school release state 变为 `rolled_back`。

### 14.2 Rollback 行为

回滚步骤：

1. candidate weight 平滑降为 0，除非 emergency。
2. adapter weight 按策略降为 0 或进入 recalibrate。
3. expert gate 回到 stable expert set。
4. fallback mode 进入 `stable_model` 或 `conservative_posture`。
5. 记录 rollback event。
6. 生成 gate feedback。
7. 上传 high-priority samples。

### 14.3 Rollback Report

```json
{
  "schema_version": "rollback_report.v0",
  "rollback_id": "rollback_000001",
  "client_id": "sim_client_001",
  "stable_model_id": "follower_stable_phase2_v001",
  "candidate_model_id": "follower_candidate_phase2_v002",
  "adapter_id": "adapter_local_001",
  "trigger": "fallback_count_threshold",
  "runtime_state_before": "limited_active",
  "metrics": {
    "fallback_count": 18,
    "dangerous_sig_peak": 0.72,
    "mean_model_disagreement": 0.38
  },
  "sample_refs": ["sample_000001"],
  "action_taken": "candidate_weight_to_zero"
}
```

## 15. Gate Feedback

Gate feedback 扩展学校系统定义：

```json
{
  "schema_version": "gate_feedback.v0",
  "feedback_id": "gate_feedback_000001",
  "client_id": "sim_client_001",
  "phase": "phase_3",
  "active_set_id": "active_set_000001",
  "stable_model_ids": {
    "follower": "follower_stable_phase2_v001",
    "cerebellum": "cerebellum_stable_phase3_v000"
  },
  "candidate_model_ids": {
    "follower": "follower_candidate_phase2_v002",
    "cerebellum": "cerebellum_candidate_phase3_v001"
  },
  "adapter_ids": ["adapter_local_001"],
  "summary": {
    "shadow_steps": 10000,
    "active_steps": 2000,
    "fallback_count": 12,
    "rollback_triggered": false,
    "mean_candidate_weight": 0.18,
    "max_candidate_weight": 0.25,
    "hard_switch_count": 0
  },
  "metrics": {
    "mean_model_disagreement": 0.21,
    "p95_model_disagreement": 0.44,
    "dangerous_sig_peak": 0.42,
    "confidence_min": 0.51,
    "fallback_recovery_success_rate": 0.98
  },
  "events": [
    {
      "event_type": "fallback_entered",
      "count": 12
    }
  ],
  "sample_refs": ["sample_000001"]
}
```

规则：

- gate feedback 必须记录 adapter ids。
- hard switch count 必须显式记录。
- sample refs 应指向学校样本或本地待上传样本。

## 16. Release Package 扩展

Release package 需要声明本地 gate 策略。

```json
{
  "schema_version": "release_package.v0",
  "release_id": "release_phase3_cerebellum_candidate_v001",
  "phase": "phase_3",
  "model_ids": {
    "follower": "follower_stable_phase2_v001",
    "cerebellum": "cerebellum_candidate_phase3_v001"
  },
  "fallback": {
    "required": true,
    "stable_follower_model_id": "follower_stable_phase2_v001",
    "stable_cerebellum_model_id": "cerebellum_stable_phase3_v000",
    "fallback_modes": ["stable_model", "brace", "stop", "conservative_posture"]
  },
  "adapter_policy": {
    "default": "recalibrate",
    "allowed": ["retain", "migrate", "reset", "recalibrate"],
    "requires_replay_check": true
  },
  "gate_recommendation": {
    "initial_candidate_weight": 0.0,
    "max_candidate_weight": 0.25,
    "delta_up_max": 0.02,
    "delta_down_max": 0.10,
    "blocked_conditions": [
      "fall_risk > 0.5",
      "confidence < 0.4",
      "model_disagreement > 0.6",
      "unreachable_risk > 0.5"
    ],
    "allowed_task_families": ["locomotion", "whole_body", "light_axis"]
  }
}
```

## 17. 学校状态回流

学校根据 gate feedback 更新 release state。

### 17.1 Candidate -> Staged

条件：

- shadow feedback 足够。
- model disagreement 可控。
- dangerous signal 不高于 stable。
- release gate 样本未发现严重退化。

### 17.2 Staged -> Stable

条件：

- 多个客户端反馈稳定。
- fallback_count 在门槛内。
- 没有严重 rollback。
- capability summary 已更新。
- offline 高保真样本没有阻断级失败。

### 17.3 Staged/Stable -> Rolled Back

条件：

- rollback rate 超过门槛。
- 发现未覆盖严重 failure。
- dangerous false negative 明显升高。
- adapter migration 大范围失败。
- school regression 重新评测失败。

## 18. 指标

### 18.1 Gate 指标

- mean candidate weight。
- max candidate weight。
- candidate weight rise time。
- candidate weight drop time。
- hard switch count。
- gate temporal smoothness。
- blocked condition count。

### 18.2 Fallback 指标

- fallback count。
- fallback duration。
- fallback recovery success rate。
- fallback abruptness score。
- fallback exit failure rate。
- brace success rate。
- stop success rate。

### 18.3 版本指标

- shadow pass rate。
- limited active pass rate。
- rollback rate。
- staged to stable rate。
- adapter retain/migrate/reset/recalibrate rate。
- candidate regression rate。

### 18.4 风险指标

- dangerous_sig peak under candidate。
- dangerous false negative rate。
- model disagreement p95。
- confidence min。
- unreachable blocked rate。

## 19. Online 验收门槛

默认门槛：

| 指标 | 门槛 |
| --- | --- |
| `hard_switch_count_per_min` | `< 1` |
| `fallback_recovery_success_rate` | `> 95%` |
| `fallback_abruptness_score` | `< 0.2` |
| `mean_model_disagreement_shadow` | `< 0.25` |
| `p95_model_disagreement_shadow` | `< 0.5` |
| `dangerous_false_negative_rate` | `< 5%` |
| `rollback_report_generated` | `100% when rollback occurs` |
| `gate_feedback_generated` | `100% for candidate runs` |
| `adapter_policy_recorded` | `100% when adapter exists` |
| `schema_validation_pass_rate` | `100%` |

## 20. Offline 分支

offline 分支验证：

- gate 在高保真动力学下是否仍平滑。
- fallback 在执行器延迟下是否及时。
- adapter 迁移是否引入真实退化。
- MoE expert 切换是否产生不可接受动作突变。
- dangerous signal 是否迟报。

offline failure 回流为：

- `fallback_abrupt`
- `expert_switch_instability`
- `adapter_migration_failure`
- `dangerous_signal_late`
- `candidate_regression`
- `sim_to_real_gate_bias`

这些样本默认进入 release gate。

## 21. 与其他子规格的交接

### 21.1 与 School System

本规格补充学校系统中的：

- release package gate recommendation。
- gate feedback。
- rollback report。
- adapter policy。
- release state 更新条件。

### 21.2 与小脑光轴

小脑 selector/gate 输出可以直接采用 `gate_decision.v0`。Phase 3 的 light-axis command 中 `gate` 字段应与本文保持一致。

### 21.3 与本地大脑语义意图

Phase 4 本地大脑需要读取：

- capability summary。
- blocked conditions。
- fallback modes。
- known failure modes。

本地大脑不直接覆盖 gate 决策，只能通过语义意图和风险偏好影响小脑输入。

## 22. 推荐文件布局

后续实现时，建议增加：

```text
configs/
  gate/
    gate_defaults.yaml
    phase1_gate.yaml
    phase3_gate.yaml
  release/
    release_state_machine.yaml
    adapter_policy.yaml

src/
  northstar/
    gate/
      decision.py
      blending.py
      fallback.py
      shadow.py
      adapter_policy.py
      state_machine.py
    registry/
      local_model_graph.py
      release_package.py
    school/
      gate_feedback.py
      rollback_report.py

tests/
  gate/
    test_gate_blending.py
    test_fallback_hysteresis.py
    test_shadow_disagreement.py
    test_adapter_policy.py
    test_runtime_state_machine.py
  school/
    test_gate_feedback_schema.py
    test_rollback_report_schema.py
```

## 23. 风险与缓解

### 23.1 Gate 过于保守

风险：candidate 永远无法接管，学校更新无法产生收益。

缓解：记录 blocked condition count，按任务族分析保守原因；允许低风险场景逐步提高上限。

### 23.2 Gate 过于激进

风险：candidate 过早接管导致跌倒或动作突变。

缓解：candidate 默认 shadow，stage scale 限制，risk scale 和 fallback hysteresis。

### 23.3 Adapter 迁移污染新模型

风险：旧 adapter 在新主模型上造成退化。

缓解：adapter 默认 recalibrate；无 replay check 不允许 retain/migrate。

### 23.4 Blending 掩盖风险

风险：风险信号被加权平均后看起来很低。

缓解：risk 使用 max 或保守组合，不用简单平均。

### 23.5 Expert 切换不连续

风险：MoE expert 权重变化造成动作跳变。

缓解：expert weight delta 限制、switch continuity 指标、fallback abruptness release gate。

### 23.6 回滚没有足够证据

风险：学校无法复现本地回滚原因。

缓解：rollback report 必须包含 metrics、trigger、sample_refs 和 active_set_id。

## 24. 完成定义

Gate/Fallback 设计完成需要满足：

1. 定义 `local_model_graph.v0`。
2. 定义 `gate_decision.v0`。
3. 定义默认 gate blending 公式。
4. 定义 fallback mode、触发和退出 hysteresis。
5. 定义 adapter retain/migrate/reset/recalibrate 策略。
6. 定义 shadow inference 与 model disagreement 指标。
7. 定义 runtime state machine。
8. 定义 rollback report。
9. 定义 gate feedback 扩展。
10. 定义 release package gate 扩展。
11. 明确学校 release state 如何根据本地反馈更新。
