# 本地大脑语义意图接口设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [小脑光轴学习与消融设计](./2026-04-23-cerebellum-light-axis-learning-design.md)
- [Model Gate、Fallback 与版本切换设计](./2026-04-23-model-gate-fallback-design.md)

## 1. 目标

本文定义 Phase 4 本地大脑与小脑之间的语义意图接口。

本地大脑的定位是轻量多模态智能核心。它负责理解任务、场景、对象、语言或视觉输入，并生成高维语义意图。它不直接输出关节命令，不直接输出 follower action，也不覆盖小脑 selector/gate 的最终决策。

Phase 4 的目标是验证：

1. 本地大脑能把任务和场景理解转化为结构化语义意图。
2. 语义意图能被小脑 generator/selector 消化，转化为候选光轴和 gate 决策。
3. 本地大脑能读取小脑、follower 和学校系统提供的能力边界，避免生成明显不可执行意图。
4. 执行失败、能力不足和降级原因能回流给本地大脑和学校系统。
5. 本地大脑仍不作为学校系统的主替换目标；学校主线继续训练小脑 + follower。

## 2. 范围

本文覆盖：

- 本地大脑输入上下文。
- 高维语义意图 schema。
- 本地大脑到小脑的接口。
- 小脑/follower 到本地大脑的能力边界和执行反馈。
- Phase 4 任务集和评测指标。
- 语义意图失败样本如何进入学校系统。
- 云端大脑、学校系统和本地大脑之间的弱耦合关系。

本文不覆盖：

- 具体 VLM/VLA/VLN 模型选型。
- 视觉编码器或语言模型训练细节。
- 开放环境安全认证。
- 完整安全仲裁状态机。
- 真实机器人强验收。
- 云端大脑规划算法。
- 用户产品交互系统。

Phase 4 仍面向封闭仿真或受控任务环境，不以开放场景安全认证为目标。

## 3. 设计原则

### 3.1 本地大脑生成意图，不生成动作

本地大脑输出目标、约束、偏好、风险容忍和关键提示。它不输出低层关节目标，也不直接输出完整中层轨迹。

### 3.2 小脑负责运动可能性

小脑负责把语义意图变成候选光轴，并通过 selector/gate 选择、混合、fallback。本地大脑不能绕过小脑直接控制 follower。

### 3.3 能力边界必须进入意图生成

本地大脑必须读取 capability summary、known failure modes、blocked conditions 和当前 gate/fallback 状态。语义意图生成应受这些能力边界约束。

### 3.4 先封闭场景，再开放安全

Phase 4 不引入完整安全仲裁系统。它只定义能力边界、拒绝/降级反馈和失败回流。开放环境安全体系后续单独设计。

### 3.5 失败是学校系统数据

语义意图失败不是单纯 task failure，它是学校系统学习能力边界、小脑训练数据和本地大脑提示约束的来源。

## 4. 模块边界

### 4.1 Local Brain

职责：

- 理解任务输入。
- 解析场景和对象。
- 维护短期任务状态。
- 读取能力边界。
- 生成语义意图。
- 接收执行反馈并修正后续意图。

不负责：

- 低层运动控制。
- 直接 action blending。
- 直接覆盖 fallback。
- 学校模型发布。

### 4.2 Cerebellum

职责：

- 接收语义意图 hint。
- 生成候选光轴。
- 选择、混合和 fallback。
- 返回可执行性、拒绝原因、降级原因和能力边界反馈。

### 4.3 Whole-Body Follower

职责：

- 执行小脑输出的 light-axis command。
- 返回执行状态、tracking error、危险信号和 fallback 事件。

### 4.4 School System

职责：

- 收集语义意图到执行失败的样本。
- 更新小脑/follower capability summary。
- 向本地大脑和云端大脑提供能力边界摘要。

不负责：

- 在当前蓝图中直接训练本地大脑。

### 4.5 Cloud Brain

职责：

- 提供高层计划、任务约束或目标结构。
- 读取学校输出的能力摘要，避免规划超出能力边界。

不负责：

- 直接控制小脑或 follower。
- 直接参与学校训练循环。

## 5. 本地大脑输入

Phase 4 本地大脑输入包括任务输入、场景输入、机器人状态摘要和能力边界。

### 5.1 Task Input

```json
{
  "schema_version": "task_input.v0",
  "task_id": "task_000001",
  "source": "user_or_cloud_brain",
  "task_type": "go_reach_hover",
  "natural_language": "走到桌边，把左手靠近杯子但不要碰到桌面",
  "structured_goal": {
    "target_object": "cup_001",
    "target_relation": "near",
    "contact_policy": "pre_contact_hover"
  },
  "priority": 0.6,
  "deadline_s": null
}
```

### 5.2 Scene Context

Phase 4 可使用仿真真值或结构化 mock perception，不要求完整视觉系统。

```json
{
  "schema_version": "scene_context.v0",
  "frame": "base",
  "objects": [
    {
      "object_id": "cup_001",
      "class": "cup",
      "position_base_m": [0.65, 0.20, 0.85],
      "size_m": [0.08, 0.08, 0.12],
      "affordances": ["approach", "hover_near"],
      "confidence": 1.0
    }
  ],
  "obstacles": [
    {
      "object_id": "table_edge_001",
      "class": "table_edge",
      "position_base_m": [0.45, 0.10, 0.75],
      "radius_m": 0.10,
      "avoidance_weight": 1.0
    }
  ],
  "free_space_hints": []
}
```

### 5.3 Robot Capability Context

来自学校、小脑和 gate/fallback：

```json
{
  "schema_version": "robot_capability_context.v0",
  "active_models": {
    "cerebellum_model_id": "cerebellum_stable_phase3_v001",
    "follower_model_id": "follower_stable_phase2_v001"
  },
  "capability_summary_ids": [
    "cap_cerebellum_phase3_v001",
    "cap_follower_phase2_v001"
  ],
  "supported_capabilities": [
    {
      "name": "walk_reach_hover",
      "conditions": {
        "max_velocity_m_s": 0.6,
        "max_reach_distance_m": 0.55,
        "requires_flat_terrain": true
      },
      "confidence": 0.78
    }
  ],
  "blocked_conditions": [
    "target_reach_distance_m > 0.65",
    "fall_risk > 0.5",
    "confidence < 0.4"
  ],
  "known_failure_modes": [
    {
      "failure_type": "high_yaw_reach_instability",
      "condition": "abs(target_yaw_rate_rad_s) > 0.8 and reach_weight > 0.7"
    }
  ],
  "current_gate_state": {
    "fallback_mode": "none",
    "candidate_weight": 0.0,
    "confidence": 0.82,
    "overall_risk": 0.12
  }
}
```

### 5.4 Execution Feedback

本地大脑接收上一轮执行反馈：

```json
{
  "schema_version": "execution_feedback.v0",
  "intent_id": "intent_000001",
  "status": "degraded",
  "progress": 0.45,
  "cerebellum_response": {
    "accepted": true,
    "degraded": true,
    "degrade_reason": "reach_distance_near_limit",
    "selected_fallback_mode": "conservative_posture"
  },
  "tracking": {
    "task_error": 0.22,
    "wrist_position_error_m": 0.09,
    "velocity_error_m_s": 0.12
  },
  "risk": {
    "overall_risk_peak": 0.46,
    "dangerous_events": ["near_unreachable"]
  }
}
```

## 6. 语义意图输出

本地大脑输出 `semantic_intent.v0`。

### 6.1 顶层结构

```json
{
  "schema_version": "semantic_intent.v0",
  "intent_id": "intent_000001",
  "source": "local_brain",
  "task_id": "task_000001",
  "created_at_step": 1200,
  "valid_for_s": 0.5,
  "goal": {},
  "motion_preference": {},
  "force_preference": {},
  "avoidance": {},
  "contact_policy": {},
  "object_constraints": {},
  "spatial_constraints": {},
  "risk_policy": {},
  "priority": {},
  "fallback_preference": {},
  "explain": {}
}
```

### 6.2 Goal

```json
{
  "goal": {
    "goal_type": "approach_and_hover",
    "target_object_id": "cup_001",
    "target_body": "left_wrist",
    "target_relation": "near",
    "target_position_base_m": [0.58, 0.18, 0.82],
    "position_tolerance_m": 0.06,
    "orientation_preference": {
      "enabled": true,
      "rotation_6d_base": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      "weight": 0.2
    }
  }
}
```

### 6.3 Motion Preference

```json
{
  "motion_preference": {
    "speed_preference": "slow",
    "max_base_velocity_m_s": 0.35,
    "max_yaw_rate_rad_s": 0.4,
    "smoothness_weight": 0.8,
    "urgency": 0.2,
    "body_posture_preference": "stable_upright"
  }
}
```

### 6.4 Force Preference

```json
{
  "force_preference": {
    "mode": "light",
    "max_contact_force_n": 5.0,
    "impedance_preference": "soft",
    "torque_pose_bias": "conservative"
  }
}
```

Phase 4 不要求真实接触力控制精确可用，但意图接口必须保留力度偏好，供小脑和 follower 做保守约束。

### 6.5 Avoidance

```json
{
  "avoidance": {
    "keypoints_base_m": [
      {
        "name": "table_edge_001",
        "position_base_m": [0.45, 0.10, 0.75],
        "radius_m": 0.10,
        "weight": 1.0
      }
    ],
    "body_parts": ["left_wrist", "left_forearm", "torso"],
    "violation_tolerance": 0.0
  }
}
```

### 6.6 Contact Policy

```json
{
  "contact_policy": {
    "mode": "pre_contact_hover",
    "allowed_contact": [],
    "forbidden_contact": ["table_edge_001", "cup_001"],
    "hover_distance_m": 0.05,
    "max_end_effector_speed_near_contact_m_s": 0.1
  }
}
```

### 6.7 Object Constraints

```json
{
  "object_constraints": {
    "target_object_id": "cup_001",
    "object_confidence_required": 0.7,
    "if_lost": "hold_or_replan",
    "relative_approach_direction_base": [1.0, 0.0, 0.0]
  }
}
```

### 6.8 Spatial Constraints

```json
{
  "spatial_constraints": {
    "workspace_hint": "front_left_reachable",
    "max_reach_distance_m": 0.55,
    "keep_base_stable": true,
    "allow_step_adjustment": true,
    "forbidden_zones": []
  }
}
```

### 6.9 Risk Policy

```json
{
  "risk_policy": {
    "risk_tolerance": 0.2,
    "prefer_degrade_over_abort": true,
    "allow_fallback_modes": ["stable_model", "conservative_posture", "brace", "stop"],
    "blocked_if": [
      "fall_risk > 0.5",
      "confidence < 0.4",
      "target_reach_distance_m > 0.65"
    ]
  }
}
```

### 6.10 Priority

```json
{
  "priority": {
    "task_priority": 0.6,
    "locomotion_priority": 0.8,
    "manipulation_priority": 0.5,
    "stability_priority": 1.0,
    "comfort_or_smoothness_priority": 0.7
  }
}
```

### 6.11 Fallback Preference

```json
{
  "fallback_preference": {
    "on_unreachable": "degrade_goal",
    "on_high_risk": "brace_or_stop",
    "on_object_lost": "hold_or_replan",
    "on_low_confidence": "request_clarification_or_replan"
  }
}
```

### 6.12 Explain

`explain` 用于调试和学校回流，不作为控制依据。

```json
{
  "explain": {
    "intent_summary": "approach cup with left wrist and hover before contact",
    "capability_constraints_used": [
      "max_reach_distance_m <= 0.55",
      "max_base_velocity_m_s <= 0.35"
    ],
    "known_risks": [
      "target near reach limit"
    ]
  }
}
```

## 7. 本地大脑到小脑映射

本地大脑不直接生成 light-axis。它生成 `semantic_intent`，由小脑 Context Encoder 接收。

### 7.1 小脑输入扩展

Phase 4 集成时，小脑输入在 Phase 3 光轴接口基础上增加：

```json
{
  "semantic_intent_hint": {
    "intent_id": "intent_000001",
    "goal_type": "approach_and_hover",
    "target_body": "left_wrist",
    "target_position_base_m": [0.58, 0.18, 0.82],
    "speed_preference": "slow",
    "force_preference": "light",
    "avoidance_keypoints_base_m": [],
    "risk_tolerance": 0.2,
    "fallback_preference": "degrade_goal"
  }
}
```

### 7.2 映射责任

本地大脑负责：

- 选择目标对象和目标关系。
- 设定速度、力度、避让和风险偏好。
- 根据能力边界降低目标激进程度。

小脑负责：

- 判断语义意图是否可执行。
- 生成候选光轴。
- 选择和 fallback。
- 输出 structured light-axis command。

### 7.3 拒绝和降级

小脑可以拒绝或降级本地大脑意图。

```json
{
  "schema_version": "intent_response.v0",
  "intent_id": "intent_000001",
  "accepted": true,
  "degraded": true,
  "rejected": false,
  "reason": "target_near_reach_limit",
  "applied_changes": {
    "max_base_velocity_m_s": 0.25,
    "target_position_base_m": [0.54, 0.18, 0.82],
    "fallback_mode": "conservative_posture"
  },
  "capability_feedback": {
    "reachable_confidence": 0.62,
    "risk_estimate": 0.38,
    "suggested_replan": false
  }
}
```

规则：

- `rejected=true` 时，小脑不应执行该意图。
- `degraded=true` 时，本地大脑必须记录实际执行目标。
- 拒绝或降级必须进入 episode log 和学校样本。

## 8. 能力边界反馈

本地大脑需要读取三类能力边界。

### 8.1 Static Capability Summary

来自学校系统，随模型版本更新。

包含：

- 支持能力。
- 条件范围。
- 风险区域。
- known failure modes。
- 推荐 planning constraints。

### 8.2 Runtime Capability Feedback

来自小脑/gate 当前状态。

包含：

- 当前 confidence。
- 当前 dangerous risk。
- fallback mode。
- candidate/stable weight。
- model disagreement。
- command feasibility。

### 8.3 Intent Response Feedback

来自小脑对当前意图的接受、降级或拒绝。

用途：

- 本地大脑调整下一步意图。
- 本地大脑降低速度、力度或 reach 距离。
- 本地大脑请求云端重新规划。
- 学校系统记录能力边界样本。

## 9. Phase 4 任务集

Phase 4 任务集以语义意图闭环为主，不追求开放世界泛化。

### 9.1 Go-To Target

任务：根据目标点或对象位置，生成接近目标的语义意图。

验证：

- 本地大脑生成速度和目标偏好。
- 小脑生成 locomotion 光轴。
- follower 稳定执行。

### 9.2 Reach Near Object

任务：让指定手腕靠近对象。

验证：

- 本地大脑选择 target body。
- 本地大脑设置 reach tolerance。
- 小脑判断可达性。
- follower 执行 reach。

### 9.3 Pre-Contact Hover

任务：靠近对象但不接触。

验证：

- contact policy 正确生成。
- hover distance 被小脑执行。
- 非预期接触产生 dangerous signal 和学校样本。

### 9.4 Avoid Keypoint

任务：接近目标同时避让关键点。

验证：

- 本地大脑生成 avoidance keypoints。
- 小脑在光轴中体现避让。
- 失败时能降级或重新规划。

### 9.5 Carry-Like Posture Intent

任务：本地大脑要求维持 carry-like 姿态，但不做真实抓取。

验证：

- 姿态偏好进入小脑。
- 速度和姿态权衡合理。
- 高风险时姿态平滑降级。

### 9.6 Capability-Aware Replan

任务：故意给出超出能力边界的目标。

验证：

- 本地大脑能根据 capability summary 先降级。
- 小脑能拒绝不可执行意图。
- 本地大脑能生成新的保守意图。

## 10. 训练与评测方式

Phase 4 不要求端到端训练本地大脑。默认先使用结构化任务输入和受控多模态/mock perception。

### 10.1 Rule/Prompt Baseline

第一版本地大脑可以是：

- 规则编排器。
- prompt-driven lightweight model。
- structured planner。
- 小型 VLM/VLA wrapper。

关键不是模型花哨，而是接口闭环正确。

### 10.2 Supervised Intent Generation

可用合成任务生成意图样本。

训练目标：

- 目标字段正确。
- 速度/力度/风险偏好符合能力边界。
- 避让关键点正确。
- 不生成 blocked intent。

### 10.3 Closed-Loop Evaluation

必须在仿真中闭环评测：

```text
task input -> local brain semantic intent -> cerebellum light axis -> follower execution -> feedback -> local brain next intent
```

不能只评估语义字段准确率。

## 11. 学校系统数据回流

Phase 4 学校系统主要收集语义意图到执行之间的失败边界。

### 11.1 Segment Types

新增或重点使用：

```text
semantic_intent_unreachable
semantic_intent_degraded
semantic_intent_rejected
semantic_to_axis_failure
semantic_to_execution_failure
capability_boundary_violation
object_context_lost
avoidance_hint_failure
contact_policy_violation
successful_semantic_intent
```

### 11.2 Sample Envelope 扩展

```json
{
  "semantic_intent": {
    "intent_id": "intent_000001",
    "goal_type": "approach_and_hover",
    "accepted": true,
    "degraded": true,
    "rejected": false,
    "degrade_reason": "target_near_reach_limit",
    "capability_summary_ids": [
      "cap_cerebellum_phase3_v001"
    ]
  }
}
```

### 11.3 优先级评分

```text
semantic_priority_score =
  0.25 * capability_boundary_violation_score +
  0.20 * semantic_to_axis_error_score +
  0.20 * execution_failure_score +
  0.15 * degradation_frequency_score +
  0.10 * object_context_uncertainty_score +
  0.10 * data_quality_score
```

用途：

- 更新 capability summary。
- 训练小脑处理语义 hint。
- 后续训练本地大脑的能力边界感知。

## 12. 评测指标

### 12.1 意图生成指标

- intent schema validation pass rate。
- blocked intent avoidance rate。
- capability-aware degradation rate。
- target object selection accuracy。
- avoidance keypoint inclusion accuracy。
- contact policy correctness。

### 12.2 闭环任务指标

- task success rate。
- task progress。
- semantic-to-axis acceptance rate。
- degraded execution success rate。
- rejected intent recovery rate。
- replan success rate。

### 12.3 运动与风险指标

- fall rate。
- near-fall rate。
- dangerous signal peak。
- fallback count。
- hard switch count。
- wrist RMSE。
- hover violation rate。
- avoidance violation rate。

### 12.4 能力边界指标

- capability boundary violation rate。
- unsupported intent generation rate。
- known failure mode repeat rate。
- local brain compliance with planning constraints。

## 13. Online 验收门槛

Phase 4 online 验收以仿真闭环为主。

| 指标 | 门槛 |
| --- | --- |
| `intent_schema_validation_pass_rate` | `100%` |
| `task_success_rate` | `> 85%` |
| `semantic_to_axis_acceptance_rate` | `> 90%` |
| `degraded_execution_success_rate` | `> 80%` |
| `unsupported_intent_generation_rate` | `< 5%` |
| `capability_boundary_violation_rate` | `< 5%` |
| `fall_rate` | `< 1.5%` |
| `hover_violation_rate` | `< 5%` |
| `avoidance_violation_rate` | `< 5%` |
| `replay_validation_pass_rate` | `100%` |
| `school_sample_generation_rate_for_failures` | `100%` |

这些门槛用于验证接口闭环，不代表产品级任务成功率。

## 14. Release Gate

本地大脑不作为学校主替换目标，但 Phase 4 仍需要 release gate 验证语义意图接口版本。

必须通过：

1. `semantic_intent.v0` schema validation。
2. 小脑 accepted/degraded/rejected response validation。
3. Phase 1 locomotion regression。
4. Phase 2 whole-body regression。
5. Phase 3 light-axis regression。
6. capability boundary compliance。
7. semantic failure sample generation。

本地大脑模型本身的版本发布，可以由独立多模态模型流程管理；本文只定义它和小脑/follower 的接口契约。

## 15. Offline 分支

Phase 4 online 通过后，创建 offline 分支，但不阻塞后续线上研究。

offline 分支验证：

- 结构化语义意图在高保真动力学下是否仍可执行。
- hover/contact policy 是否过于乐观。
- 避让关键点在真实几何误差下是否仍有效。
- 能力边界是否需要收紧。

offline failure 回流：

- `semantic_intent_unreachable`
- `contact_policy_violation`
- `avoidance_hint_failure`
- `sim_to_real_semantic_gap`

## 16. 与云端大脑的关系

云端大脑向本地大脑提供高层计划和任务约束。本地大脑结合本地感知和能力边界，生成可执行语义意图。

云端大脑不直接给小脑输出动作，也不直接参与学校训练。

云端大脑应读取学校 capability summary，避免规划超出当前机器人能力的任务。

如果本地大脑连续收到小脑拒绝或降级，可以请求云端重新规划，但本地仍应能在无云情况下执行常见任务的保守版本。

## 17. 推荐文件布局

后续实现时，建议增加：

```text
configs/
  local_brain/
    semantic_intent_schema.yaml
    phase4_task_set.yaml
  eval/
    phase4_semantic_intent_scenarios.yaml
    phase4_release_gate.yaml
  school/
    phase4_semantic_sample_scoring.yaml

src/
  northstar/
    local_brain/
      intent_schema.py
      intent_generator.py
      capability_context.py
      feedback_handler.py
    cerebellum/
      semantic_hint_adapter.py
      intent_response.py
    school/
      phase4_semantic_scorer.py
    eval/
      phase4_semantic_intent_eval.py

tests/
  local_brain/
    test_semantic_intent_schema.py
    test_capability_context.py
    test_intent_generator_baseline.py
  cerebellum/
    test_semantic_hint_adapter.py
    test_intent_response_schema.py
  school/
    test_phase4_semantic_scorer.py
  eval/
    test_phase4_semantic_intent_eval.py
```

## 18. 风险与缓解

### 18.1 语义意图过高维

风险：小脑无法稳定映射语义意图。

缓解：先用结构化 intent schema 和受控任务集；记录 rejected/degraded cases；必要时引入更明确的子目标编译层。

### 18.2 本地大脑无视能力边界

风险：持续生成不可执行目标。

缓解：capability context 作为必需输入；unsupported intent generation rate 作为验收指标。

### 18.3 本地大脑试图绕过小脑

风险：端到端模型直接输出轨迹或 action，破坏架构分层。

缓解：接口只允许 semantic intent；小脑是唯一 intent-to-light-axis 路径。

### 18.4 Mock perception 过于理想

风险：Phase 4 在结构化真值中通过，但无法迁移到真实感知。

缓解：在 offline 分支中引入感知噪声、对象丢失和几何误差；这些失败进入学校样本。

### 18.5 安全体系缺失

风险：没有完整安全仲裁时，语义意图可能过激。

缓解：Phase 4 限定封闭场景；小脑 gate/fallback 仍负责运动层风险；开放安全后续单独设计。

## 19. 完成定义

Phase 4 本地大脑语义意图接口完成，需要满足：

1. 定义 `semantic_intent.v0`。
2. 定义 `intent_response.v0`。
3. 定义 capability context 输入。
4. 定义 execution feedback。
5. 至少完成 Go-To、Reach Near Object、Pre-Contact Hover、Avoid Keypoint、Capability-Aware Replan 任务。
6. 本地大脑不直接输出 action 或 light-axis。
7. 小脑可以 accepted/degraded/rejected 语义意图。
8. 所有 rejected/degraded/failure 都能进入学校样本。
9. 生成 Phase 4 capability boundary 反馈。
10. 通过 Phase 1/2/3 regression。
