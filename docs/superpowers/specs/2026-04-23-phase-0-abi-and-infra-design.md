# Phase 0 ABI 与基础设施设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：[北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)

## 1. 目标

Phase 0 的目标不是训练出高性能策略，而是建立后续所有阶段共用的最小技术底座：

1. 仿真环境能够稳定启动、重置、step、记录和评测。
2. 机器人 embodiment 能够通过统一 manifest 描述，避免把 G1 43 DOF 直接写死在训练代码里。
3. `observation / command / action / confidence / dangerous_sig` ABI 有明确字段、单位、shape、mask 和版本。
4. 经验日志、模型版本、评测结果和学校上传样本有统一格式。
5. 后续 Phase 1 的基础运动训练、学校最小闭环和模型候选发布，可以在该 ABI 上直接展开。

Phase 0 的验收重点是“接口可用、数据可回放、评测可复现、版本可追踪”，不是 reward 极限性能。

## 2. 设计原则

### 2.1 ABI 优先

Phase 0 必须先定义模块之间的共同语言。后续模型结构、训练算法和 reward 都可以调整，但 ABI 不能频繁重写。

### 2.2 支持扩展，不提前复杂化

Phase 0 只强制实现基础运动需要的字段，但 schema 需要预留全身协调、小脑光轴、学校系统和本地大脑接入所需的扩展位。

### 2.3 区分部署可观测与训练特权信息

部署可观测字段进入 `obs_public`。仿真中可用但真实部署不可直接获得的信息进入 `obs_privileged`。teacher 可以使用 privileged 信息，student 和 follower 部署路径默认只依赖 public 信息。

### 2.4 所有数据都必须可回放

Phase 0 记录的数据，必须足以重建 episode、回放策略输入输出、复算基础指标，并作为学校系统高价值片段筛选的输入。

### 2.5 形态不写死

G1 43 DOF 是阶段目标 embodiment，但 ABI 应通过 `embodiment_manifest` 描述关节、刚体、末端、接触点和 actuator，而不是把具体长度散落在代码中。

## 3. 非目标

Phase 0 不覆盖：

- 高性能 locomotion 策略训练。
- 小脑 generator/selector 的完整实现。
- 本地大脑多模态模型接入。
- 真实机器人控制部署。
- 开放环境安全体系。
- 完整联邦学习算法。
- 视觉、雷达或完整世界模型。

Phase 0 可以记录这些模块未来需要的字段，但不要求实现它们。

## 4. 模块边界

Phase 0 需要形成以下基础模块边界。

### 4.1 Embodiment Manifest

描述机器人形态、关节顺序、刚体、末端、接触点、action 映射和形态 token 输入。

### 4.2 Environment Adapter

封装仿真环境，统一输出 observation，接收 action，返回 reward、termination、truncation 和诊断信息。

### 4.3 ABI Validator

验证 observation、command、action、confidence、dangerous_sig、episode log 和 model manifest 是否符合 schema。

### 4.4 Command Generator

Phase 0 使用简单 command generator 产生基础运动命令，例如站立、速度跟踪、转向和 stop。后续 Phase 2 以后可扩展上肢和光轴命令。

### 4.5 Policy/Follower Adapter

封装当前策略或 follower，使其输入输出严格符合 ABI。Phase 0 可以使用 noop、random、PD baseline 或最小 MLP policy 做连通性测试。

### 4.6 Episode Logger

把每个 episode 的 manifest、step records、events、metrics 和 replay 索引写入统一目录。

### 4.7 Evaluation Runner

按固定 seeds 和 command scenarios 运行评测，输出标准 metrics 与 regression report。

### 4.8 Model Registry

记录模型版本、ABI 版本、输入输出 schema hash、训练数据来源、评测结果、fallback 父版本和 artifact checksum。

## 5. 命名与版本规则

### 5.1 ABI Version

Phase 0 ABI 版本命名为：

```text
abi.northstar.v0
```

版本兼容原则：

- PATCH：只增加可选字段或修正文档，不破坏旧日志读取。
- MINOR：增加新字段、新 mask、新事件类型，但旧模型仍可运行。
- MAJOR：改变必需字段、shape、单位或语义，需要显式迁移。

Phase 0 内默认只允许 PATCH 和受控 MINOR，不允许无记录地改变必需字段。

### 5.2 Coordinate Frames

必须显式记录所有向量所属坐标系：

- `world`：仿真世界坐标系。
- `base`：机器人根部或骨盆坐标系。
- `body:<name>`：指定刚体局部坐标系。
- `ee:<name>`：末端执行器坐标系。
- `contact:<name>`：接触点局部坐标系。

Phase 0 public observation 默认使用 `base` 坐标系表示线速度、角速度、重力方向和命令速度。世界坐标系信息应进入 privileged 或 episode metadata，避免部署路径依赖全局真值。

### 5.3 Units

统一单位：

- 长度：meter。
- 时间：second。
- 角度：radian。
- 速度：meter/second 或 radian/second。
- 力矩：newton-meter。
- 质量：kilogram。
- 概率、mask、confidence：`[0.0, 1.0]`。

## 6. Embodiment Manifest

`embodiment_manifest` 是 Phase 0 的根形态描述。它应从机器人 URDF/MJCF/USD 或手写配置生成，并在训练数据和模型版本中记录 hash。

建议 JSON 结构：

```json
{
  "schema_version": "embodiment_manifest.v0",
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "robot_family": "unitree_g1",
  "dof_count": 43,
  "control_dof_count": 43,
  "root_body": "pelvis",
  "joint_order": ["joint_000", "joint_001"],
  "actuator_order": ["motor_000", "motor_001"],
  "rigid_bodies": [
    {
      "name": "pelvis",
      "parent": null,
      "mass_kg": 0.0,
      "local_com_m": [0.0, 0.0, 0.0],
      "role": "root"
    }
  ],
  "end_effectors": [
    {
      "name": "left_wrist",
      "body": "left_wrist_link",
      "role": "manipulation"
    },
    {
      "name": "right_wrist",
      "body": "right_wrist_link",
      "role": "manipulation"
    }
  ],
  "contact_sites": [
    {
      "name": "left_foot",
      "body": "left_foot_link",
      "role": "support"
    },
    {
      "name": "right_foot",
      "body": "right_foot_link",
      "role": "support"
    }
  ],
  "joint_limits": {
    "position_lower_rad": [],
    "position_upper_rad": [],
    "velocity_abs_rad_s": [],
    "torque_abs_nm": []
  },
  "default_pd": {
    "stiffness": [],
    "damping": []
  },
  "morphology_features": {
    "per_joint_feature_names": [
      "joint_type",
      "axis_x",
      "axis_y",
      "axis_z",
      "limit_lower",
      "limit_upper",
      "torque_limit"
    ],
    "per_body_feature_names": [
      "mass",
      "com_x",
      "com_y",
      "com_z"
    ]
  }
}
```

规则：

- `joint_order` 和 `actuator_order` 一旦进入数据集和模型版本，不能无版本迁移地改变。
- 真实 joint 名称应来自资产文件；上面的 `joint_000` 只表示 schema 示例，不作为实际命名。
- `dof_count` 与所有 joint 向量长度必须一致。
- `contact_sites` 至少包含双脚支撑点。
- 双腕 end effector 从 Phase 0 就进入 manifest，即使 Phase 1 暂不训练上肢任务。
- 形态 token 的具体网络实现不是 Phase 0 目标，但输入特征名和来源必须稳定。

## 7. Observation ABI

Observation 分成 public、privileged 和 metadata。

### 7.1 顶层结构

```json
{
  "schema_version": "observation.v0",
  "abi_version": "abi.northstar.v0",
  "env_time_s": 12.34,
  "control_dt_s": 0.02,
  "episode_id": "ep_000001",
  "step_index": 617,
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "obs_public": {},
  "obs_privileged": {},
  "masks": {},
  "metadata": {}
}
```

### 7.2 Public Observation

`obs_public` 是部署路径允许依赖的信息。

字段定义：

| 字段 | Shape | 坐标系 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `joint_pos` | `[J]` | joint | rad | 当前关节位置 |
| `joint_vel` | `[J]` | joint | rad/s | 当前关节速度 |
| `base_ang_vel` | `[3]` | base | rad/s | IMU 角速度 |
| `projected_gravity` | `[3]` | base | unitless | 重力方向在 base 下的投影 |
| `foot_contact` | `[C]` | contact | 0/1 | 接触点二值接触 |
| `last_action` | `[A]` | action | mixed | 上一次 action ABI 展平向量 |
| `active_command` | object | mixed | mixed | 当前 command 结构 |
| `command_mask` | object | none | 0/1 | 当前 command 哪些字段有效 |
| `morphology_token_input` | object | none | mixed | 形态编码输入 |
| `history` | object | mixed | mixed | 最近 H 帧关键 public 状态 |

默认维度：

- `J = control_dof_count`。
- `A = action_dim`，由 action ABI 决定。
- `C = contact_sites.length`。
- `H = 4`，Phase 0 默认 history stack 长度。

`history` 最小内容：

```json
{
  "joint_pos": [[0.0]],
  "joint_vel": [[0.0]],
  "base_ang_vel": [[0.0, 0.0, 0.0]],
  "projected_gravity": [[0.0, 0.0, -1.0]],
  "last_action": [[0.0]]
}
```

### 7.3 Privileged Observation

`obs_privileged` 只允许 teacher、critic、评测和诊断使用，不进入默认部署 follower 输入。

建议字段：

| 字段 | Shape | 坐标系 | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `base_lin_vel` | `[3]` | base | m/s | 仿真真值根线速度 |
| `base_height` | `[1]` | world | m | 根部高度 |
| `body_pos_world` | `[B, 3]` | world | m | 关键刚体世界位置 |
| `body_quat_world` | `[B, 4]` | world | unit quat | 关键刚体世界姿态 |
| `contact_force` | `[C, 3]` | contact | N | 接触力 |
| `external_push` | `[3]` | base | N | 当前扰动力 |
| `terrain_params` | object | world | mixed | 地形参数 |
| `domain_randomization` | object | none | mixed | 当前随机化参数 |

规则：

- 任何使用 privileged 字段训练出来的 teacher，必须在蒸馏或部署路径中显式移除该依赖。
- 评测报告必须标注 policy 是否使用 privileged 输入。

### 7.4 Masks

Phase 0 masks 至少包含：

```json
{
  "valid_joint": [true],
  "controllable_joint": [true],
  "valid_contact": [true],
  "valid_end_effector": [true],
  "command_field_enabled": {
    "locomotion": true,
    "upper_body": false,
    "light_axis": false,
    "semantic_intent": false
  }
}
```

mask 的作用：

- 支持 23 DOF 与 43 DOF 之间的形态差异。
- 支持 Phase 0 不启用但后续会启用的上肢和小脑字段。
- 支持 batch 中不同任务启用不同 command 字段。

## 8. Command ABI

Command 是上层模块给 follower 或后续小脑/follower 链路的中层目标。Phase 0 只强制 locomotion 子集，但 schema 必须包含后续扩展位置。

### 8.1 顶层结构

```json
{
  "schema_version": "command.v0",
  "command_id": "cmd_000001",
  "source": "phase0_command_generator",
  "valid_for_s": 0.2,
  "mode_mask": {
    "stand": true,
    "locomotion": false,
    "upper_body": false,
    "light_axis": false,
    "semantic_intent": false
  },
  "locomotion": {},
  "upper_body": {},
  "light_axis_hint": {},
  "semantic_hint": {}
}
```

### 8.2 Locomotion Command

Phase 0 必需字段：

```json
{
  "locomotion": {
    "target_base_height_m": 0.0,
    "target_velocity_base_m_s": [0.0, 0.0, 0.0],
    "target_yaw_rate_rad_s": 0.0,
    "target_heading_rad": 0.0,
    "stop_request": false,
    "brace_request": false
  }
}
```

语义：

- `target_velocity_base_m_s` 使用 base 坐标系，只允许 Phase 0 使用平面速度，z 分量应为 0。
- `target_yaw_rate_rad_s` 与 `target_heading_rad` 不能同时作为强约束；需要通过 command mask 指定哪个生效。
- `stop_request` 表示进入稳定停止。
- `brace_request` 表示进入抗扰或保守姿态。

### 8.3 Upper Body Extension

Phase 0 不强制训练上肢，但 schema 预留：

```json
{
  "upper_body": {
    "end_effector_targets": [
      {
        "name": "left_wrist",
        "enabled": false,
        "position_knots_base_m": [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        "rotation_6d_knots_base": [
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ],
        "knot_times_s": [0.0, 0.1, 0.2, 0.3],
        "position_weight": 0.0,
        "rotation_weight": 0.0
      }
    ]
  }
}
```

规则：

- Phase 0 可以只记录 disabled upper body command。
- `knot_times_s` 默认四个 knot，对应 0.0s 到 0.3s。
- rotation 使用 6D representation，避免四元数符号不连续。

### 8.4 Light Axis Hint Extension

小脑 Phase 3 前不强制启用，但 Phase 0 需要预留字段：

```json
{
  "light_axis_hint": {
    "enabled": false,
    "pose_weight": 0.0,
    "velocity_weight": 0.0,
    "torque_pose_weight": 0.0,
    "risk_tolerance": 0.0,
    "fallback_preference": "stable_model"
  }
}
```

### 8.5 Semantic Hint Extension

本地大脑 Phase 4 前不强制启用，但 Phase 0 需要预留字段：

```json
{
  "semantic_hint": {
    "enabled": false,
    "goal_type": "none",
    "priority": 0.0,
    "avoidance_keypoints_base_m": [],
    "force_preference": "neutral",
    "task_style": "neutral"
  }
}
```

## 9. Action ABI

Action 是 follower 到底层 PD/伺服映射的输出。Phase 0 采用“关节目标增量 + 可选速度增量 + 可选前馈力矩”的结构。

### 9.1 顶层结构

```json
{
  "schema_version": "action.v0",
  "action_id": "act_000001",
  "policy_id": "policy_debug_noop_v0",
  "joint_pos_delta_rad": [],
  "joint_vel_delta_rad_s": [],
  "feedforward_torque_nm": [],
  "stiffness_scale": [],
  "damping_scale": [],
  "action_mask": [],
  "action_confidence": 1.0
}
```

字段规则：

- `joint_pos_delta_rad` 必需，shape `[J]`。
- `joint_vel_delta_rad_s` 可选但字段存在，未启用时填 0，shape `[J]`。
- `feedforward_torque_nm` 可选但字段存在，未启用时填 0，shape `[J]`。
- `stiffness_scale` 和 `damping_scale` 默认全 1，shape `[J]`。
- `action_mask` 表示哪些 joint action 有效，shape `[J]`。
- `action_confidence` 是 policy 输出侧自评估，不替代小脑的 `confidence`。

### 9.2 安全裁剪

Phase 0 action adapter 必须在进入仿真前执行裁剪：

- 关节位置目标不能超过 `joint_limits.position_lower_rad / position_upper_rad`。
- 速度增量不能超过 `joint_limits.velocity_abs_rad_s`。
- 前馈力矩不能超过 `joint_limits.torque_abs_nm`。
- `stiffness_scale` 和 `damping_scale` 必须在配置范围内。

被裁剪的 action 必须记录 event，不能静默吞掉。

## 10. Confidence 与 Dangerous Signal ABI

Phase 0 需要定义字段，但可以先由规则或简单诊断器产生。

### 10.1 Confidence

```json
{
  "schema_version": "confidence.v0",
  "overall": 1.0,
  "locomotion": 1.0,
  "upper_body": 0.0,
  "model_version": 1.0,
  "state_estimation": 1.0,
  "command_feasibility": 1.0,
  "fallback_readiness": 1.0
}
```

Phase 0 规则：

- `overall` 不应简单平均所有字段；应由当前启用的 command mask 决定。
- 未启用模块的 confidence 可以为 0，但不能拉低 overall。
- confidence 必须随 step 记录，用于后续 gate 和学校训练。

### 10.2 Dangerous Signal

```json
{
  "schema_version": "dangerous_sig.v0",
  "overall_risk": 0.0,
  "fall_risk": 0.0,
  "collision_risk": 0.0,
  "overload_risk": 0.0,
  "unreachable_risk": 0.0,
  "model_disagreement": 0.0,
  "low_confidence": 0.0,
  "triggered_events": []
}
```

Phase 0 规则：

- `fall_risk` 可以先由 base height、projected gravity、angular velocity 阈值估计。
- `overload_risk` 可以先由 torque limit ratio 估计。
- `model_disagreement` 在只有一个模型时为 0。
- `triggered_events` 必须使用标准事件类型。

标准事件类型：

```text
fall_detected
near_fall
joint_limit_clip
torque_limit_clip
velocity_limit_clip
nan_detected
command_invalid
episode_timeout
manual_reset
fallback_entered
fallback_exited
```

## 11. Episode Logging

每个 episode 应写入独立目录。

目录结构：

```text
runs/<run_id>/
  run_manifest.json
  models/
    <model_id>.model_manifest.json
  episodes/
    <episode_id>/
      episode_manifest.json
      steps.parquet
      events.jsonl
      metrics.json
      replay_index.json
```

### 11.1 Run Manifest

```json
{
  "schema_version": "run_manifest.v0",
  "run_id": "run_20260423_000001",
  "created_at": "2026-04-23T00:00:00+08:00",
  "abi_version": "abi.northstar.v0",
  "embodiment_manifest_hash": "sha256:...",
  "simulator": {
    "name": "isaac_lab",
    "version": "recorded_at_runtime"
  },
  "seed": 12345,
  "phase": "phase_0",
  "purpose": "abi_infra_validation"
}
```

### 11.2 Step Records

`steps.parquet` 每行对应一个环境 step。

必需列：

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `episode_id` | string | episode 标识 |
| `step_index` | int64 | step 序号 |
| `env_time_s` | float64 | 仿真时间 |
| `command_json` | string | command 结构 JSON |
| `obs_public_json` | string | public observation JSON |
| `obs_privileged_json` | string | privileged observation JSON |
| `action_json` | string | action JSON |
| `confidence_json` | string | confidence JSON |
| `dangerous_sig_json` | string | dangerous signal JSON |
| `reward` | float64 | 当前 reward |
| `terminated` | bool | 是否自然终止 |
| `truncated` | bool | 是否超时截断 |
| `info_json` | string | 诊断信息 |

Phase 0 可以先用 JSON 字符串保存结构化字段。后续学校系统设计可以将高频字段列式展开，以提高过滤效率。

### 11.3 Events

`events.jsonl` 每行一个事件：

```json
{
  "schema_version": "event.v0",
  "episode_id": "ep_000001",
  "step_index": 124,
  "event_type": "near_fall",
  "severity": 0.7,
  "source": "dangerous_signal_rule_v0",
  "payload": {
    "base_height_m": 0.42,
    "projected_gravity_z": -0.55
  }
}
```

事件规则：

- 事件必须可追溯到 step。
- `severity` 范围为 `[0.0, 1.0]`。
- `payload` 不限制字段，但必须 JSON 可序列化。

### 11.4 Metrics

`metrics.json` 至少包含：

```json
{
  "schema_version": "episode_metrics.v0",
  "episode_id": "ep_000001",
  "fall": false,
  "near_fall_count": 0,
  "duration_s": 20.0,
  "mean_reward": 0.0,
  "base_height_rmse_m": 0.0,
  "velocity_rmse_m_s": 0.0,
  "yaw_rate_rmse_rad_s": 0.0,
  "mean_action_delta_norm": 0.0,
  "mean_torque_abs_nm": 0.0,
  "joint_limit_clip_count": 0,
  "torque_limit_clip_count": 0,
  "fallback_count": 0
}
```

## 12. School Upload Sample Envelope

Phase 0 不实现完整学校系统，但必须定义上传样本 envelope，供 Phase 1 最小学校闭环使用。

```json
{
  "schema_version": "school_sample_envelope.v0",
  "sample_id": "sample_000001",
  "source_run_id": "run_20260423_000001",
  "source_episode_id": "ep_000001",
  "step_start": 100,
  "step_end": 180,
  "abi_version": "abi.northstar.v0",
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "segment_type": "near_failure",
  "priority_score": 0.8,
  "selection_reasons": [
    "near_fall",
    "high_action_delta"
  ],
  "artifact_uri": "runs/run_20260423_000001/episodes/ep_000001",
  "metrics": {
    "fall_risk_peak": 0.9,
    "confidence_min": 0.4
  }
}
```

Phase 0 只需生成 envelope 并验证可读写。训练如何采样、聚合和发布，在学校系统子规格中展开。

## 13. Model Manifest

每个策略、follower、diagnostic model 或 candidate model 都必须有 manifest。

```json
{
  "schema_version": "model_manifest.v0",
  "model_id": "follower_debug_noop_v0",
  "model_family": "whole_body_follower",
  "phase": "phase_0",
  "abi_version": "abi.northstar.v0",
  "embodiment_ids": ["unitree_g1_43dof_sim_v0"],
  "input_schema": "observation.v0",
  "output_schema": "action.v0",
  "input_schema_hash": "sha256:...",
  "output_schema_hash": "sha256:...",
  "artifact": {
    "uri": "models/follower_debug_noop_v0.pt",
    "sha256": "sha256:..."
  },
  "training_data": {
    "run_ids": [],
    "dataset_ids": []
  },
  "parent_model_id": null,
  "fallback_model_id": null,
  "gate_policy_id": null,
  "evaluation": {
    "latest_report_id": "eval_phase0_000001",
    "pass": true
  }
}
```

规则：

- 模型不能脱离 ABI 版本和 schema hash 注册。
- 候选模型必须记录 fallback model。
- 评测报告必须能追溯到 model manifest。

## 14. Evaluation Runner

Phase 0 evaluation runner 负责验证基础设施，不评价高性能策略。

### 14.1 场景集合

最小场景：

1. `reset_stability`：重置后保持静止若干 step。
2. `zero_command_noop`：零速度命令下运行 noop 或 PD baseline。
3. `random_command_schema`：随机合法 command 输入，验证 schema 与 clipping。
4. `event_injection`：人为触发 near-fall、limit clip、invalid command，验证 events 与 dangerous_sig。
5. `log_replay_integrity`：读取已记录 episode，重建 step 序列并复算 metrics。

### 14.2 Run Metrics

评测报告至少包含：

```json
{
  "schema_version": "evaluation_report.v0",
  "report_id": "eval_phase0_000001",
  "phase": "phase_0",
  "abi_version": "abi.northstar.v0",
  "model_id": "follower_debug_noop_v0",
  "scenario_results": [
    {
      "scenario": "reset_stability",
      "pass": true,
      "seed_count": 3,
      "failure_reasons": []
    }
  ],
  "summary": {
    "pass": true,
    "episode_count": 15,
    "schema_validation_pass": true,
    "replay_validation_pass": true,
    "event_validation_pass": true
  }
}
```

### 14.3 Phase 0 通过标准

Phase 0 通过需要满足：

- 所有必需 schema 能被 validator 读取和校验。
- 仿真环境可按固定 seed 启动、reset、step、关闭。
- 至少一个 baseline policy 通过 policy/follower adapter 产生合法 action。
- action clipping 事件可被记录和回放。
- 至少 3 个 seeds 的 evaluation runner 完成并输出 report。
- episode logs 可被 replay runner 读取并复算 metrics。
- model manifest 可注册并关联 evaluation report。
- school sample envelope 可从 episode 片段生成。
- root blueprint 中 Phase 1 需要的字段在 ABI 中有明确位置。

## 15. 推荐文件布局

后续实现时，推荐使用以下布局。具体语言和框架可以在 implementation plan 中根据仓库技术栈调整。

```text
configs/
  embodiment/
    unitree_g1_43dof_sim_v0.json
  abi/
    abi_northstar_v0.json
  eval/
    phase0_scenarios.yaml

src/
  northstar/
    abi/
      observation.py
      command.py
      action.py
      confidence.py
      dangerous_signal.py
      validators.py
    embodiment/
      manifest.py
      loader.py
    env/
      adapter.py
    policy/
      follower_adapter.py
      debug_baselines.py
    logging/
      episode_logger.py
      replay_reader.py
    eval/
      phase0_runner.py
      metrics.py
    registry/
      model_manifest.py

tests/
  abi/
  embodiment/
  logging/
  eval/
```

如果项目选择纯 Rust、Python 或混合栈，目录名可以调整，但责任边界应保持一致。

## 16. 与 Phase 1 的交接

Phase 0 完成后，Phase 1 不应再重新定义基础字段，而应在以下位置继续展开：

- 在 `command.locomotion` 中增加 Phase 1 command distribution。
- 在 reward 配置中引用 observation 与 privileged observation 字段。
- 在 evaluation runner 中增加 locomotion 指标和场景。
- 在 school sample envelope 中增加 Phase 1 片段评分字段。
- 在 model manifest 中注册第一个可训练 whole-body follower。

Phase 1 可以新增字段，但必须通过 ABI MINOR 版本记录。

## 17. 主要风险

### 17.1 ABI 过细导致实现负担过重

缓解方式：Phase 0 只要求字段存在和可校验，不要求所有字段都被训练使用。未启用字段通过 mask 标注。

### 17.2 ABI 过粗导致 Phase 1 返工

缓解方式：Phase 0 必须包含 locomotion、upper body、light axis、semantic hint 的最小扩展位置，即使后者暂不启用。

### 17.3 Privileged 信息泄漏到部署路径

缓解方式：validator 和 model manifest 必须区分 public-only、teacher、critic、diagnostic 模型类型。评测报告必须标注输入依赖。

### 17.4 记录格式后续无法高效训练

缓解方式：Phase 0 可用 JSON 字符串降低实现门槛，但必须保留 schema 版本和 replay 能力。学校系统子规格再定义列式展开和高效采样。

### 17.5 形态顺序不稳定

缓解方式：所有数据、模型和评测都记录 `embodiment_manifest_hash`。joint/order 变化必须产生新 embodiment id。

## 18. 待用户确认的设计取舍

以下取舍不阻塞 Phase 0 文档，但会影响 implementation plan：

1. Phase 0 的 baseline policy 使用 noop、PD baseline，还是最小 MLP policy。
2. Phase 0 schema validator 使用 JSON Schema、Pydantic、dataclass 手写校验，还是 Rust serde 类型。
3. Episode 高频数据先用 Parquet + JSON 字符串，还是直接使用 Arrow nested schema。
4. 仿真环境第一实现是否直接绑定 Isaac Lab，还是先写一个 mock env adapter 保证 ABI 测试可在无 Isaac 环境中运行。

默认建议：

- baseline policy 使用 noop + PD baseline。
- schema validator 使用 Python Pydantic 或等价 typed model，后续再生成 JSON Schema。
- 高频数据先用 Parquet + JSON 字符串，学校系统设计再做列式优化。
- 先实现 mock env adapter 和 Isaac Lab adapter 接口，避免 CI 或本地无 Isaac 时完全无法测试。

