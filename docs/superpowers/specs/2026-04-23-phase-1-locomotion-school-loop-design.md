# Phase 1 基础运动底座与学校最小闭环设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)

## 1. 目标

Phase 1 的目标，是在 Phase 0 ABI 和学校系统骨架之上，训练并验证第一个基础运动 whole-body follower，同时让学校最小闭环从一开始参与。

Phase 1 必须证明：

1. 机器人能在仿真中完成站立、行走、转向、速度跟踪、抗扰动、stop、brace 和 fallback。
2. 训练、评测、日志、经验精炼、学校上传、候选模型评测和候选发布形成闭环。
3. 基础运动 follower 的输入输出严格遵守 Phase 0 ABI。
4. 学校系统能收集跌倒、近跌倒、扰动、速度跟踪误差、action clipping、fallback 相关片段。
5. 候选模型不能绕过评测和 gate 直接替换稳定模型。

Phase 1 的重点不是上肢 reach，也不是小脑光轴。它的重点是建立可靠运动底座和第一条学校学习回路。

## 2. 范围

本文覆盖：

- Phase 1 任务集。
- command distribution。
- follower 输入输出。
- 训练流程。
- reward 结构。
- termination 与 dangerous signal。
- online simulation 验收指标。
- offline 分支触发条件。
- 学校最小闭环。
- 候选模型评测和发布门槛。
- Phase 1 到 Phase 2 的交接条件。

本文不覆盖：

- 上肢轨迹跟踪和全身操作任务。
- 小脑 generator/selector。
- 本地大脑语义意图。
- 真实机器人强验收。
- 完整 gate blending 公式。
- 完整联邦算法选择。
- 视觉、语言、雷达或世界模型。

## 3. 前置条件

Phase 1 开始前，Phase 0 应提供：

- `abi.northstar.v0`。
- `unitree_g1_43dof_sim_v0` 或等价 embodiment manifest。
- 可运行的 Environment Adapter。
- 可校验的 Observation / Command / Action / Confidence / Dangerous Signal ABI。
- Episode Logger。
- Evaluation Runner。
- Model Manifest。
- School Sample Envelope。
- 至少一个 debug baseline policy。

学校系统应提供：

- 文件级或本地目录级 sample ingress。
- Experience Pool metadata index。
- Dataset Manifest。
- Model Registry 状态机。
- Candidate Release Package。
- Gate Feedback 格式。
- Capability Summary 格式。

如果 Isaac Lab 尚未可用，Phase 1 可以先用 mock env 完成 ABI 和学校闭环测试，但 locomotion 验收必须在真实目标仿真环境中完成。

## 4. Phase 1 子阶段

Phase 1 分成四个子阶段，避免一次性把训练、学校和发布全部压在一起。

### 4.1 Phase 1-A：站立与姿态稳定

目标：

- 在零速度命令下保持站立。
- 维持骨盆高度、身体直立和低动作抖动。
- 建立 fall / near-fall / brace 事件。

输出：

- `follower_loco_stand_v001`。
- 站立 episode logs。
- 站立评测报告。
- 第一批 `clean_success_reference` 和 `near_failure` 学校样本。

### 4.2 Phase 1-B：平面速度跟踪

目标：

- 跟踪 base 坐标系下的平面线速度和 yaw rate。
- 完成前进、后退、横移、转向和组合命令。
- 在动作平滑、能耗和关节限位约束下维持稳定步态。

输出：

- `follower_loco_velocity_v001`。
- 速度跟踪数据集。
- locomotion capability summary 初版。

### 4.3 Phase 1-C：扰动、Stop 与 Brace

目标：

- 在外部推力、命令突变和轻微地形变化下保持稳定。
- 支持 `stop_request` 和 `brace_request`。
- 记录 fallback 进入、退出、恢复结果。

输出：

- `follower_loco_robust_v001`。
- high-priority `fallback_transition` 和 `recovery_success` 样本。
- dangerous signal 阈值校准。

### 4.4 Phase 1-D：学校候选模型闭环

目标：

- 从学校经验池构建 Phase 1 dataset version。
- 训练或评测 candidate follower。
- 通过 release gate 生成 candidate release package。
- 本地执行影子推理、低比例 gate 和 fallback。
- 将 gate feedback 和失败片段回流学校。

输出：

- `follower_candidate_phase1_v001`。
- `release_follower_candidate_phase1_v001`。
- gate feedback report。
- Phase 1 学校闭环评估报告。

## 5. Command Distribution

Phase 1 使用 Phase 0 `command.locomotion`。

### 5.1 Command 字段

启用字段：

```json
{
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
    "target_heading_rad": 0.0,
    "stop_request": false,
    "brace_request": false
  }
}
```

Phase 1 不启用：

- `upper_body.end_effector_targets`
- `light_axis_hint`
- `semantic_hint`

这些字段必须保留在 schema 中，但 mask 为 false。

### 5.2 训练 Command 范围

默认训练范围：

| 命令 | 范围 | 说明 |
| --- | --- | --- |
| `target_velocity_base_m_s.x` | `[-0.6, 1.0]` | 前后速度 |
| `target_velocity_base_m_s.y` | `[-0.4, 0.4]` | 横向速度 |
| `target_velocity_base_m_s.z` | `0.0` | Phase 1 不训练竖直速度 |
| `target_yaw_rate_rad_s` | `[-1.0, 1.0]` | 转向速度 |
| `target_base_height_m` | embodiment 默认高度附近 `±0.03` | 小范围高度调节 |
| `target_heading_rad` | disabled | Phase 1 默认用 yaw rate |
| `stop_request` | 事件式启用 | 用于停止测试 |
| `brace_request` | 事件式启用 | 用于扰动或高风险测试 |

训练课程逐步扩大范围：

1. 只训练零速度站立。
2. 前进速度 `x in [0.0, 0.5]`。
3. 加入后退和横移。
4. 加入 yaw rate。
5. 加入命令突变。
6. 加入 stop 和 brace。
7. 加入扰动和轻微随机地形。

### 5.3 评测 Command 范围

评测范围应覆盖训练范围，并包含边界命令：

| 场景 | 命令 |
| --- | --- |
| `stand_zero` | `vx=0, vy=0, yaw=0` |
| `walk_forward_slow` | `vx=0.3, vy=0, yaw=0` |
| `walk_forward_fast` | `vx=0.9, vy=0, yaw=0` |
| `walk_backward` | `vx=-0.4, vy=0, yaw=0` |
| `sidestep_left` | `vx=0, vy=0.3, yaw=0` |
| `sidestep_right` | `vx=0, vy=-0.3, yaw=0` |
| `turn_left` | `vx=0.2, vy=0, yaw=0.8` |
| `turn_right` | `vx=0.2, vy=0, yaw=-0.8` |
| `stop_from_walk` | `vx=0.8 -> stop_request=true` |
| `brace_under_push` | `external_push + brace_request=true` |

## 6. Follower 输入输出

### 6.1 输入

Phase 1 follower 默认只使用 `obs_public`：

- `joint_pos`
- `joint_vel`
- `base_ang_vel`
- `projected_gravity`
- `foot_contact`
- `last_action`
- `active_command.locomotion`
- `command_mask`
- `morphology_token_input`
- `history`

训练时可以使用 privileged critic 或 teacher，但部署 follower 必须有 public-only 路径。

### 6.2 Privileged 信息使用边界

允许：

- critic 使用 `base_lin_vel`、`base_height`、`contact_force`、`external_push`。
- teacher 使用 privileged observation。
- reward 和评测使用 privileged truth。

禁止：

- 部署 follower 直接依赖 `body_pos_world`、`body_quat_world`、`terrain_params` 或 domain randomization 真值。
- 评测报告不标注 privileged 依赖。

### 6.3 输出

Phase 1 follower 输出 Phase 0 `action.v0`：

- `joint_pos_delta_rad`
- `joint_vel_delta_rad_s`
- `feedforward_torque_nm`
- `stiffness_scale`
- `damping_scale`
- `action_mask`
- `action_confidence`

默认策略：

- 必需输出 `joint_pos_delta_rad`。
- `joint_vel_delta_rad_s` Phase 1 默认填 0，后续可启用。
- `feedforward_torque_nm` Phase 1 默认填 0，除非实验明确启用。
- `stiffness_scale` 与 `damping_scale` 默认全 1。

### 6.4 43 DOF 处理

Phase 1 控制全身 DOF，但上肢不执行任务语义：

- 腿部、腰部、躯干参与 locomotion 稳定。
- 手臂和手腕通过姿态正则维持自然摆臂或安全姿态。
- 灵巧手若存在，默认保持安全 rest pose，不参与 grasp。
- `upper_body` command mask 为 false。

这样可以从一开始保持 whole-body action space，但不把 Phase 2 的上肢任务提前压进 Phase 1。

## 7. 模型结构建议

Phase 1 不锁死最终模型结构，但需要有一个默认 baseline。

### 7.1 默认 Baseline

建议默认结构：

```text
obs_public -> normalization -> shared MLP trunk -> action head
                                     |
                                     -> value head
                                     |
                                     -> optional dangerous signal head
```

默认 MLP：

- trunk hidden sizes: `[768, 512, 256]`
- activation: `ELU`
- output distribution: Gaussian policy
- action mean: `joint_pos_delta_rad`
- action std: learned or scheduled
- value head: scalar

### 7.2 形态输入

Phase 1 应接入 `morphology_token_input`，但可以先用小型 MLP 编码：

```text
morphology_token_input -> morphology_mlp -> morphology_embedding
concat(obs_features, morphology_embedding) -> trunk
```

Phase 1 不要求 Transformer morphology tokens，但不能把形态输入完全从 ABI 中移除。

### 7.3 Teacher/Student

推荐路径：

1. 用 privileged critic 或 teacher 加速训练。
2. 训练 public-only student/follower。
3. 评测时分别报告 teacher-capable 指标和 deployable public-only 指标。

Phase 1 发布的 stable/candidate follower 必须是 public-only 可部署模型，除非 manifest 明确标记为 teacher-only 且禁止发布到本地执行路径。

## 8. 训练流程

### 8.1 推荐训练主线

默认主线：

```text
PPO/RSL-RL teacher or actor-critic
  -> public-only student distillation if privileged teacher used
  -> Phase 1 evaluation
  -> school sample generation
  -> school dataset version
  -> candidate retrain or fine-tune
  -> release gate evaluation
```

Phase 1 不把 AWAC/IQL 作为主训练器。AWAC/WSRL 类方法可以在本地 adapter 或学校后续 consolidation 中使用，但 Phase 1 主干仍以 on-policy locomotion 训练为主。

### 8.2 Curriculum

训练课程：

1. `stand_balance`
   零速度站立，低扰动，无地形变化。

2. `forward_walk`
   小范围前进速度，保持姿态和高度。

3. `velocity_tracking`
   前后、横移、yaw rate 组合。

4. `command_switch`
   命令分段切换，训练平滑响应。

5. `push_recovery`
   外部推力和扰动恢复。

6. `stop_brace`
   stop_request 与 brace_request。

7. `light_domain_randomization`
   小范围质量、摩擦、PD、延迟、地形扰动。

课程升级条件：

- 当前课程 fall rate 低于门槛。
- 速度/高度误差低于门槛。
- action clipping 事件不过量。
- near-failure 样本已进入学校经验池。

### 8.3 Domain Randomization

Phase 1 默认随机化：

| 项 | 范围 | 说明 |
| --- | --- | --- |
| ground friction | `[0.6, 1.2]` | 平面摩擦 |
| motor strength scale | `[0.85, 1.15]` | 执行器强度 |
| joint damping scale | `[0.8, 1.2]` | 阻尼 |
| mass scale | `[0.9, 1.1]` | 全身或局部质量 |
| control latency | `[0, 2]` steps | 轻微延迟 |
| push force | curriculum controlled | 抗扰课程启用 |

Phase 1 不默认启用复杂视觉地形或大范围崎岖地形。轻微高度扰动可以作为鲁棒性测试，但不是早期通过条件。

## 9. Reward 设计

Phase 1 reward 分为主任务奖励和正则项。所有 reward 都应在配置中有权重，并写入 run manifest。

### 9.1 总 reward

推荐结构：

```text
r_total =
  w_alive              * r_alive
+ w_vel_xy             * r_vel_xy
+ w_yaw_rate           * r_yaw_rate
+ w_base_height        * r_base_height
+ w_upright            * r_upright
+ w_contact            * r_contact
+ w_stop_brace         * r_stop_brace
- w_foot_slip          * p_foot_slip
- w_action_rate        * p_action_rate
- w_joint_limit        * p_joint_limit
- w_torque             * p_torque
- w_energy             * p_energy
- w_collision          * p_collision
```

### 9.2 主任务项

速度跟踪：

```text
r_vel_xy = exp(-||v_xy_cmd - v_xy_base||^2 / sigma_vel^2)
sigma_vel = 0.25
```

yaw rate 跟踪：

```text
r_yaw_rate = exp(-(yaw_rate_cmd - yaw_rate)^2 / sigma_yaw^2)
sigma_yaw = 0.35
```

骨盆高度：

```text
r_base_height = exp(-(h_cmd - h_base)^2 / sigma_height^2)
sigma_height = 0.06
```

直立：

```text
r_upright = clamp((projected_gravity_z_abs - 0.5) / 0.5, 0, 1)
```

存活：

```text
r_alive = 1.0 if not terminated else 0.0
```

### 9.3 正则项

足端滑动：

```text
p_foot_slip = sum(contact_i * ||foot_velocity_xy_i||^2)
```

action rate：

```text
p_action_rate = ||action_t - action_{t-1}||^2
```

关节限位：

```text
p_joint_limit = mean(near_limit_ratio_j^2)
```

力矩：

```text
p_torque = mean((torque_j / torque_limit_j)^2)
```

能耗：

```text
p_energy = mean(abs(torque_j * joint_vel_j))
```

碰撞：

```text
p_collision = count(non_allowed_body_contact)
```

### 9.4 初始权重建议

| 项 | 初始权重 |
| --- | --- |
| `w_alive` | `1.0` |
| `w_vel_xy` | `2.0` |
| `w_yaw_rate` | `1.0` |
| `w_base_height` | `0.8` |
| `w_upright` | `1.0` |
| `w_contact` | `0.2` |
| `w_stop_brace` | `0.5` |
| `w_foot_slip` | `0.2` |
| `w_action_rate` | `0.05` |
| `w_joint_limit` | `0.5` |
| `w_torque` | `0.02` |
| `w_energy` | `0.01` |
| `w_collision` | `1.0` |

权重不是最终答案。每次变更必须记录到 run manifest，并在评测报告中与前一组权重对比。

## 10. Termination、Confidence 与 Dangerous Signal

### 10.1 Termination

以下条件触发 episode termination：

- `base_height_m < h_min`。
- `abs(roll) > roll_max` 或 `abs(pitch) > pitch_max`。
- 非允许刚体严重碰撞。
- NaN 或 Inf 出现在 observation/action/reward。
- 连续 action clipping 超过阈值。

建议默认：

| 条件 | 阈值 |
| --- | --- |
| `h_min` | 默认站立高度的 `65%` |
| `roll_max` | `0.9 rad` |
| `pitch_max` | `0.9 rad` |
| consecutive clipping | `10 steps` |

### 10.2 Truncation

以下条件触发 truncation：

- episode 达到最大时长。
- 评测场景完成。
- 手动重置。

默认 episode 时长：

- 训练：`20s`
- 评测：`20s` 到 `60s`，按场景设置。

### 10.3 Confidence

Phase 1 confidence 最小字段：

- `overall`
- `locomotion`
- `state_estimation`
- `command_feasibility`
- `fallback_readiness`

规则：

- 速度命令接近训练边界时，`command_feasibility` 应下降。
- dangerous signal 上升时，`overall` 应下降。
- brace/fallback 可用时，`fallback_readiness` 应保持高值。

### 10.4 Dangerous Signal

Phase 1 dangerous signal 由规则 + 统计阈值开始：

- `fall_risk`：base height、projected gravity、angular velocity。
- `overload_risk`：torque limit ratio、joint limit proximity。
- `unreachable_risk`：命令超出 Phase 1 速度/yaw 能力边界。
- `low_confidence`：confidence 低于阈值。

near-fall 不等于 fall。near-fall 应产生高价值学校样本，而不是简单丢弃。

## 11. Episode Logging 扩展

Phase 1 在 Phase 0 logging 基础上增加 locomotion 指标。

### 11.1 Step Info

`info_json` 中增加：

```json
{
  "phase": "phase_1",
  "curriculum_stage": "velocity_tracking",
  "command_error": {
    "velocity_xy_norm": 0.12,
    "yaw_rate_abs": 0.08,
    "base_height_abs": 0.015
  },
  "stability": {
    "base_height_m": 0.72,
    "projected_gravity_z": -0.98,
    "roll_rad": 0.02,
    "pitch_rad": -0.03
  },
  "energy": {
    "mean_torque_abs_nm": 12.0,
    "mean_power_abs": 40.0
  }
}
```

### 11.2 Episode Metrics

Phase 1 `metrics.json` 增加：

```json
{
  "schema_version": "episode_metrics.v1.phase1",
  "fall": false,
  "near_fall_count": 0,
  "duration_s": 20.0,
  "velocity_rmse_m_s": 0.12,
  "yaw_rate_rmse_rad_s": 0.08,
  "base_height_rmse_m": 0.02,
  "mean_action_delta_norm": 0.18,
  "mean_torque_abs_nm": 12.0,
  "energy_per_meter": 0.0,
  "foot_slip_mean": 0.0,
  "joint_limit_clip_count": 0,
  "torque_limit_clip_count": 0,
  "fallback_count": 0,
  "stop_success": true,
  "brace_recovery_success": true
}
```

## 12. 学校最小闭环

Phase 1 从一开始接入学校系统。

### 12.1 上传片段类型

Phase 1 必须上传：

- `failure`
- `near_failure`
- `fallback_transition`
- `recovery_success`
- `rare_command`
- `high_uncertainty`
- `clean_success_reference`
- `regression_case`

### 12.2 Phase 1 片段优先级

Phase 1 使用学校系统基础公式，并设置 locomotion-specific 子分数。

风险子分数：

```text
risk_score = max(
  fall_risk_peak,
  near_fall_severity,
  overload_risk_peak,
  joint_limit_risk_peak
)
```

学习误差子分数：

```text
learning_error_score = normalize(
  velocity_rmse_m_s +
  0.5 * yaw_rate_rmse_rad_s +
  2.0 * base_height_rmse_m
)
```

任务相关性：

```text
task_relevance_score = 1.0 for Phase 1 locomotion commands
task_relevance_score = 0.5 for disabled upper body passive posture events
```

### 12.3 数据集构建

Phase 1 dataset version 至少分三类：

```text
dataset_phase1_locomotion_train
dataset_phase1_locomotion_validation
dataset_phase1_locomotion_release_gate
```

推荐比例：

- train: `80%`
- validation: `10%`
- release_gate: `10%`

Release gate 集合要求：

- 包含所有已知 failure 类型。
- 包含 high-priority near-failure。
- 包含 stop/brace 场景。
- 包含边界速度和 yaw command。
- 包含上一稳定模型表现良好的 clean success reference，防止候选模型退化。

### 12.4 学校最小训练任务

Phase 1 学校最小 job：

1. `follower_training`
   从 Phase 1 dataset 或仿真 rollout 训练候选 follower。

2. `dangerous_signal_training` 或 `dangerous_signal_calibration`
   校准 fall risk、near-fall、overload、low confidence。

3. `candidate_evaluation`
   在 release gate 和在线评测场景中比较候选模型与稳定模型。

### 12.5 学校输出

Phase 1 学校输出：

- `dataset_manifest.v0`
- `training_job_manifest.v0`
- `school_evaluation_report.v0`
- `model_manifest.v0`
- `release_package.v0`
- `capability_summary.v0`

## 13. Candidate Release 与 Gate

### 13.1 稳定模型与候选模型

Phase 1 至少维护：

- `follower_stable_phase1_v000`：保守稳定 baseline。
- `follower_candidate_phase1_v001`：学校训练或评测后的候选模型。

候选模型不能直接成为 stable。

### 13.2 候选发布包

Phase 1 release package 默认 gate 建议：

```json
{
  "initial_takeover_ratio": 0.0,
  "max_takeover_ratio": 0.25,
  "allowed_task_families": ["locomotion"],
  "blocked_conditions": [
    "fall_risk > 0.5",
    "confidence < 0.4",
    "abs(target_yaw_rate_rad_s) > 1.0",
    "target_velocity_norm_m_s > 1.1"
  ]
}
```

### 13.3 影子推理

候选模型先进行 shadow inference：

- 稳定模型控制环境。
- 候选模型只计算 action，不执行。
- 记录 stable/candidate action disagreement。
- 记录候选 confidence 与 dangerous signal。
- high disagreement 片段进入学校。

### 13.4 低比例接管

通过 shadow 后，允许低风险场景低比例接管：

- 只在 `stand_zero`、`walk_forward_slow`、`walk_forward_fast` 的低风险区间启用。
- 初始接管比例为 0。
- 最大接管比例为 0.25。
- fall risk 或 low confidence 触发连续 fallback。

### 13.5 回滚条件

候选模型触发以下任一条件，应进入 `rolled_back` 或暂停 staged：

- 候选 active fall rate 高于稳定模型。
- dangerous_sig peak 明显高于稳定模型。
- fallback_count 超过 release gate 门槛。
- command tracking 退化超过允许阈值。
- 发生未覆盖的严重 failure 类型。

## 14. Online 验收指标

Phase 1 online 验收以仿真为主。

### 14.1 基础通过门槛

默认通过门槛：

| 指标 | 门槛 |
| --- | --- |
| `fall_rate` | `< 1%` |
| `near_fall_rate` | `< 5%` |
| `velocity_rmse_m_s` | `< 0.15` |
| `yaw_rate_rmse_rad_s` | `< 0.20` |
| `base_height_rmse_m` | `< 0.03` |
| `episode_completion_rate` | `> 98%` |
| `stop_success_rate` | `> 95%` |
| `brace_recovery_success_rate` | `> 90%` |
| `schema_validation_pass_rate` | `100%` |
| `replay_validation_pass_rate` | `100%` |

### 14.2 Release Gate 门槛

候选模型发布为 staged 前：

| 指标 | 门槛 |
| --- | --- |
| ABI/schema validation | pass |
| release gate replay | pass |
| fall rate vs stable | 不高于 stable |
| near-fall rate vs stable | 不高于 stable + 1% absolute |
| velocity RMSE vs stable | 不劣于 stable 10% 以上 |
| action smoothness vs stable | 不劣于 stable 15% 以上 |
| fallback recovery | pass |
| capability summary | generated |

### 14.3 回归测试

每次 Phase 1 候选模型必须跑：

- Phase 0 ABI tests。
- Phase 0 log replay integrity。
- Phase 1 standing tests。
- Phase 1 velocity tracking tests。
- Phase 1 stop/brace tests。
- Phase 1 school sample generation tests。
- Phase 1 release package validation。

## 15. Offline 分支

Phase 1 online 通过后，创建 offline 分支。offline 分支不阻塞 Phase 2。

### 15.1 Offline 分支目标

offline 分支验证：

- 真实或高保真环境中的站立稳定性。
- 低速行走或硬件在环回放。
- action clipping 和 torque limit 的现实偏差。
- dangerous signal 阈值是否过于乐观。
- 仿真中没有暴露的接触、延迟、执行器问题。

### 15.2 Offline 结果回流

offline failure 必须生成：

- `offline_high_fidelity_case` 样本。
- capability summary 的 known failure mode。
- Phase 1 或 Phase 2 的 regression case。

offline failure 不阻塞线上 Phase 2，但必须进入学校经验池，并在下一轮候选模型评测中作为 release gate 样本。

## 16. 能力摘要

Phase 1 结束时，学校应生成 locomotion capability summary。

示例：

```json
{
  "schema_version": "capability_summary.v0",
  "model_id": "follower_stable_phase1_v001",
  "phase": "phase_1",
  "capabilities": [
    {
      "name": "stand_flat",
      "status": "supported",
      "conditions": {
        "terrain": "flat",
        "duration_s_max_tested": 60.0
      },
      "confidence": 0.95
    },
    {
      "name": "velocity_tracking_flat",
      "status": "supported",
      "conditions": {
        "target_velocity_m_s_abs_max": 1.0,
        "target_lateral_velocity_m_s_abs_max": 0.4,
        "target_yaw_rate_rad_s_abs_max": 1.0
      },
      "confidence": 0.85
    },
    {
      "name": "brace_under_push",
      "status": "limited",
      "conditions": {
        "push_force": "curriculum_range_only"
      },
      "confidence": 0.75
    }
  ],
  "unsupported_or_risky": [
    {
      "name": "high_speed_running",
      "condition": "target_velocity_m_s > 1.0",
      "risk": "not_validated"
    },
    {
      "name": "high_yaw_turn",
      "condition": "abs(target_yaw_rate_rad_s) > 1.0",
      "risk": "near_fall"
    }
  ],
  "recommended_planning_constraints": {
    "max_velocity_m_s": 1.0,
    "max_lateral_velocity_m_s": 0.4,
    "max_yaw_rate_rad_s": 1.0,
    "requires_flat_or_light_randomized_terrain": true
  }
}
```

云端大脑和后续本地大脑应读取该摘要，避免规划超出 Phase 1 能力边界的任务。

## 17. 与 Phase 2 的交接

Phase 1 通过后，Phase 2 可以开始全身协调任务。

Phase 1 需要向 Phase 2 提供：

- 稳定 public-only locomotion follower。
- Phase 1 model manifest。
- locomotion capability summary。
- Phase 1 release gate 数据集。
- `clean_success_reference` locomotion 样本。
- `near_failure` 和 `fallback_transition` 样本。
- 已校准的 fall risk / near-fall dangerous signal。
- 上肢 passive posture 的 action/regularization 数据。

Phase 2 不应重新定义基础 locomotion ABI，而应启用 `upper_body` command 扩展。

## 18. 推荐文件布局

后续实现时，建议在 Phase 0 文件布局上增加：

```text
configs/
  train/
    phase1_locomotion.yaml
  rewards/
    phase1_locomotion_rewards.yaml
  command/
    phase1_command_distribution.yaml
  eval/
    phase1_locomotion_scenarios.yaml
  school/
    phase1_dataset_query.yaml
    phase1_release_gate.yaml

src/
  northstar/
    rewards/
      phase1_locomotion.py
    command/
      phase1_distribution.py
    training/
      phase1_runner.py
    school/
      local_refiner.py
      phase1_sample_scorer.py
    eval/
      phase1_locomotion_eval.py

tests/
  rewards/
    test_phase1_locomotion_rewards.py
  command/
    test_phase1_command_distribution.py
  school/
    test_phase1_sample_scorer.py
  eval/
    test_phase1_locomotion_eval.py
```

如果实现语言不是 Python，文件名可以调整，但责任边界应保持。

## 19. 风险与缓解

### 19.1 训练过早引入复杂随机化

风险：策略在基础步态未稳定前被随机化扰乱，导致收敛慢。

缓解：按 curriculum 增加随机化；每个课程有 fall rate 和 tracking error 门槛。

### 19.2 手臂被忽略导致 Phase 2 返工

风险：Phase 1 只训腿，Phase 2 引入上肢时破坏整体动作空间。

缓解：Phase 1 控制全身 DOF，但上肢只做 passive posture 和自然摆臂正则，不启用上肢任务。

### 19.3 学校只收失败导致策略保守

风险：数据集被 failure/near-failure 主导。

缓解：dataset builder 强制保留 `clean_success_reference`，release gate 同时检查成功动作退化。

### 19.4 候选模型平均更好但边界更差

风险：平均指标提升，边界 command 或 near-failure 退化。

缓解：release gate 包含边界速度、yaw、stop、brace 和高风险样本。

### 19.5 Privileged 依赖泄漏

风险：发布模型依赖仿真 privileged 信息。

缓解：model manifest 标注 input dependency；stable/candidate 发布路径只允许 public-only follower。

### 19.6 Offline 失败被忽略

风险：线下分支不阻塞线上，导致真实偏差长期不被修正。

缓解：offline failure 默认生成 high-priority `offline_high_fidelity_case`，并进入后续 release gate。

## 20. Phase 1 完成定义

Phase 1 完成需要满足：

1. 至少一个 public-only locomotion follower 通过 online simulation 验收。
2. Episode logging、replay、metrics 和 model manifest 全部通过 schema validation。
3. 学校系统接收并索引 Phase 1 样本。
4. 至少生成一个 Phase 1 dataset manifest。
5. 至少生成一个 candidate model 或 candidate evaluation report。
6. Release package 包含 fallback model 和 gate recommendation。
7. 本地或仿真客户端产生 gate feedback。
8. 生成 Phase 1 locomotion capability summary。
9. Phase 1 online 通过后创建 offline 分支任务，不要求 offline 通过才能进入 Phase 2。

