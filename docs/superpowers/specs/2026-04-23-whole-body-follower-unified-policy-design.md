# Whole-Body Follower 统一策略设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [Phase 1 基础运动底座与学校最小闭环设计](./2026-04-23-phase-1-locomotion-school-loop-design.md)
- [Mimic Motion Prior 集成与应用测试设计](./2026-05-01-mimic-motion-prior-integration-design.md)

## 1. 目标

本文定义 Phase 2 与 Phase 2.5 中 whole-body follower 统一策略的设计和验证方案。

核心目标不是“让机器人多做几个动作”，而是验证一个关键架构假设：

> 一个 unified whole-body follower，是否能用同一策略或共享 trunk 支撑 locomotion、上肢 reach、carry、avoidance、brace、posture 和 fallback，而不是退化为多个割裂控制器拼接。

该验证结果会直接决定后续小脑层如何设计：

- 如果统一策略成立，小脑可以把 follower 作为稳定、连续、可泛化的全身执行底座。
- 如果统一策略只在部分任务成立，小脑需要把 follower 能力边界显式纳入 generator/selector。
- 如果统一策略不成立，需要采用 MoE、分区 expert 或更强 gate，而不能假设单一 follower 可承载所有任务。

## 2. 范围

本文覆盖：

- Phase 2 全身协调任务定义。
- Phase 2.5-A/B/C 统一策略验收。
- 单策略、shared trunk + multi-head、MoE、拼接式 baseline 的对照实验。
- Mimic-style motion prior、teacher-student 和 unified controller 路线的对照实验。
- follower 输入输出与 ABI 使用方式。
- 上下肢互不破坏的训练和评测指标。
- 学校系统需要收集的数据类型。
- follower 何时可以作为小脑 teacher、MoE expert 或 candidate generator。

本文不覆盖：

- 小脑 generator/selector 的完整训练方案。
- 本地大脑语义意图接口。
- gate blending 的最终公式。
- 真实机器人强验收。
- 完整上肢 manipulation 或 grasp 策略。
- 视觉和语言输入。
- 开放环境安全认证。

## 3. 前置条件

进入本设计前，应已有：

- Phase 0 ABI。
- Phase 1 public-only locomotion follower。
- Phase 1 locomotion capability summary。
- Phase 1 release gate dataset。
- 学校系统经验池和 candidate release 流程。
- 可回放的 locomotion success、near-failure、fallback、stop、brace 样本。
- 可选 Mimic teacher、retargeted motion dataset 或 motion prior source manifest。

Phase 2/2.5 不应重写 Phase 0/1 的基础 ABI，而应启用 Phase 0 已预留的 `upper_body` command 和相关 masks。

## 4. 核心假设

本文要验证以下假设：

1. 同一 action space 可以覆盖下肢、躯干、上肢和手腕的协调控制。
2. 上肢任务可以通过统一 command schema 与 locomotion 同时表达。
3. 共享 trunk 可以学习全身动力学耦合，而不是让上下肢在隐空间中互相干扰。
4. 上肢任务权重增加时，locomotion 稳定性不会显著退化。
5. 下肢受到扰动时，上肢目标可以维持趋势或平滑降级，而不是产生突变。
6. 统一 ABI 可以承载 Phase 2/2.5 任务，并继续作为 Phase 3 小脑光轴的执行接口。

## 5. Phase 2 任务集

Phase 2 的任务集用于训练和初步验证全身协调。

### 5.1 Walking + Reach

机器人在速度命令下行走，同时双腕或单腕跟踪未来 spline。

任务变量：

- `target_velocity_base_m_s`
- `target_yaw_rate_rad_s`
- `left_wrist.position_knots_base_m`
- `right_wrist.position_knots_base_m`
- `position_weight`
- `rotation_weight`

验收重点：

- 速度跟踪不因 reach 大幅退化。
- 腕端误差在可接受范围。
- 质心、骨盆和脚接触保持稳定。

### 5.2 Avoidance Keypoints

机器人在行走和 reach 时避开关键空间点或禁入区域。

Phase 2 不需要完整视觉避障，只需要结构化关键点：

```json
{
  "avoidance_keypoints_base_m": [
    {
      "name": "table_edge",
      "position_base_m": [0.4, 0.2, 0.8],
      "radius_m": 0.12,
      "weight": 1.0
    }
  ]
}
```

验收重点：

- 腕端、前臂、膝部和躯干不穿越关键区域。
- 避让不导致步态发散。
- 避让失败会提高 dangerous signal。

### 5.3 Posture Hold

机器人维持指定躯干、骨盆或手臂姿态趋势。

任务变量：

- 骨盆高度。
- 躯干 pitch/roll 限制。
- 双臂 rest posture 或 carry posture。
- 末端姿态趋势。

验收重点：

- 姿态保持不破坏 locomotion。
- 突发扰动时姿态能平滑降级。

### 5.4 Pre-Contact Hover

腕端接近目标点但保持接触前悬停。

任务变量：

- 目标点。
- hover distance。
- 末端速度上限。
- 姿态权重。

验收重点：

- 末端靠近目标时速度下降。
- 不发生非预期接触。
- 下肢仍保持稳定。

### 5.5 Carry Posture

机器人维持双臂携带姿态，同时执行基础行走。

任务变量：

- 双腕相对躯干的位置。
- 双腕之间的相对距离。
- 躯干姿态。
- 速度命令。

验收重点：

- carry 姿态稳定。
- locomotion 不明显退化。
- 高风险时可放弃 carry 精度，优先保稳定。

## 6. Command ABI 扩展使用

Phase 2 启用 Phase 0 `upper_body` command。

### 6.1 Locomotion + Upper Body Command

示例：

```json
{
  "mode_mask": {
    "stand": false,
    "locomotion": true,
    "upper_body": true,
    "light_axis": false,
    "semantic_intent": false
  },
  "locomotion": {
    "target_base_height_m": 0.72,
    "target_velocity_base_m_s": [0.4, 0.0, 0.0],
    "target_yaw_rate_rad_s": 0.2,
    "target_heading_rad": 0.0,
    "stop_request": false,
    "brace_request": false
  },
  "upper_body": {
    "end_effector_targets": [
      {
        "name": "left_wrist",
        "enabled": true,
        "position_knots_base_m": [
          [0.30, 0.25, 0.75],
          [0.34, 0.25, 0.76],
          [0.38, 0.24, 0.76],
          [0.42, 0.23, 0.75]
        ],
        "rotation_6d_knots_base": [
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ],
        "knot_times_s": [0.0, 0.1, 0.2, 0.3],
        "position_weight": 1.0,
        "rotation_weight": 0.2
      }
    ]
  }
}
```

### 6.2 Phase 2 Mask 规则

- `upper_body=true` 时，至少一个 end effector target 必须 enabled。
- `position_weight=0` 的 target 不参与 tracking reward，但仍可用于日志。
- `rotation_weight` 初期应低于 `position_weight`，避免姿态跟踪破坏步态。
- `light_axis=false`，因为小脑光轴还未启用。
- `semantic_intent=false`，因为本地大脑尚未接入。

## 7. Action Space

Phase 2 继续使用 `action.v0`，控制全身 DOF。

默认启用：

- `joint_pos_delta_rad`
- `stiffness_scale`
- `damping_scale`
- `action_confidence`

默认不启用：

- 大范围 `feedforward_torque_nm`
- 高增益上肢 torque control

规则：

- 上肢和下肢不能拆成两个互不通信的 policy。
- 即使采用 multi-head 或 MoE，最终 action 也必须经过统一 action adapter 和统一 clipping/event logging。
- 上肢失败不能绕过 dangerous signal。

## 8. 候选策略架构

Phase 2.5 需要比较四类候选。

### 8.1 Unified Policy

单一 trunk + 单一 action head：

```text
obs_public + command -> normalization -> shared trunk -> action head
                                                |
                                                -> value head
                                                |
                                                -> confidence/dangerous head
```

优点：

- 最符合统一 follower 假设。
- 接口简单。
- 动作连续性最好验证。

风险：

- 上下肢任务可能互相干扰。
- 高维 command 可能导致训练不稳定。

### 8.2 Shared Trunk + Multi-Head

共享 trunk，下肢/躯干/上肢使用不同 action heads：

```text
obs_public + command -> shared trunk
                       -> lower_body_head
                       -> torso_head
                       -> upper_body_head
                       -> value/confidence heads
```

优点：

- 保留共享表示。
- 减少上下肢输出层冲突。

风险：

- head 之间可能出现局部最优。
- 如果 heads 缺少耦合约束，仍可能退化为弱拼接。

约束：

- heads 不能独立训练到互相不可见。
- loss 必须包含全身协调项和干扰项。

### 8.3 MoE Whole-Body Policy

多个 whole-body experts，每个 expert 仍输出全身 action，由 gate 混合：

```text
obs_public + command -> router
                       -> expert_0 whole-body action
                       -> expert_1 whole-body action
                       -> expert_2 whole-body action
                       -> mixed whole-body action
```

优点：

- 可覆盖不同运动族。
- 适合后续小脑 selector/gate。

风险：

- gate 不稳定会造成动作突变。
- experts 如果只学局部动作，会破坏 whole-body 假设。

约束：

- 每个 expert 必须输出全身 action。
- gate 权重变化需要平滑正则。
- release gate 必须测试 expert switch continuity。

### 8.4 Stitched Baseline

下肢 locomotion policy + 上肢 tracking controller 拼接。

用途：

- 作为对照 baseline，不作为北极星推荐终态。
- 用于证明 unified 或 shared trunk 是否确实带来收益。

风险：

- 上下肢互相不知道对方意图。
- 接触和质心变化难以协调。
- fallback 不连续。

该 baseline 如果在部分任务上更强，应作为风险信号，说明统一 follower 假设需要修正。

## 9. 训练阶段

### 9.1 Phase 2-A：Locomotion Retention

从 Phase 1 stable follower 初始化，先验证 Phase 1 能力不退化。

训练内容：

- 重放 Phase 1 locomotion commands。
- 保留 Phase 1 reward。
- 加入上肢 passive posture 正则。

通过条件：

- Phase 1 eval 指标不明显退化。
- 上肢 passive posture 不引入步态不稳定。

### 9.2 Phase 2-B：Single-Arm Reach

启用单腕 position spline，低权重。

训练内容：

- 低速 locomotion + 单腕 reach。
- reach target 限制在可达且低风险区域。
- rotation weight 初期为 0 或低权重。

通过条件：

- 腕端 RMSE 达到阶段门槛。
- fall rate 不明显高于 Phase 1。
- locomotion tracking 退化在允许范围内。

### 9.3 Phase 2-C：Dual-Arm Reach and Posture

启用双腕 reach、姿态保持和 carry posture。

训练内容：

- 双腕 position spline。
- 逐步加入 rotation 6D tracking。
- carry posture。
- stop/brace 下的上肢平滑降级。

通过条件：

- 双腕任务不破坏 locomotion。
- brace 时上肢目标可以降级。
- dangerous signal 能捕捉不可达或高风险姿态。

### 9.4 Phase 2-D：Avoidance and Pre-Contact Hover

加入避让关键点和接触前悬停。

训练内容：

- 腕端避让。
- 躯干/膝部/前臂避让关键区域。
- hover velocity regularization。

通过条件：

- 关键点碰撞率低。
- hover 末端速度可控。
- 避让失败进入学校高价值样本。

### 9.5 Phase 2.5：统一策略对照验证

冻结任务集，比较候选架构：

- Unified Policy。
- Shared Trunk + Multi-Head。
- MoE Whole-Body Policy。
- Stitched Baseline。

每个候选都使用同一 eval suite、同一 release gate、同一 school sample scoring。

## 10. Reward 结构

Phase 2 reward 在 Phase 1 基础上增加上肢和协调项。

### 10.1 总 reward

```text
r_total =
  r_phase1_locomotion
+ w_wrist_pos          * r_wrist_pos
+ w_wrist_rot          * r_wrist_rot
+ w_carry_posture      * r_carry_posture
+ w_hover              * r_hover
+ w_avoidance          * r_avoidance
+ w_coupling           * r_coupling
- w_loco_regression    * p_loco_regression
- w_upper_jerk         * p_upper_jerk
- w_com_shift          * p_com_shift
- w_self_collision     * p_self_collision
```

### 10.2 Wrist Position Tracking

```text
r_wrist_pos = mean_i exp(-||p_wrist_i - p_target_i||^2 / sigma_wrist_pos^2)
sigma_wrist_pos = 0.08
```

### 10.3 Wrist Rotation Tracking

```text
r_wrist_rot = mean_i exp(-d_6d_rot(wrist_i, target_i)^2 / sigma_wrist_rot^2)
sigma_wrist_rot = 0.35
```

### 10.4 Carry Posture

```text
r_carry_posture = exp(-||relative_wrist_pose - target_relative_pose||^2 / sigma_carry^2)
sigma_carry = 0.10
```

### 10.5 Hover

```text
r_hover = exp(-distance_to_hover_target^2 / sigma_hover^2)
          * exp(-end_effector_speed^2 / sigma_hover_speed^2)
sigma_hover = 0.05
sigma_hover_speed = 0.20
```

### 10.6 Avoidance

```text
r_avoidance = mean_k clamp((distance_to_keypoint_k - radius_k) / margin_k, 0, 1)
margin_k = 0.10
```

### 10.7 Coupling Reward

协调项用于避免上下肢各自为战：

```text
r_coupling =
  exp(-com_shift_due_to_upper_body^2 / sigma_com^2)
  * exp(-stance_contact_disruption^2 / sigma_contact^2)
```

推荐：

- `sigma_com = 0.05m`
- `sigma_contact = 0.25`

### 10.8 Locomotion Regression Penalty

```text
p_loco_regression =
  max(0, velocity_rmse_current - velocity_rmse_phase1_ref)
+ max(0, base_height_rmse_current - base_height_rmse_phase1_ref)
+ max(0, fall_risk_current - fall_risk_phase1_ref)
```

该项强制 Phase 2 在学习上肢任务时尊重 Phase 1 能力。

## 11. 初始权重建议

| 项 | Phase 2-A | Phase 2-B | Phase 2-C | Phase 2-D |
| --- | --- | --- | --- | --- |
| `w_wrist_pos` | `0.0` | `0.5` | `1.0` | `1.0` |
| `w_wrist_rot` | `0.0` | `0.0` | `0.3` | `0.3` |
| `w_carry_posture` | `0.1` | `0.1` | `0.5` | `0.5` |
| `w_hover` | `0.0` | `0.0` | `0.2` | `0.8` |
| `w_avoidance` | `0.0` | `0.0` | `0.2` | `0.8` |
| `w_coupling` | `0.5` | `0.8` | `1.0` | `1.0` |
| `w_loco_regression` | `1.0` | `1.0` | `1.0` | `1.2` |
| `w_upper_jerk` | `0.05` | `0.05` | `0.08` | `0.08` |
| `w_com_shift` | `0.5` | `0.7` | `0.8` | `1.0` |
| `w_self_collision` | `1.0` | `1.0` | `1.0` | `1.0` |

权重变更必须记录在 run manifest，并进入学校对照实验元数据。

## 12. Phase 2.5-A：单策略覆盖多控制模式

### 12.1 验收问题

同一策略或共享 trunk 是否能覆盖：

- stand。
- walk。
- turn。
- stop。
- brace。
- single-arm reach。
- dual-arm reach。
- carry posture。
- avoidance。
- pre-contact hover。

### 12.2 指标

| 指标 | 门槛 |
| --- | --- |
| `mode_success_rate` | `> 95%` |
| `fall_rate` | `< 1.5%` |
| `velocity_rmse_m_s` | `< 0.18` |
| `yaw_rate_rmse_rad_s` | `< 0.24` |
| `wrist_position_rmse_m` | `< 0.08` |
| `wrist_rotation_error` | `< 0.35` |
| `mode_switch_recovery_s` | `< 1.0s` |
| `schema_validation_pass_rate` | `100%` |

### 12.3 对照结论规则

统一策略成立的最低条件：

- Unified Policy 或 Shared Trunk + Multi-Head 至少在 80% 任务族上达到门槛。
- 相比 stitched baseline，fallback continuity 和上下肢协调指标更好。
- 相比 MoE，单策略或 shared trunk 的退化不超过可接受阈值，或 MoE 的收益足以解释其复杂度。

如果 stitched baseline 显著更好，必须记录为架构风险，不能强行宣布 unified follower 成立。

## 13. Phase 2.5-B：上下肢互不破坏

### 13.1 验收问题

上肢任务是否显著破坏步态稳定；下肢扰动时，上肢目标是否能保持趋势或平滑降级。

### 13.2 干扰测试

测试场景：

1. `walk_reach_no_disturbance`
2. `walk_reach_command_switch`
3. `walk_reach_external_push`
4. `walk_carry_yaw_turn`
5. `brace_while_reaching`
6. `stop_while_carrying`
7. `avoidance_while_walking`

### 13.3 指标

| 指标 | 说明 | 门槛 |
| --- | --- | --- |
| `loco_degradation_ratio` | 上肢任务下 locomotion RMSE 相对 Phase 1 增幅 | `< 20%` |
| `fall_rate_delta` | 上肢任务相对无上肢任务的跌倒率增幅 | `< 1% absolute` |
| `wrist_trend_preservation` | 下肢扰动时腕端仍朝目标趋势移动的比例 | `> 80%` |
| `graceful_degrade_rate` | 高风险时上肢平滑降级而非突变的比例 | `> 90%` |
| `contact_disruption_rate` | 上肢动作导致异常脚接触的比例 | `< 5%` |
| `com_shift_peak_m` | 上肢动作引起质心偏移峰值 | `< 0.08m` |

### 13.4 失败分类

失败必须分成：

- `upper_body_breaks_locomotion`
- `locomotion_breaks_upper_body`
- `uncoordinated_com_shift`
- `contact_pattern_disruption`
- `abrupt_upper_body_fallback`
- `self_collision_or_near_collision`
- `command_unreachable`

这些失败类型进入学校系统，用于 Phase 3 小脑训练。

## 14. Phase 2.5-C：统一 ABI 成立

### 14.1 验收问题

同一套 ABI 是否能承载 Phase 1、Phase 2 和 Phase 2.5 的任务，并成为 Phase 3 小脑接口基础。

### 14.2 必须保持稳定的字段

- `obs_public.joint_pos`
- `obs_public.joint_vel`
- `obs_public.base_ang_vel`
- `obs_public.projected_gravity`
- `obs_public.foot_contact`
- `obs_public.last_action`
- `obs_public.active_command`
- `obs_public.command_mask`
- `obs_public.morphology_token_input`
- `command.locomotion`
- `command.upper_body`
- `action.joint_pos_delta_rad`
- `confidence.overall`
- `dangerous_sig.overall_risk`

### 14.3 可扩展字段

Phase 3 可以扩展但不能破坏：

- `command.light_axis_hint`
- `confidence.model_version`
- `dangerous_sig.model_disagreement`
- `action_confidence`
- school sample envelope 中的 `segment_type`

### 14.4 验收指标

| 指标 | 门槛 |
| --- | --- |
| Phase 1 logs 可由新 reader 读取 | `100%` |
| Phase 2 logs 可由统一 reader 读取 | `100%` |
| Phase 1 model manifest 仍可注册 | pass |
| Phase 2 model manifest 可注册 | pass |
| action adapter 无分支重写 | pass |
| school sample envelope 兼容 Phase 1/2 | pass |

如果 Phase 2 需要 ABI MAJOR 变更，说明 Phase 0 设计失败，需要回到 ABI 文档修订，而不是在 Phase 2 私自 fork。

## 15. 学校系统数据需求

Phase 2/2.5 需要学校收集以下片段：

```text
upper_body_breaks_locomotion
locomotion_breaks_upper_body
uncoordinated_com_shift
contact_pattern_disruption
abrupt_upper_body_fallback
self_collision_or_near_collision
command_unreachable
clean_whole_body_success
mode_switch_success
mode_switch_failure
moe_gate_instability
stitched_baseline_failure
```

### 15.1 优先级评分扩展

在学校基础 priority formula 上增加：

```text
coordination_score =
  0.35 * loco_degradation_score +
  0.25 * wrist_error_score +
  0.20 * com_shift_score +
  0.10 * contact_disruption_score +
  0.10 * fallback_abruptness_score
```

Phase 2/2.5 样本 priority：

```text
priority_score =
  0.60 * base_priority_score +
  0.40 * coordination_score
```

### 15.2 Release Gate 数据集

Phase 2.5 release gate 必须包含：

- Phase 1 locomotion regression cases。
- Phase 2 clean whole-body success cases。
- 上下肢互扰 failure cases。
- boundary reach targets。
- stop/brace with upper body command。
- mode switch cases。
- stitched baseline 对照失败样本。

## 16. Candidate Model 对照矩阵

每个候选策略都必须在同一矩阵中评测。Mimic 路线不是单独豁免项；如果使用 Mimic teacher、MaskedMimic/ProtoMotions 风格统一策略、AMP/ASE prior 或 ResMimic 风格 residual policy，也必须进入同一对照矩阵。

| 任务族 | Unified | Shared Trunk + Heads | MoE | Stitched Baseline | Mimic Prior | Mimic Teacher Student |
| --- | --- | --- | --- | --- | --- | --- |
| stand | required | required | required | required | optional | required |
| walk | required | required | required | required | required | required |
| turn | required | required | required | required | required | required |
| stop/brace | required | required | required | required | optional | required |
| single reach | required | required | required | required | required | required |
| dual reach | required | required | required | required | required | required |
| carry | required | required | required | required | required | required |
| avoidance | required | required | required | required | optional | required |
| hover | required | required | required | required | required | required |
| mode switch | required | required | required | required | optional | required |

每个格子必须产出：

- success rate。
- fall rate。
- locomotion RMSE。
- wrist RMSE。
- dangerous signal peak。
- fallback count。
- sample ids for failures。

Mimic 路线额外必须产出：

- teacher/student disagreement。
- retargeting gap score。
- privileged input dependency。
- source license class。
- runtime_allowed flag。

## 17. Follower 作为小脑 Teacher/Expert 的条件

whole-body follower 只有满足以下条件，才能进入 Phase 3 小脑训练：

1. Phase 2.5-A/B/C 全部通过。
2. public-only 部署路径通过。
3. release gate 中没有严重 ABI 兼容问题。
4. 上下肢互扰 failure 已分类并进入学校经验池。
5. mode switch 不出现明显动作突变。
6. fallback 行为可回放、可记录、可由 dangerous signal 解释。
7. 学校生成 capability summary 和 known failure modes。

可进入小脑的角色：

- Teacher：提供稳定可执行 action/light-axis 近似样本。
- MoE expert：作为一种 whole-body expert 被 selector/gate 调用。
- Candidate generator：为小脑 generator 提供候选运动轨迹分布。
- Regression oracle：用于判断小脑输出是否破坏已知稳定行为。

如果 follower 只在 locomotion 成立、上肢协调不稳定，则只能作为 locomotion expert，不能作为统一 whole-body teacher。

## 18. Online 验收门槛

Phase 2/2.5 online 验收以仿真为主。

### 18.1 Phase 2 基础门槛

| 指标 | 门槛 |
| --- | --- |
| `fall_rate` | `< 1.5%` |
| `near_fall_rate` | `< 6%` |
| `velocity_rmse_m_s` | `< 0.18` |
| `yaw_rate_rmse_rad_s` | `< 0.24` |
| `base_height_rmse_m` | `< 0.04` |
| `wrist_position_rmse_m` | `< 0.08` |
| `wrist_rotation_error` | `< 0.35` |
| `self_collision_rate` | `< 1%` |
| `schema_validation_pass_rate` | `100%` |
| `replay_validation_pass_rate` | `100%` |

### 18.2 Phase 2.5 统一策略门槛

| 指标 | 门槛 |
| --- | --- |
| `mode_success_rate` | `> 95%` |
| `loco_degradation_ratio` | `< 20%` |
| `graceful_degrade_rate` | `> 90%` |
| `mode_switch_recovery_s` | `< 1.0s` |
| `fallback_abruptness_score` | `< 0.2` |
| `ABI compatibility` | pass |
| `school release gate` | pass |

## 19. Offline 分支

Phase 2 或 2.5 online 通过后，创建 offline 分支，但不阻塞 Phase 3。

offline 分支验证：

- 上肢姿态导致的真实质心偏移。
- 执行器延迟对上肢/下肢协调的影响。
- 手臂摆动对实际步态的扰动。
- 接触前悬停的距离误差。
- action clipping 和 torque limit 的高保真偏差。

offline failure 必须生成：

- `offline_high_fidelity_case`
- `upper_body_breaks_locomotion`
- `uncoordinated_com_shift`
- `abrupt_upper_body_fallback`

这些样本进入 Phase 3 小脑 release gate。

## 20. 与 Phase 3 小脑的交接

Phase 2.5 完成后，向 Phase 3 提供：

- 通过验证的 whole-body follower model manifest。
- unified/shared/MoE/stitched 对照报告。
- clean whole-body success dataset。
- upper-lower interference failure dataset。
- mode switch dataset。
- fallback transition dataset。
- capability summary。
- known failure modes。
- follower 作为 teacher/expert/candidate generator 的资格判断。

Phase 3 不应重新证明基础 whole-body follower 是否成立，而应基于 Phase 2.5 的结果设计小脑 generator/selector。

## 21. 推荐文件布局

后续实现时，建议增加：

```text
configs/
  train/
    phase2_whole_body.yaml
    phase25_unified_policy_ablation.yaml
  rewards/
    phase2_whole_body_rewards.yaml
  command/
    phase2_upper_body_distribution.yaml
  eval/
    phase2_whole_body_scenarios.yaml
    phase25_unified_policy_matrix.yaml
  school/
    phase2_coordination_sample_scoring.yaml
    phase25_release_gate.yaml

src/
  northstar/
    rewards/
      phase2_whole_body.py
    command/
      phase2_upper_body_distribution.py
    training/
      phase2_runner.py
      phase25_ablation_runner.py
    policy/
      unified_policy.py
      shared_trunk_multihead.py
      moe_whole_body.py
      stitched_baseline.py
    school/
      phase2_coordination_scorer.py
    eval/
      phase2_whole_body_eval.py
      phase25_unified_policy_eval.py

tests/
  rewards/
    test_phase2_whole_body_rewards.py
  command/
    test_phase2_upper_body_distribution.py
  policy/
    test_whole_body_policy_shapes.py
  school/
    test_phase2_coordination_scorer.py
  eval/
    test_phase25_unified_policy_eval.py
```

## 22. 风险与缓解

### 22.1 统一策略失败

风险：单策略无法同时处理 locomotion 和上肢任务。

缓解：保留 shared trunk、MoE 和 stitched baseline 对照；不强行把失败结果解释为成功。

### 22.2 上肢 reward 压过 locomotion

风险：腕端误差下降，但步态稳定退化。

缓解：加入 locomotion regression penalty，release gate 强制跑 Phase 1 regression。

### 22.3 Multi-head 退化为弱拼接

风险：shared trunk 只是表面共享，heads 实际各自为战。

缓解：加入 coupling reward、COM/contact disruption 指标和 head 间协调评测。

### 22.4 MoE gate 不稳定

风险：expert 切换造成动作突变。

缓解：gate 平滑正则、mode switch recovery 指标、fallback abruptness 指标。

### 22.5 ABI 被 Phase 2 私自 fork

风险：为了上肢任务临时新增不兼容字段，破坏 Phase 3。

缓解：Phase 2 只能通过 ABI MINOR 扩展；MAJOR 变更必须回到 Phase 0 ABI 文档修订。

### 22.6 只看平均成功率

风险：长尾 reach、扰动和 fallback 失败被平均指标掩盖。

缓解：学校 release gate 包含 near-failure、boundary command 和 interference failure 样本。

## 23. 完成定义

Whole-body follower 统一策略设计完成，需要满足：

1. Phase 2 任务集通过 online simulation 验收。
2. Phase 2.5-A 单策略覆盖多控制模式完成。
3. Phase 2.5-B 上下肢互不破坏完成。
4. Phase 2.5-C 统一 ABI 成立完成。
5. unified、shared trunk + heads、MoE、stitched baseline 至少完成一轮对照实验。
6. 学校系统收集并索引协调失败、mode switch、clean whole-body success 样本。
7. 生成候选 follower 的 evaluation report、release package 和 capability summary。
8. 明确 follower 是否可作为 Phase 3 小脑 teacher、MoE expert 或 candidate generator。
