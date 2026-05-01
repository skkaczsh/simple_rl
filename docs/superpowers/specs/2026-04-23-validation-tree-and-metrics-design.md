# 线上/线下验证树与指标设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [Phase 1 基础运动与学校闭环设计](./2026-04-23-phase-1-locomotion-school-loop-design.md)
- [Whole-Body Follower 统一策略设计](./2026-04-23-whole-body-follower-unified-policy-design.md)
- [小脑光轴学习与消融设计](./2026-04-23-cerebellum-light-axis-learning-design.md)
- [Model Gate、Fallback 与版本切换设计](./2026-04-23-model-gate-fallback-design.md)
- [本地大脑语义意图接口设计](./2026-04-23-local-brain-semantic-intent-interface-design.md)
- [Mimic Motion Prior 集成与应用测试设计](./2026-05-01-mimic-motion-prior-integration-design.md)

## 1. 目标

本文定义北极星架构的线上/线下验证树、指标分层、阶段推进规则和学校系统消费方式。

核心目标：

1. 保持线上仿真主线快速推进。
2. 在每个线上 Phase 通过后派生线下真实或高保真验证分支。
3. 让线下分支失败不阻塞下一线上 Phase，但必须回流为风险、样本、能力边界和后续 release gate。
4. 让每个 Phase 的验收有明确进入条件、退出条件、回归矩阵和失败处理。
5. 让学校系统从 Phase 1 起贯穿所有阶段，消费验证结果，而不是只在最后做训练。

本文是验证体系总览层。各 Phase 的详细任务、训练算法和局部门槛仍以对应子规格为准。

## 2. 范围

本文覆盖：

- 线上/线下二叉树式验证结构。
- 验证节点状态机。
- Phase 0、Phase 1、Phase 2、Phase 2.5、Phase 3、Phase 4 的线上门槛汇总。
- 每个 Phase 线上通过后派生的线下分支。
- 线下失败如何记录风险并进入学校系统。
- 回归测试矩阵。
- 指标分级、指标归属和决策规则。
- release gate 与 school dataset 的关系。

本文不覆盖：

- 具体仿真器或真实机器人测试台选型。
- 具体模型结构。
- 每个 reward 项的细节。
- 完整开放环境安全认证。
- 产品级用户验收。
- 云端大脑规划算法。

## 3. 设计原则

### 3.1 线上主线负责推进速度

线上主线使用仿真实时闭环或可重复评测环境。每个 Phase 的线上验收通过后，可以进入下一线上 Phase。

线上主线必须满足：

- 可固定 seed 复现。
- 可生成 evaluation report。
- 可回放 episode。
- 可与 stable model 做对照。
- 可生成学校样本。

### 3.2 线下分支负责真实偏差

线下分支使用高保真动力学、硬件在环、真实机器人或真实数据回放。

线下分支不阻塞下一线上 Phase，但它的结果不能被忽略。失败必须进入：

- risk register。
- school sample pool。
- capability summary。
- release gate regression set。
- 后续 Phase 的已知失败模式。

### 3.3 不用平均指标掩盖长尾失败

每个 Phase 都必须同时看：

- 平均任务表现。
- 高风险场景。
- near-failure 场景。
- boundary command 场景。
- 失败回放场景。
- 前序 Phase 回归。

### 3.4 Release Gate 优先于训练 Loss

训练 loss、平均 reward 或单次任务成功率不能作为发布依据。候选模型必须通过 release gate，尤其是稳定性、fallback、schema/replay 和前序回归。

### 3.5 学校系统横向贯穿

学校系统从 Phase 1 开始运行。它不是后置 Phase，也不是只收集最终数据。

学校系统在每个阶段负责：

- 收集验证样本。
- 标注失败类型。
- 计算优先级。
- 构造训练集和 release gate。
- 输出 capability summary。
- 支撑候选模型发布和回滚。

### 3.6 线下失败不阻塞线上，但能阻断发布

线下失败不阻塞下一线上 Phase 的研究推进。但如果某个线下失败已进入 release gate，候选模型在未处理该失败前不能发布为 stable。

这条规则区分：

- research progression：线上阶段继续推进。
- model release：候选版本必须过 gate。

## 4. 验证树模型

验证结构是带异步分支的阶段树。

```text
online trunk:

  Phase 0
    |
    v
  Phase 1  ---->  Offline 1
    |
    v
  Phase 2  ---->  Offline 2
    |
    v
  Phase 2.5 ----> Offline 2.5
    |
    v
  Phase 3  ---->  Offline 3
    |
    v
  Phase 4  ---->  Offline 4
    |
    v
  later phases ----> later offline branches
```

规则：

1. `Phase N online passed` 是进入 `Phase N+1 online` 的必要条件。
2. `Phase N offline passed` 不是进入 `Phase N+1 online` 的必要条件。
3. `Phase N offline failed` 必须生成风险记录和学校样本。
4. `Phase N offline failed` 可以成为 `Phase N+1` 或后续候选模型 release gate 的阻断项。
5. 如果线下失败暴露 ABI 根本性问题，可以触发架构修订，但这属于显式决策，而不是默认阻塞。

## 5. 验证节点状态机

每个验证节点包括 online 节点和可选 offline 节点。

### 5.1 Online Node 状态

```text
not_started
running
blocked
passed
failed
superseded
```

状态含义：

- `not_started`：前序 online 未通过，或资源尚未分配。
- `running`：正在训练、评测或补齐样本。
- `blocked`：schema、环境、数据或关键能力缺失导致无法验收。
- `passed`：满足当前 Phase online gate。
- `failed`：完成评测但未达门槛。
- `superseded`：被更新的 Phase 定义或 ABI 版本替代。

### 5.2 Offline Node 状态

```text
not_created
created
running
passed
failed_nonblocking
failed_release_blocking
risk_accepted
superseded
```

状态含义：

- `not_created`：对应 online Phase 尚未通过。
- `created`：online passed 后创建线下任务。
- `running`：线下测试或高保真回放进行中。
- `passed`：线下验证通过。
- `failed_nonblocking`：失败不阻塞线上，但进入风险和学校样本。
- `failed_release_blocking`：失败不阻塞研究推进，但阻断相关候选模型发布。
- `risk_accepted`：团队显式接受短期风险，并记录缓解计划。
- `superseded`：被更高保真或更新测试替代。

### 5.3 状态记录 Schema

```json
{
  "schema_version": "validation_node.v0",
  "node_id": "phase3_online_20260423",
  "phase": "phase_3",
  "branch": "online",
  "status": "passed",
  "abi_version": "abi.northstar.v0",
  "model_ids": [
    "cerebellum_candidate_phase3_v001",
    "follower_stable_phase2_v001"
  ],
  "started_at": "2026-04-23T10:00:00+08:00",
  "completed_at": "2026-04-23T14:00:00+08:00",
  "evaluation_report_ids": [
    "eval_phase3_online_000001"
  ],
  "release_gate_report_ids": [
    "gate_phase3_000001"
  ],
  "derived_offline_node_ids": [
    "phase3_offline_20260423"
  ],
  "risk_record_ids": [],
  "school_sample_query": {
    "phase": "phase_3",
    "validation_node_id": "phase3_online_20260423"
  }
}
```

## 6. 验收层级

每个 Phase 的验收分四层。

### 6.1 L0：Schema 与可回放

最低层，必须 100% 通过。

包括：

- ABI/schema validation。
- episode log 可读取。
- replay runner 可复算指标。
- model manifest 可关联评测报告。
- school sample envelope 可生成。

任一 L0 失败，当前 Phase 不允许通过。

### 6.2 L1：任务能力

验证当前 Phase 新增能力是否成立。

示例：

- Phase 1：locomotion。
- Phase 2：全身协调和 reach。
- Phase 2.5：统一策略成立。
- Phase 3：小脑光轴、gate、fallback。
- Phase 4：语义意图闭环。

### 6.3 L2：风险与恢复

验证系统在高风险和近失败场景中的表现。

包括：

- fall rate。
- near-fall rate。
- dangerous signal。
- fallback recovery。
- hard switch。
- model disagreement。
- boundary command。

### 6.4 L3：前序回归

每个 Phase 都必须保留前序能力。

示例：

- Phase 3 不能破坏 Phase 1 locomotion。
- Phase 4 不能绕过小脑或破坏 Phase 3 fallback。
- 候选模型不能在 release gate 中显著退化 stable model 已有能力。

## 7. Phase 0 Online Gate

Phase 0 验收重点是基础设施，不评价高性能策略。

### 7.1 通过条件

Phase 0 通过需要满足：

- 所有必需 schema 能被 validator 读取和校验。
- 仿真环境可按固定 seed 启动、reset、step、关闭。
- 至少一个 baseline policy 通过 policy/follower adapter 产生合法 action。
- action clipping 事件可被记录和回放。
- 至少 3 个 seeds 的 evaluation runner 完成并输出 report。
- episode logs 可被 replay runner 读取并复算 metrics。
- model manifest 可注册并关联 evaluation report。
- school sample envelope 可从 episode 片段生成。
- Phase 1 需要的字段在 ABI 中有明确位置。

### 7.2 必需场景

```text
reset_stability
zero_command_noop
random_command_schema
event_injection
log_replay_integrity
```

### 7.3 Phase 0 输出

Phase 0 通过后输出：

- `abi.northstar.v0`。
- evaluation runner。
- episode log schema。
- model manifest schema。
- school sample envelope。
- Phase 1 可复用的 baseline pipeline。

Phase 0 默认不创建真实机器人 offline 分支；它可以创建 infra replay branch，用于验证日志、artifact 和 schema 迁移。

## 8. Phase 1 Online Gate

Phase 1 验证基础运动底座和学校最小闭环。

### 8.1 Online 门槛

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

### 8.2 Release Gate

候选模型发布为 staged 前必须满足：

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

### 8.3 Offline 1

Phase 1 online 通过后创建 Offline 1。

Offline 1 验证：

- 真实或高保真环境中的站立稳定性。
- 低速行走或硬件在环回放。
- action clipping 和 torque limit 的现实偏差。
- dangerous signal 阈值是否过于乐观。
- 仿真中没有暴露的接触、延迟和执行器问题。

Offline 1 failure 必须生成：

- `offline_high_fidelity_case`。
- locomotion known failure mode。
- Phase 1 或 Phase 2 regression case。

## 9. Phase 2 Online Gate

Phase 2 验证全身协调任务，同时保持 Phase 1 locomotion 能力。

### 9.1 Online 门槛

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

### 9.2 必需任务族

Phase 2 online gate 至少包含：

- walk + reach。
- standing + reach。
- posture hold。
- avoid keypoint。
- pre-contact hover。
- carry-like posture。
- perturbation with upper-body task。

### 9.3 Offline 2

Phase 2 online 通过后创建 Offline 2。

Offline 2 验证：

- 上肢姿态导致的真实质心偏移。
- 执行器延迟对上肢/下肢协调的影响。
- 手臂摆动对实际步态的扰动。
- 接触前悬停的距离误差。
- action clipping 和 torque limit 的高保真偏差。

Offline 2 failure 必须生成：

- `offline_high_fidelity_case`。
- `upper_body_breaks_locomotion`。
- `uncoordinated_com_shift`。
- `pre_contact_hover_violation`。
- Phase 2.5 release gate sample。

## 10. Phase 2.5 Online Gate

Phase 2.5 是 whole-body 统一策略验证阶段。它决定后续小脑层应该建立在统一 follower、shared trunk + multi-head、MoE，还是更碎片化的 baseline 上。

### 10.1 Phase 2.5-A：单策略覆盖多控制模式

验收问题：

同一策略或同一共享 trunk 是否能覆盖站立、行走、转向、reach、carry、avoidance、brace 和 stop。

重点指标：

| 指标 | 门槛 |
| --- | --- |
| `mode_success_rate` | `> 95%` |
| `task_family_pass_fraction` | `>= 80%` |
| `hard_mode_failure_rate` | `< 5%` |
| `mode_switch_recovery_s` | `< 1.0s` |

### 10.2 Phase 2.5-B：上下肢互不破坏

验收问题：

上肢任务执行不应显著破坏步态稳定；下肢扰动时，上肢目标应平滑降级，而不是造成整体失稳。

重点指标：

| 指标 | 门槛 |
| --- | --- |
| `loco_degradation_ratio` | `< 20%` |
| `upper_body_task_under_perturbation_success_rate` | `> 80%` |
| `graceful_degrade_rate` | `> 90%` |
| `com_shift_peak_m` | `< 0.08m` |
| `contact_disruption_rate` | `< 5%` |
| `self_collision_rate` | `< 1%` |

### 10.3 Phase 2.5-C：统一 ABI 成立

验收问题：

同一套 `obs / command / action / confidence / dangerous_sig` schema 是否能承载 Phase 1、Phase 2 和 Phase 2.5 的任务，并成为 Phase 3 小脑接口基础。

重点指标：

| 指标 | 门槛 |
| --- | --- |
| `schema_validation_pass_rate` | `100%` |
| `replay_validation_pass_rate` | `100%` |
| `adapter_branch_rewrite_required` | `false` |
| `light_axis_reserved_fields_compatible` | `true` |
| `school release gate` | pass |

### 10.4 Phase 2.5 决策规则

Phase 2.5 应比较：

- unified policy。
- shared trunk + multi-head。
- MoE。
- stitched baseline。
- Mimic prior。
- Mimic teacher-student。

推荐决策：

1. 如果 unified policy 或 shared trunk + multi-head 在 80% 以上任务族达到门槛，且 fallback 连续性优于 stitched baseline，则作为 Phase 3 默认底座。
2. 如果 MoE 明显更稳，但切换连续性可被 gate 控制，则 Phase 3 小脑应按 expert/gate 架构设计。
3. 如果 Mimic teacher-student 显著提升协调性但依赖 privileged input 或受限许可证，则只能作为 teacher/oracle，不得作为 runtime candidate。
4. 如果所有统一路线都失败，Phase 3 仍可继续，但不能假设单一 follower 是完整 teacher，必须显式保留 expert/fallback 路径。

### 10.5 Offline 2.5

Phase 2.5 online 通过后创建 Offline 2.5。

Offline 2.5 验证：

- 统一策略在高保真动力学下的跨模式稳定性。
- shared trunk 或 MoE expert 在真实延迟下的切换连续性。
- 上下肢耦合失败是否被 online 指标低估。
- `confidence` 和 `dangerous_sig` 是否能覆盖模式切换风险。

Offline 2.5 failure 必须生成：

- `unified_policy_sim_to_real_gap`。
- `expert_switch_instability`。
- `upper_body_breaks_locomotion`。
- `uncoordinated_com_shift`。
- `mimic_retargeting_gap`。
- `mimic_runtime_observation_gap`。
- Phase 3 release gate sample。

## 11. Phase 3 Online Gate

Phase 3 验证小脑 generator、selector/gate、光轴、confidence、dangerous_sig 和 smooth fallback。

### 11.1 Online 门槛

| 指标 | 门槛 |
| --- | --- |
| `task_success_rate` | `> 90%` |
| `fall_rate` | `< 1.5%` |
| `near_fall_rate` | `< 6%` |
| `candidate_validity_rate` | `> 95%` |
| `unreachable_axis_rate` | `< 3%` |
| `fallback_recovery_success_rate` | `> 95%` |
| `hard_switch_count_per_min` | `< 1` |
| `dangerous_signal_lead_time_s` | `> 0.25s` for high-risk events |
| `dangerous_false_negative_rate` | `< 5%` |
| `phase1_regression` | pass |
| `phase2_regression` | pass |
| `schema_validation_pass_rate` | `100%` |
| `replay_validation_pass_rate` | `100%` |

### 11.2 Release Gate

小脑候选模型发布前必须通过：

1. ABI validation。
2. Light-axis command schema validation。
3. Phase 1 locomotion regression。
4. Phase 2 whole-body regression。
5. Phase 3 light-axis release gate。
6. stable/candidate shadow inference。
7. fallback abruptness test。
8. dangerous signal lead-time test。
9. offline high-fidelity case replay。

### 11.3 Offline 3

Phase 3 online 通过后创建 Offline 3。

Offline 3 验证：

- 光轴输出对真实或高保真 follower 的可执行性。
- dangerous signal 在执行器延迟下是否仍有提前量。
- fallback 是否在真实动力学下平滑。
- expert/gate 切换是否产生动作突变。
- contact 和 torque 风险是否被低估。

Offline 3 failure 必须生成：

- `offline_high_fidelity_case`。
- `dangerous_signal_late`。
- `fallback_abrupt`。
- `expert_switch_instability`。
- `unreachable_axis`。
- `sim_to_real_axis_bias`。

这些样本进入 Phase 4 release gate 和后续小脑再训练。

## 12. Gate/Fallback 横向 Online Gate

Gate/fallback 是 Phase 3 的核心能力，但它也横向影响 Phase 2.5、Phase 4 和后续候选模型替换。

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

任何候选模型只要涉及模型切换、adapter、MoE、selector 或新旧模型混合，都必须跑 gate/fallback 横向 gate。

## 13. Phase 4 Online Gate

Phase 4 验证本地大脑语义意图到小脑光轴再到 follower 执行的闭环。

### 13.1 Online 门槛

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

### 13.2 Release Gate

Phase 4 语义意图接口版本必须通过：

1. `semantic_intent.v0` schema validation。
2. 小脑 accepted/degraded/rejected response validation。
3. Phase 1 locomotion regression。
4. Phase 2 whole-body regression。
5. Phase 3 light-axis regression。
6. capability boundary compliance。
7. semantic failure sample generation。

本地大脑模型本身不作为当前学校主替换目标。学校主线继续训练和发布小脑 + follower，但 Phase 4 的语义失败会更新 capability summary 和后续训练集。

### 13.3 Offline 4

Phase 4 online 通过后创建 Offline 4。

Offline 4 验证：

- 结构化语义意图在高保真动力学下是否仍可执行。
- hover/contact policy 是否过于乐观。
- 避让关键点在真实几何误差下是否仍有效。
- 能力边界是否需要收紧。
- 本地大脑是否在能力边界变化后仍能生成保守意图。

Offline 4 failure 必须生成：

- `semantic_intent_unreachable`。
- `contact_policy_violation`。
- `avoidance_hint_failure`。
- `sim_to_real_semantic_gap`。
- `capability_boundary_too_optimistic`。

## 14. 回归测试矩阵

每个 Phase 的候选模型必须跑当前 Phase gate 和所有相关前序回归。

| 当前阶段 | 必跑回归 | 目的 |
| --- | --- | --- |
| Phase 0 | schema/replay self-check | 保证 ABI 基础可用 |
| Phase 1 | Phase 0 ABI + replay | 保证基础字段和日志不破坏 |
| Phase 2 | Phase 0 + Phase 1 locomotion | 保证上肢任务不破坏基础运动 |
| Phase 2.5 | Phase 0 + Phase 1 + Phase 2 | 验证统一策略覆盖前序任务 |
| Phase 3 | Phase 0 + Phase 1 + Phase 2 + Phase 2.5 | 保证小脑光轴不破坏 follower 能力 |
| Phase 4 | Phase 0 + Phase 1 + Phase 2 + Phase 2.5 + Phase 3 | 保证语义意图不绕过运动层安全边界 |

### 14.1 回归集合类型

```text
abi_regression
log_replay_regression
locomotion_regression
whole_body_regression
unified_policy_regression
light_axis_regression
gate_fallback_regression
semantic_intent_regression
offline_high_fidelity_regression
```

### 14.2 回归集生成规则

回归样本来自：

- 当前 Phase release gate。
- stable model 历史失败。
- candidate model 失败。
- offline branch failure。
- near-failure 高风险片段。
- boundary command。
- 用户手工指定关键场景。

回归集必须区分：

- training split。
- validation split。
- release gate split。
- holdout split。

release gate split 不能进入训练。

## 15. 指标分类

### 15.1 Infrastructure Metrics

用于判断系统是否可测、可回放、可发布。

- schema validation pass rate。
- replay validation pass rate。
- artifact hash validation pass rate。
- model manifest validation。
- evaluation report completeness。
- school sample generation rate。

### 15.2 Task Metrics

用于判断当前 Phase 新能力是否成立。

- task success rate。
- episode completion rate。
- target tracking error。
- wrist RMSE。
- hover violation rate。
- semantic-to-axis acceptance rate。

### 15.3 Stability Metrics

用于判断整体运动是否稳定。

- fall rate。
- near-fall rate。
- base height RMSE。
- COM disruption score。
- contact slip rate。
- self-collision rate。

### 15.4 Recovery Metrics

用于判断系统是否能平滑恢复。

- stop success rate。
- brace recovery success rate。
- fallback recovery success rate。
- graceful degrade rate。
- rejected intent recovery rate。

### 15.5 Gate Metrics

用于判断候选模型切换和混合是否安全。

- hard switch count per min。
- fallback abruptness score。
- model disagreement distribution。
- candidate takeover ratio。
- rollback rate。
- gate feedback generation rate。

### 15.6 Dangerous Signal Metrics

用于判断风险信号是否及时。

- dangerous signal lead time。
- dangerous false negative rate。
- dangerous false positive rate。
- dangerous signal peak。
- dangerous event recall。

### 15.7 School Metrics

用于判断学校系统是否真正消费了验证结果。

- high-priority sample usage rate。
- release gate pass rate。
- near-failure reproduction rate。
- known failure mode freshness。
- capability summary coverage。
- candidate regression count。

## 16. 指标归属与冲突处理

指标有三层来源：

1. Phase 子规格：当前 Phase 的详细任务和局部门槛。
2. 本文：跨 Phase 的验证树、回归矩阵和通用门槛。
3. 学校系统：数据质量、发布流程和 candidate/stable 对照。

冲突处理规则：

1. 如果 Phase 子规格门槛更严格，使用 Phase 子规格。
2. 如果本文的横向 gate 更严格，使用横向 gate。
3. 如果学校 release gate 因线下失败新增阻断样本，候选发布必须服从 release gate。
4. 如果门槛冲突无法自动合并，必须生成 validation decision record。

## 17. 线下失败风险分级

线下失败按影响分为四级。

| 等级 | 名称 | 是否阻塞线上推进 | 是否阻塞发布 | 处理 |
| --- | --- | --- | --- | --- |
| R0 | data quality issue | 否 | 仅当数据进入 release gate 时阻塞 | 修复日志、schema 或测试装置 |
| R1 | capability gap | 否 | 仅当能力边界影响候选发布时阻塞 | 更新 capability summary 和训练样本 |
| R2 | release-blocking regression | 否 | 是 | 加入 release gate，候选不得 stable |
| R3 | architecture risk | 需要显式决策 | 是 | 触发架构评审或 Phase 修订 |

### 17.1 R0 Data Quality Issue

示例：

- 线下日志缺字段。
- 时间戳不同步。
- artifact hash 缺失。
- replay runner 无法读取。

处理：

- 修复数据管线。
- 不直接训练模型。
- 只有清洗后样本可进入学校系统。

### 17.2 R1 Capability Gap

示例：

- 高保真环境中 reach 距离更保守。
- hover distance 误差较大。
- 真实延迟使动作变慢。

处理：

- 更新 capability summary。
- 降低 planner constraints。
- 生成高优先级训练样本。

### 17.3 R2 Release-Blocking Regression

示例：

- 候选模型在 offline high-fidelity case 中反复跌倒。
- fallback abruptness 超标。
- dangerous signal 迟报。
- 上肢任务明显破坏 locomotion。

处理：

- 加入 release gate split。
- 候选模型不得发布为 stable。
- 需要新候选修复后重新评测。

### 17.4 R3 Architecture Risk

示例：

- 统一 ABI 无法承载小脑光轴。
- whole-body 统一策略方向失败。
- 语义意图无法稳定映射到光轴。
- gate/fallback 机制无法避免危险切换。

处理：

- 生成 architecture decision record。
- 明确是否调整 Phase 定义。
- 明确是否回退到 MoE、expert 或更显式中间层。

## 18. 学校系统消费验证结果

学校系统把验证结果转成训练、评测和发布资产。

### 18.1 输入

学校系统接收：

- online evaluation report。
- offline evaluation report。
- episode logs。
- release gate reports。
- validation node status。
- risk records。
- capability summary。
- candidate/stable comparison reports。

### 18.2 样本优先级

验证样本的优先级来自：

```text
validation_priority_score =
  0.25 * risk_severity_score +
  0.20 * near_failure_score +
  0.20 * regression_relevance_score +
  0.15 * phase_novelty_score +
  0.10 * offline_fidelity_score +
  0.10 * data_quality_score
```

含义：

- risk severity 越高，越应进入 release gate。
- near-failure 比普通失败更适合训练 dangerous_sig 和 fallback。
- 与当前候选回归相关的样本优先级更高。
- 线下高保真样本默认提高优先级。
- 数据质量不足的样本不能直接进入训练。

### 18.3 输出

学校系统输出：

- training dataset。
- validation dataset。
- release gate dataset。
- high-risk replay set。
- capability summary。
- known failure mode summary。
- model release package。
- rollback report。

### 18.4 禁止规则

学校系统禁止：

- 把 release gate split 混入训练。
- 删除仍被 release gate 引用的失败样本。
- 只用平均 reward 决定发布。
- 忽略 offline failure。
- 在没有 capability summary 的情况下发布新 stable。

## 19. Validation Report

每次 online 或 offline 验证都必须生成统一报告。

```json
{
  "schema_version": "validation_report.v0",
  "report_id": "validation_phase4_online_000001",
  "validation_node_id": "phase4_online_20260423",
  "phase": "phase_4",
  "branch": "online",
  "abi_version": "abi.northstar.v0",
  "model_ids": [
    "local_brain_baseline_v001",
    "cerebellum_stable_phase3_v001",
    "follower_stable_phase2_v001"
  ],
  "scenario_set_id": "phase4_semantic_intent_scenarios_v001",
  "seed_count": 128,
  "summary": {
    "passed": true,
    "blocking_failures": [],
    "nonblocking_failures": [],
    "risk_records_created": 0,
    "school_samples_created": 42
  },
  "metrics": {
    "schema_validation_pass_rate": 1.0,
    "replay_validation_pass_rate": 1.0,
    "task_success_rate": 0.88,
    "fall_rate": 0.004,
    "capability_boundary_violation_rate": 0.03
  },
  "regression": {
    "phase1_regression": "pass",
    "phase2_regression": "pass",
    "phase3_regression": "pass"
  },
  "school_outputs": {
    "sample_ids": ["sample_001", "sample_002"],
    "capability_summary_id": "cap_phase4_v001",
    "release_gate_dataset_id": "gate_phase4_v001"
  }
}
```

## 20. Validation Decision Record

当出现门槛冲突、线下高风险失败或 Phase 定义变更时，必须生成决策记录。

```json
{
  "schema_version": "validation_decision_record.v0",
  "decision_id": "vdr_000001",
  "title": "accept offline hover distance gap as nonblocking",
  "phase": "phase_4",
  "source_report_ids": [
    "validation_phase4_offline_000003"
  ],
  "risk_level": "R1",
  "decision": "continue_online_progression_and_add_release_gate_case",
  "rationale": "offline gap affects hover tolerance but does not invalidate semantic intent interface",
  "required_actions": [
    "tighten capability summary hover_distance_m",
    "add contact_policy_violation to release gate",
    "create high-priority school samples"
  ],
  "owner": "architecture",
  "review_before": "2026-05-15"
}
```

## 21. 推进节奏

每个 Phase 的节奏：

1. 定义 scenario set。
2. 固定 ABI/schema 版本。
3. 运行 baseline eval。
4. 训练或接入 candidate。
5. 运行 online evaluation。
6. 运行前序 regression。
7. 生成 validation report。
8. online passed 后创建 offline branch。
9. online passed 后允许下一 online Phase 开始。
10. offline branch 异步回流学校。
11. release gate 持续吸收 offline failure。

### 21.1 最小周期开销

每个 online Phase 至少需要：

- 一个 scenario set。
- 一个 evaluation runner。
- 一个 release gate dataset。
- 一个 regression matrix。
- 一个 capability summary。
- 一个 school sample scoring 配置。

### 21.2 候选模型发布节奏

候选模型发布应遵循：

```text
trained
  -> evaluated
  -> release_gate_passed
  -> staged
  -> shadow
  -> limited_active
  -> stable
```

如果任一步产生 R2/R3 风险，候选模型不能进入 stable。

## 22. 推荐文件布局

后续实现时，建议增加：

```text
configs/
  validation/
    validation_tree.yaml
    phase0_online_gate.yaml
    phase1_online_gate.yaml
    phase2_online_gate.yaml
    phase25_online_gate.yaml
    phase3_online_gate.yaml
    phase4_online_gate.yaml
    cross_phase_regression_matrix.yaml
    offline_branch_policy.yaml
    risk_severity.yaml

src/
  northstar/
    validation/
      validation_node.py
      validation_report.py
      decision_record.py
      gate_registry.py
      regression_matrix.py
      offline_branch.py
      risk_register.py
    school/
      validation_sample_priority.py
      release_gate_builder.py

tests/
  validation/
    test_validation_node_state.py
    test_validation_report_schema.py
    test_regression_matrix.py
    test_offline_failure_policy.py
    test_release_gate_split_isolation.py
```

## 23. 风险与缓解

### 23.1 线下失败被忽略

风险：因为线下不阻塞线上，团队可能持续推进但不修复真实偏差。

缓解：线下失败必须进入 risk register、capability summary 和 release gate；R2/R3 可以阻断 stable 发布。

### 23.2 验收指标过多导致推进变慢

风险：每个 Phase 被指标淹没，线上主线无法快速推进。

缓解：指标分层；L0 必须全过，L1/L2 使用 Phase gate，L3 只跑相关前序回归；非关键线下检查异步执行。

### 23.3 平均指标提升掩盖危险退化

风险：candidate 平均表现更好，但边界场景和 near-failure 退化。

缓解：release gate 必须包含 boundary command、near-failure、offline failure 和 stable regression。

### 23.4 Release Gate 污染训练集

风险：模型直接训练 release gate 样本，导致发布评测失真。

缓解：school dataset 强制 split isolation；release gate split 不允许进入训练。

### 23.5 Phase 2.5 决策不清

风险：whole-body 统一策略验证结果模糊，Phase 3 建在错误 follower 假设上。

缓解：Phase 2.5-A/B/C 必须输出明确决策：unified、shared trunk、MoE、expert/fallback 或架构修订。

### 23.6 ABI 频繁变更

风险：每个阶段都改接口，导致回归和 replay 失效。

缓解：Phase 0 ABI 只允许受控扩展；破坏性变化必须生成 validation decision record。

## 24. 完成定义

线上/线下验证树设计完成，需要满足：

1. 定义 online trunk 与 offline branch 的推进规则。
2. 定义 online/offline 节点状态机。
3. 汇总 Phase 0 到 Phase 4 的 online gate。
4. 定义 Phase 1 到 Phase 4 的 offline branch 产物。
5. 定义 Phase 2.5-A/B/C 的验证位置和决策规则。
6. 定义 gate/fallback 横向 gate。
7. 定义跨 Phase 回归矩阵。
8. 定义线下失败风险分级。
9. 定义学校系统如何消费验证结果。
10. 定义 validation report 和 decision record schema。
11. 明确线下失败不阻塞线上推进，但可以阻断候选模型 stable 发布。
12. 明确 release gate split 不能进入训练。
