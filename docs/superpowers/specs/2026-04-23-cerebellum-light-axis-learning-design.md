# 小脑光轴学习与消融设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [Phase 1 基础运动底座与学校最小闭环设计](./2026-04-23-phase-1-locomotion-school-loop-design.md)
- [Whole-Body Follower 统一策略设计](./2026-04-23-whole-body-follower-unified-policy-design.md)

## 1. 目标

本文定义 Phase 3 小脑层的学习方案、接口、训练阶段和消融实验。

小脑层的目标，是把上层意图、当前身体状态、历史上下文、模型版本状态和风险信号，转化为可执行、可选择、可回退的全身协调中层命令光轴。

这里的“光轴”不是单个关节动作，也不是单条手写轨迹，而是：

1. generator 在隐空间中提出的多条全身协调候选路径。
2. selector/gate 对候选路径评分、混合和选择后的结构化中层命令。
3. `pose / velocity / torque_pose / confidence / dangerous_sig` 这些字段在同一条全身协调路径上的不同投影。

Phase 3 必须回答：

- 无语义动作序列能否学出全身协调光轴。
- generator 如何产生候选路径，而不是直接输出关节命令。
- selector/gate 如何在候选路径、新旧模型、风险和任务目标之间做平滑选择。
- whole-body follower 何时作为 teacher、MoE expert、candidate generator 或 regression oracle 介入。
- 哪些方案真正提升任务成功率、fallback 连续性和危险信号提前量。

## 2. 范围

本文覆盖：

- 小脑 generator 和 selector/gate 的模块边界。
- 光轴 latent representation。
- 小脑到 follower 的结构化中层命令。
- 无语义动作序列学习路径。
- latent dynamics、teacher、MoE、RL rollout、真实或高保真片段校准的候选方案。
- 学校系统如何为小脑构建数据集。
- 消融实验设计。
- Phase 3 online 验收门槛。
- 与 Phase 4 本地大脑语义意图接口的交接。

本文不覆盖：

- 本地大脑 VLM/VLA/VLN 的模型选择。
- 完整自然语言任务规划。
- 真实机器人强验收。
- 开放环境安全认证。
- 最终 gate blending 公式的部署细节。
- 完整世界模型。

## 3. 前置条件

进入 Phase 3 前，应已有：

- Phase 0 ABI。
- Phase 1 locomotion follower。
- Phase 2/2.5 whole-body follower 对照结果。
- 至少一个可作为稳定执行底座的 follower 或 expert 集合。
- 学校系统经验池。
- `clean_whole_body_success`、`upper_body_breaks_locomotion`、`mode_switch_failure`、`fallback_transition` 等样本。
- Phase 2.5 capability summary 和 known failure modes。

如果 Phase 2.5 证明统一 follower 完全不成立，Phase 3 仍可进行，但小脑必须按 expert/gate 架构设计，不能假设单一 follower 是完整 teacher。

## 4. 小脑模块边界

Phase 3 小脑由四个子模块组成。

### 4.1 Context Encoder

把当前状态、历史、command、形态信息、模型版本信息编码为上下文向量。

输入：

- `obs_public`
- `active_command`
- `command_mask`
- `morphology_token_input`
- 最近 H 帧历史。
- 当前 stable/candidate model ids。
- 当前 capability summary 的局部可用约束。

输出：

- `context_z`
- `state_quality`
- `command_feasibility_prior`

### 4.2 Generator

在隐空间中生成多个候选光轴。

输入：

- `context_z`
- 可选 high-level intent hint。
- 可选 follower teacher embedding。
- 可选 MoE expert priors。

输出：

```text
candidate_light_axes = [
  light_axis_z_0,
  light_axis_z_1,
  ...
  light_axis_z_K
]
```

每条候选光轴都应表示全身协调路径，而不是某个局部 body part 的独立动作。

### 4.3 Selector/Gate

对候选光轴、stable/candidate model 输出和风险信号进行评分、混合和选择。

输入：

- `candidate_light_axes`
- `context_z`
- stable follower 预测。
- candidate follower 预测。
- MoE expert 输出。
- confidence/dangerous predictions。

输出：

- 最终结构化中层命令光轴。
- 新旧模型接管比例。
- fallback 建议。
- 失败或低置信原因。

### 4.4 Light-Axis Decoder

把被选中的 latent light axis 解码成 follower 可执行的中层命令。

输出字段：

- `pose`
- `velocity`
- `torque_pose`
- `confidence`
- `dangerous_sig`

decoder 不能直接绕过 follower 输出低层关节命令。底层执行仍由 whole-body follower 完成。

## 5. 光轴表示

### 5.1 Latent Light Axis

内部表示：

```text
light_axis_z = {
  z_global: [Dg],
  z_body: [B, Db],
  z_contact: [C, Dc],
  z_end_effector: [E, De],
  horizon_s: float,
  validity_mask: object
}
```

建议默认：

- `Dg = 128`
- `Db = 32`
- `Dc = 16`
- `De = 32`
- horizon: `0.3s` 到 `1.0s`

语义：

- `z_global` 表示全身运动趋势、节奏、风险和协调模式。
- `z_body` 表示关键刚体的姿态趋势。
- `z_contact` 表示接触模式和支撑变化。
- `z_end_effector` 表示末端执行器的局部运动趋势。

这些向量不是人工语义标签，而是从非语义动作序列、rollout、teacher 和任务误差中学习出来的协调表示。

### 5.2 Structured Light-Axis Command

selector/gate 最终输出：

```json
{
  "schema_version": "light_axis_command.v0",
  "source": "cerebellum_selector_v0",
  "horizon_s": 0.5,
  "pose": {
    "body_targets": [],
    "end_effector_targets": [],
    "base_posture_trend": {}
  },
  "velocity": {
    "base_velocity_target_m_s": [0.0, 0.0, 0.0],
    "base_yaw_rate_target_rad_s": 0.0,
    "end_effector_velocity_targets_m_s": []
  },
  "torque_pose": {
    "impedance_intent": [],
    "contact_intent": [],
    "force_preference": "neutral"
  },
  "confidence": {
    "overall": 1.0,
    "candidate_validity": [],
    "selected_axis_confidence": 1.0,
    "fallback_readiness": 1.0
  },
  "dangerous_sig": {
    "overall_risk": 0.0,
    "fall_risk": 0.0,
    "collision_risk": 0.0,
    "overload_risk": 0.0,
    "unreachable_risk": 0.0,
    "model_disagreement": 0.0,
    "low_confidence": 0.0
  },
  "gate": {
    "stable_model_weight": 1.0,
    "candidate_model_weight": 0.0,
    "expert_weights": [],
    "fallback_mode": "none"
  }
}
```

规则：

- `pose / velocity / torque_pose / confidence / dangerous_sig` 是同一条光轴的不同投影。
- `gate` 是 selector 的最终决策结果，不是外部附加逻辑。
- `candidate_model_weight` 初始可为 0，通过 shadow 和低风险场景逐步上升。
- `fallback_mode` 至少支持 `none / stable_model / brace / stop / conservative_posture`。

## 6. 数据来源

小脑训练使用混合来源。

### 6.1 非语义 Rollout

来自 Phase 1/2/2.5 的状态-动作-rollout，不使用动作语义标签。

包含：

- observation。
- command。
- action。
- contact。
- reward。
- confidence。
- dangerous signal。
- termination。
- follower output。

用途：

- latent dynamics。
- contact prediction。
- risk prediction。
- light-axis consistency。

### 6.2 Whole-Body Follower Teacher

来自 Phase 2.5 验证后的 follower。

用途：

- 提供可执行 action/reference。
- 提供 clean whole-body success。
- 作为 candidate generator 或 expert。
- 作为 regression oracle。

限制：

- 如果 follower 只在部分任务成立，只能作为局部 expert。
- teacher 输出不能被小脑无条件复制，小脑仍需风险评估和 gate。

### 6.3 MoE Experts

来自 Phase 2.5 的 MoE 或任务族 expert。

用途：

- 提供不同运动族候选。
- 支持 selector 学习 expert 权重。
- 支持失败时 fallback 到更保守 expert。

限制：

- 每个 expert 必须输出 whole-body action 或 whole-body light-axis，不允许只输出孤立肢体动作。

### 6.4 RL Rollout 扩展

在仿真中让小脑参与 rollout，扩展长尾场景。

用途：

- 学习候选光轴的任务价值。
- 训练 selector 的风险-收益权衡。
- 搜索 teacher 未覆盖的恢复路径。

限制：

- 不能让 RL 直接绕过 follower 输出关节命令。
- 不能用平均 reward 掩盖 fallback 突变。

### 6.5 Offline 高保真或真实片段

来自线下分支。

用途：

- 校准危险信号。
- 校准模型分歧。
- 修正 sim-to-real 偏差。
- 构建 release gate 的高风险样本。

限制：

- 线下样本不阻塞线上推进，但必须进入学校经验池。

## 7. 学校数据集

Phase 3 学校系统需要构建以下数据集。

### 7.1 Dataset Families

```text
dataset_phase3_latent_dynamics
dataset_phase3_generator_pretrain
dataset_phase3_selector_gate
dataset_phase3_dangerous_signal
dataset_phase3_model_disagreement
dataset_phase3_release_gate
```

### 7.2 样本类型

必须包含：

```text
clean_whole_body_success
upper_body_breaks_locomotion
locomotion_breaks_upper_body
uncoordinated_com_shift
contact_pattern_disruption
abrupt_upper_body_fallback
mode_switch_success
mode_switch_failure
fallback_transition
model_disagreement
command_unreachable
offline_high_fidelity_case
```

### 7.3 Segment Metadata 扩展

Phase 3 sample envelope 增加：

```json
{
  "cerebellum": {
    "candidate_axis_count": 8,
    "selected_axis_index": 0,
    "stable_model_weight": 1.0,
    "candidate_model_weight": 0.0,
    "expert_weights": [],
    "fallback_mode": "none",
    "axis_confidence_min": 0.8,
    "axis_risk_peak": 0.2
  }
}
```

### 7.4 优先级评分

Phase 3 priority:

```text
priority_score =
  0.25 * dangerous_signal_score +
  0.20 * model_disagreement_score +
  0.15 * fallback_transition_score +
  0.15 * coordination_failure_score +
  0.10 * novelty_score +
  0.10 * task_error_score +
  0.05 * data_quality_score
```

说明：

- `dangerous_signal_score` 强调危险提前量不足的片段。
- `model_disagreement_score` 强调 stable/candidate/expert 分歧。
- `fallback_transition_score` 强调进入和退出 fallback 的连续性。
- `coordination_failure_score` 继承 Phase 2.5 的上下肢互扰指标。

## 8. 训练阶段

Phase 3 分成六个训练阶段。

### 8.1 Phase 3-A：Non-Semantic Latent Dynamics

目标：从无语义状态-动作序列学习短期动力学和全身协调隐变量。

输入：

- `obs_public`
- action。
- contact。
- command mask。
- morphology input。
- history。

预测目标：

- 未来 `projected_gravity`。
- 未来 `base_ang_vel`。
- 未来 joint velocity residual。
- 未来 foot contact。
- 未来 end-effector residual。
- dangerous signal precursor。

损失：

```text
L_dynamics =
  w_state     * L_state_residual
+ w_contact   * BCE_contact
+ w_ee        * L_end_effector_residual
+ w_risk      * BCE_risk_precursor
+ w_smooth_z  * L_latent_smoothness
```

规则：

- 不使用“walk/reach/avoid”等语义标签。
- 可以使用 command 数值和 mask。
- 目标是学习身体协调规律，而不是任务分类。

### 8.2 Phase 3-B：Generator Pretraining

目标：让 generator 从 context 产生多个可行 light-axis candidates。

训练来源：

- clean whole-body success。
- recovery_success。
- follower teacher trajectories。
- MoE expert trajectories。

损失：

```text
L_generator =
  w_reconstruct * L_light_axis_reconstruct
+ w_diversity    * L_candidate_diversity
+ w_validity     * L_candidate_validity
+ w_teacher      * L_teacher_alignment
+ w_coordination * L_whole_body_coordination
```

关键要求：

- candidates 需要多样，但不能生成明显不可执行路径。
- 不同 candidate 应覆盖不同风险-收益选择。
- 每条 candidate 必须保持 whole-body 耦合。

### 8.3 Phase 3-C：Selector/Gate Supervised Training

目标：训练 selector 在候选光轴中选择、混合和 fallback。

监督来源：

- release gate 成功/失败标签。
- teacher/follower success。
- dangerous signal。
- stable/candidate model disagreement。
- fallback transition 成败。

损失：

```text
L_selector =
  w_success     * CE_or_BCE_success
+ w_risk        * BCE_risk
+ w_rank        * L_candidate_ranking
+ w_gate_smooth * L_gate_temporal_smoothness
+ w_fallback    * CE_fallback_mode
+ w_disagree    * L_model_disagreement_calibration
```

关键要求：

- selector 不只选最高 reward candidate，也要考虑危险和 fallback 连续性。
- gate weight 变化需要时间平滑。
- low confidence 时必须倾向 conservative fallback。

### 8.4 Phase 3-D：Closed-Loop RL Fine-Tuning

目标：在仿真中让小脑参与闭环，优化长时任务表现。

约束：

- 小脑输出 structured light-axis command。
- follower 执行动作。
- 小脑不得直接输出关节命令绕过 follower。

奖励：

```text
r_cerebellum =
  r_task_success
+ r_follower_tracking
+ r_fallback_smoothness
+ r_dangerous_signal_lead
- p_fall
- p_collision
- p_hard_switch
- p_unreachable_axis
- p_confidence_miscalibration
```

重点：

- 提高任务成功率。
- 降低危险信号迟报。
- 降低硬切换和 fallback 突变。
- 提升候选模型接管时的稳定性。

### 8.5 Phase 3-E：Teacher/MoE Distillation

目标：把高精度 follower、MoE expert 或强 baseline 的可执行行为蒸馏到 generator/selector。

蒸馏对象：

- candidate light-axis distribution。
- selected axis。
- fallback decision。
- confidence calibration。
- dangerous signal。

损失：

```text
L_distill =
  w_axis      * L_axis_distribution
+ w_action    * L_follower_action_consistency
+ w_gate      * L_gate_policy
+ w_risk      * L_risk_prediction
+ w_cap       * L_capability_boundary
```

限制：

- 如果 teacher 在某类任务上失败，不能把失败行为当成正样本。
- teacher 数据必须带 capability boundary。

### 8.6 Phase 3-F：Release Gate Calibration

目标：对候选小脑版本做发布前校准。

内容：

- replay release gate。
- online simulation stress test。
- stable/candidate shadow inference。
- fallback abruptness test。
- dangerous signal lead time test。
- Phase 1/2 regression。

输出：

- cerebellum model manifest。
- release package。
- capability summary。
- known failure modes。

## 9. Generator 设计候选

### 9.1 Deterministic Multi-Candidate Generator

单网络输出 K 个 candidates。

优点：

- 简单。
- 易评测。
- 适合 Phase 3 初版。

风险：

- candidate diversity 不足。

### 9.2 CVAE Generator

用条件 VAE 建模候选光轴分布。

优点：

- 能表达多模态路径可能性。
- 适合无语义动作序列。

风险：

- latent collapse。
- 训练和评测复杂度更高。

### 9.3 Diffusion-Style Generator

对 light-axis sequence 做去噪生成。

优点：

- 能生成平滑、多样轨迹。

风险：

- 推理成本高。
- Phase 3 初期可能过重。

### 9.4 Expert-Proposal Generator

由多个 follower/MoE expert 提供候选。

优点：

- 候选可执行性强。
- 便于接入 Phase 2.5 结果。

风险：

- 受 expert 覆盖范围限制。
- 未见场景泛化弱。

## 10. Selector/Gate 设计候选

### 10.1 Score-Based Selector

对每个 candidate 输出 score，选择最高或加权混合。

评分维度：

- task score。
- risk score。
- confidence。
- follower feasibility。
- model disagreement。
- fallback cost。

### 10.2 Pairwise Ranking Selector

学习 candidate 之间的相对优劣。

适合：

- 多个 candidate 都可行，但风险/收益不同。
- 有 release gate 对照样本。

### 10.3 Temporal Gate

使用小型 recurrent/causal module 平滑 gate。

目标：

- 避免每步硬切换。
- 捕捉风险积累。
- 让 fallback 进入和退出连续。

### 10.4 Safety-Biased Selector

在危险信号高时优先保守。

规则：

- `fall_risk > threshold` 时禁止 candidate model 接管。
- `confidence < threshold` 时降低 high-risk axis 权重。
- `model_disagreement` 高时进入 stable_model 或 brace fallback。

该 selector 可以作为 baseline，与学习型 selector 对照。

## 11. 消融实验矩阵

Phase 3 至少比较以下方案。

| 方案 | Generator | Selector | Teacher | MoE | RL Fine-Tune | 高保真校准 |
| --- | --- | --- | --- | --- | --- | --- |
| A | deterministic | score-based | no | no | no | no |
| B | deterministic | temporal gate | yes | no | no | no |
| C | CVAE | temporal gate | yes | no | no | no |
| D | expert-proposal | temporal gate | yes | yes | no | no |
| E | expert-proposal | temporal gate | yes | yes | yes | no |
| F | expert-proposal | temporal gate | yes | yes | yes | yes |

必须包含 baseline：

- No cerebellum：直接 follower。
- Rule-based gate：规则 gate + stable/candidate follower。
- Stitched gate：任务模式硬切换。

结论不能只看平均任务成功率，必须同时比较 fallback 连续性、危险信号提前量和回归风险。

## 12. 评测指标

### 12.1 任务指标

- task success rate。
- command tracking RMSE。
- wrist RMSE。
- hover distance error。
- avoidance violation rate。
- mode switch success rate。

### 12.2 稳定性指标

- fall rate。
- near-fall rate。
- base height RMSE。
- projected gravity error。
- contact disruption rate。
- self-collision rate。

### 12.3 光轴质量指标

- candidate validity rate。
- candidate diversity。
- selected axis smoothness。
- whole-body coordination score。
- latent discontinuity score。
- unreachable axis rate。

### 12.4 Gate/Fallback 指标

- hard switch count。
- fallback abruptness score。
- fallback recovery success rate。
- stable/candidate takeover ratio。
- model disagreement calibration error。
- gate temporal smoothness。

### 12.5 Dangerous Signal 指标

- dangerous signal lead time。
- false negative rate。
- false positive rate。
- risk calibration error。
- low confidence detection rate。

### 12.6 回归指标

- Phase 1 locomotion regression。
- Phase 2 whole-body regression。
- follower tracking regression。
- school release gate regression。

## 13. Online 验收门槛

Phase 3 online 验收以仿真为主。

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

这些门槛是进入 Phase 4 的最低门槛，不代表最终性能目标。

## 14. Release Gate

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

### 14.1 Release Package 扩展

```json
{
  "schema_version": "release_package.v0",
  "model_id": "cerebellum_candidate_phase3_v001",
  "model_family": "cerebellum_generator_selector",
  "fallback": {
    "cerebellum_model_id": "cerebellum_stable_phase3_v000",
    "follower_model_id": "follower_stable_phase2_v001",
    "required": true
  },
  "gate_recommendation": {
    "initial_candidate_axis_weight": 0.0,
    "max_candidate_axis_weight": 0.25,
    "blocked_conditions": [
      "fall_risk > 0.5",
      "confidence < 0.4",
      "model_disagreement > 0.6",
      "unreachable_risk > 0.5"
    ]
  }
}
```

## 15. Offline 分支

Phase 3 online 通过后，创建 offline 分支，但不阻塞 Phase 4。

offline 分支验证：

- 光轴输出对真实或高保真 follower 的可执行性。
- dangerous signal 在执行器延迟下是否仍有提前量。
- fallback 是否在真实动力学下平滑。
- expert/gate 切换是否产生动作突变。
- contact 和 torque 风险是否被低估。

offline failure 回流为：

- `offline_high_fidelity_case`
- `dangerous_signal_late`
- `fallback_abrupt`
- `expert_switch_instability`
- `unreachable_axis`
- `sim_to_real_axis_bias`

这些样本进入 Phase 4 release gate 和后续小脑再训练。

## 16. 与 Phase 4 本地大脑的交接

Phase 3 完成后，向 Phase 4 提供：

- 小脑 model manifest。
- light-axis command schema。
- generator/selector capability summary。
- known failure modes。
- fallback modes。
- accepted high-level hint fields。
- rejected or risky intent conditions。
- release gate dataset。

Phase 4 本地大脑输出语义意图时，不应直接控制 follower，而应通过小脑输入接口：

```json
{
  "semantic_intent_hint": {
    "goal": {},
    "speed_preference": "normal",
    "force_preference": "neutral",
    "avoidance_keypoints_base_m": [],
    "task_priority": 0.5,
    "risk_tolerance": 0.2
  }
}
```

小脑负责把这些高层 hint 映射成候选光轴和 gate 决策。

## 17. 推荐文件布局

后续实现时，建议增加：

```text
configs/
  train/
    phase3_latent_dynamics.yaml
    phase3_generator.yaml
    phase3_selector_gate.yaml
    phase3_rl_finetune.yaml
  eval/
    phase3_light_axis_scenarios.yaml
    phase3_release_gate.yaml
  school/
    phase3_dataset_queries.yaml
    phase3_sample_scoring.yaml

src/
  northstar/
    cerebellum/
      context_encoder.py
      generator.py
      selector_gate.py
      light_axis_decoder.py
      losses.py
    training/
      phase3_latent_dynamics_runner.py
      phase3_generator_runner.py
      phase3_selector_runner.py
      phase3_rl_runner.py
    school/
      phase3_sample_scorer.py
      phase3_dataset_builder.py
    eval/
      phase3_light_axis_eval.py
      phase3_release_gate_eval.py

tests/
  cerebellum/
    test_light_axis_schema.py
    test_generator_shapes.py
    test_selector_gate_outputs.py
    test_light_axis_decoder.py
  school/
    test_phase3_sample_scorer.py
  eval/
    test_phase3_light_axis_eval.py
```

## 18. 风险与缓解

### 18.1 光轴学不出全身协调

风险：latent 只编码局部动作，不能牵动全身。

缓解：使用 whole-body coordination loss、COM/contact disruption 指标、全身 action reconstruction 和 Phase 2.5 failure samples。

### 18.2 Generator 多样但不可执行

风险：候选很多，但 follower 无法稳定执行。

缓解：candidate validity loss、follower feasibility prediction、release gate 中加入 unreachable axis rate。

### 18.3 Selector 只学平均成功率

风险：selector 忽视危险提前量和 fallback 连续性。

缓解：risk loss、fallback abruptness loss、dangerous signal lead-time 指标。

### 18.4 Teacher 误导小脑

风险：teacher 在某些任务族失败，小脑仍蒸馏其行为。

缓解：teacher 数据必须绑定 capability boundary；失败任务只能作为 negative 或 risky sample。

### 18.5 MoE Gate 突变

风险：expert 切换导致动作不连续。

缓解：temporal gate、gate smoothness loss、expert switch continuity release gate。

### 18.6 RL Fine-Tune 绕过接口

风险：RL 直接优化低层动作，破坏小脑/follower 分层。

缓解：Phase 3 RL 只允许输出 structured light-axis command，底层 action 仍由 follower 产生。

### 18.7 高保真失败未进入训练

风险：offline 不阻塞 online，真实偏差被忽略。

缓解：offline failure 默认高 priority，并进入 Phase 3/4 release gate。

## 19. 完成定义

Phase 3 小脑光轴设计完成，需要满足：

1. 定义并验证 light-axis command schema。
2. 完成 non-semantic latent dynamics 训练或基线。
3. 完成 generator pretraining。
4. 完成 selector/gate supervised training。
5. 至少完成一轮消融矩阵对照。
6. 小脑闭环输出不绕过 follower。
7. dangerous signal lead time 达到最低门槛。
8. fallback abruptness 通过 release gate。
9. Phase 1/2 regression 通过。
10. 生成 Phase 3 capability summary 和 known failure modes。
11. 明确 Phase 4 本地大脑可以输入哪些 semantic hint，以及哪些 intent 条件应被拒绝或降级。

