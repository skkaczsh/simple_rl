# Mimic Motion Prior 集成与应用测试设计

日期：2026-05-01

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [Phase 1 基础运动底座与学校最小闭环设计](./2026-04-23-phase-1-locomotion-school-loop-design.md)
- [Whole-Body Follower 统一策略设计](./2026-04-23-whole-body-follower-unified-policy-design.md)
- [School System 数据、训练与发布设计](./2026-04-23-school-system-data-and-release-design.md)
- [小脑光轴学习与消融设计](./2026-04-23-cerebellum-light-axis-learning-design.md)
- [Model Gate、Fallback 与版本切换设计](./2026-04-23-model-gate-fallback-design.md)
- [线上/线下验证树与指标设计](./2026-04-23-validation-tree-and-metrics-design.md)

## 1. 目标

本文定义主流 Mimic 系列算法、开源框架、运动数据和预训练模型如何进入北极星技术路线，并确认可进行应用测试的区域。

这里的 Mimic 系列包括但不限于：

- DeepMimic：基于参考动作的物理角色模仿强化学习。
- AMP：用对抗式 motion prior 约束动作自然性。
- ASE：学习可复用 skill latent 或动作技能嵌入。
- MaskedMimic：基于 masked motion inpainting 的统一物理角色控制。
- ProtoMotions：面向 humanoid/robot 的大规模 motion learning、retargeting 和多仿真后端框架。
- ResMimic：使用 general motion tracking policy 作为底座，再训练 residual policy 支持 whole-body loco-manipulation。

本文的核心结论：

1. Mimic 系列可以作为北极星训练系统的重要开源组件和方法来源。
2. Mimic 系列适合进入学校系统、follower 训练、Phase 2.5 统一策略验证、Phase 3 小脑 teacher/expert 和离线回归评测。
3. Mimic 系列不应直接绕过 NorthStar ABI、episode logger、school sample、release gate、gate/fallback 和 public-only deployment 约束。
4. 预训练模型和公开数据集可以用于研究验证，但商业化或产品化使用必须逐项确认许可证。

## 2. 范围

本文覆盖：

- Mimic 系列在 Phase 1、Phase 2、Phase 2.5、Phase 3、School、Gate/Fallback 和 Offline 分支中的应用测试位置。
- 可复用开源组件和算法类型。
- teacher、motion prior、expert、baseline、regression oracle 的角色定义。
- 数据、retarget、仿真器、ABI 和许可证约束。
- 与现有北极星文档的交接规则。

本文不覆盖：

- 具体训练脚本实现。
- 具体模型结构超参数。
- AMASS、HumanML3D 或其他数据集的授权谈判。
- 将第三方预训练模型直接部署到真实机器人。
- 完整商业化合规审查。

## 3. 设计原则

### 3.1 Mimic 是训练资产，不是运行时捷径

Mimic policy、motion prior 或 pretrained model 可以帮助训练北极星 follower、小脑或 expert，但不能绕过 NorthStar runtime contract。

所有进入执行路径的模型必须满足：

- NorthStar ABI input/output。
- public-only deployable path。
- episode replay。
- schema validation。
- school sample traceability。
- gate/fallback。
- release gate。

### 3.2 Teacher 可以用 privileged 信息，Student 不能依赖它

Mimic teacher 可以使用未来动作、完整 motion clip、heightmap、仿真真值或 privileged observation。

部署到本地执行路径的 student/follower 必须声明输入依赖。若模型依赖 privileged 信息，则只能标记为 `teacher_only`、`oracle_only` 或 `research_only`，不得发布为 stable/candidate runtime model。

### 3.3 先离线蒸馏，再影子推理，最后低比例接管

第三方 Mimic 模型即使表现优秀，也必须按以下顺序进入系统：

1. 离线 retarget 和 rollout。
2. 转换为 NorthStar episode log。
3. 生成 school sample。
4. 训练或蒸馏 NorthStar student。
5. shadow inference。
6. low-risk gated takeover。
7. release gate。

### 3.4 运动自然性不等于任务安全

Mimic 系列擅长动作自然性、全身协调和模仿泛化，但这不自动保证：

- dangerous signal 提前量。
- fallback 连续性。
- 真实执行器延迟下的稳定性。
- 与本地大脑语义意图的一致性。
- 商业或真实机器人安全。

### 3.5 许可证是架构约束

开源代码、预训练 checkpoint 和 motion dataset 必须分开审查许可证。代码可用不代表数据和模型可用于同一目标。

## 4. 主流 Mimic 组件定位

| 组件 | 当前定位 | 北极星可用角色 | 主要风险 |
| --- | --- | --- | --- |
| DeepMimic | 经典动作模仿 RL 方法，原仓库已建议转向 MimicKit | 单 clip imitation baseline、reward 参考、早期教学样例 | 工程栈较旧，动作覆盖有限 |
| AMP | 对抗式动作先验 | motion naturalness prior、动作质量 discriminator | 容易提升自然性但不保证任务成功 |
| ASE | skill latent / reusable skill embedding | skill prior、MoE expert latent、Phase 3 candidate prior | latent 与 NorthStar light-axis 语义不一定一致 |
| MimicKit | 轻量 motion imitation 方法集合，包含 DeepMimic、AMP、ASE、ADD、SMP 等 | 研究 baseline、算法复现、训练方法库 | 与目标仿真器/机器人形态仍需适配 |
| ProtoMotions | GPU humanoid/robot motion learning 框架，支持 AMASS retarget、多仿真后端、G1 等 | retarget pipeline、G1 motion tracker、sim2sim、MaskedMimic 训练入口 | 依赖栈重，数据许可证需单独确认 |
| MaskedMimic | 从部分约束生成全身动作的 unified controller 路线 | Phase 2.5 unified policy baseline、teacher、partial constraint oracle | 预训练 SMPL 模型许可证和形态迁移需审查 |
| ResMimic | GMT base + residual task policy 的 whole-body loco-manipulation 路线 | Phase 2/2.5 residual post-training 参考、object-conditioned correction | 需要复现或授权代码/模型，任务侧工程复杂 |

## 5. 应用测试区域总览

| 北极星区域 | Mimic 应用方式 | 测试目标 | 通过信号 |
| --- | --- | --- | --- |
| Phase 1 locomotion | DeepMimic/AMP/ProtoMotions tracking policy 作为 teacher 或 pretrain | 加速基础运动 follower 学习 | public-only student 指标不差于 scratch baseline |
| Phase 1 school loop | Mimic rollout 转成 episode log 和 school envelope | 验证外部 teacher 数据能进入学校闭环 | schema/replay/envelope validation 100% |
| Phase 2 whole-body | MaskedMimic/ProtoMotions 产生 reach/carry/hover tracking teacher | 验证上下肢全身协调训练收益 | wrist error 下降且 locomotion 不明显退化 |
| Phase 2.5 unified policy | Mimic-style unified policy 与 shared trunk、MoE、stitched baseline 对照 | 判断统一 follower 是否成立 | Phase 2.5-A/B/C 对照矩阵给出明确决策 |
| Phase 3 cerebellum | Mimic policy 作为 motion prior、teacher、expert、oracle | 改善 light-axis 候选质量和动作自然性 | candidate validity、fallback smoothness、danger lead time 提升 |
| School System | 收集 retarget failure、teacher disagreement、motion prior failure | 丰富高价值训练片段 | priority 和 segment type 可追溯 |
| Gate/Fallback | Mimic candidate 做 shadow inference 和低比例接管 | 验证外部 candidate 不破坏稳定模型 | hard switch count、fallback abruptness、fall rate 受控 |
| Offline/Sim2Sim | ProtoMotions/MaskedMimic 在 IsaacLab、Newton、MuJoCo、Genesis 或高保真分支测试 | 识别 sim-to-sim / sim-to-real 偏差 | 失败进入 release gate 和 capability summary |

## 6. Phase 1 应用测试

### 6.1 基础运动预训练

Phase 1 可以使用 Mimic 方法进行基础 locomotion 预训练，但发布模型必须是 public-only follower。

测试组：

- scratch PPO/RSL-RL follower。
- Mimic pretrain + RL fine-tune。
- privileged mimic teacher -> public-only student。
- AMP-style motion prior + task reward。

输出：

- `follower_loco_mimic_pretrain_v001`
- `follower_loco_mimic_student_v001`
- `mimic_teacher_rollout_dataset_phase1_v001`

验收指标：

- Phase 1 online gate 不低于 scratch baseline。
- public-only student 能独立运行。
- teacher-only 模型不得进入 stable/candidate runtime release。
- replay validation pass rate 为 100%。

### 6.2 学校最小闭环

Mimic rollout 必须转换为 NorthStar episode 格式：

```text
mimic source rollout
  -> retargeted robot trajectory
  -> NorthStar observation/action/command records
  -> episode manifest
  -> replay metrics
  -> school sample envelope
```

新增 segment type：

```text
mimic_teacher_success
mimic_teacher_failure
mimic_student_disagreement
retargeting_failure
motion_prior_outlier
privileged_teacher_only_case
```

## 7. Phase 2 与 Phase 2.5 应用测试

### 7.1 Whole-Body Teacher

Phase 2 可以让 MaskedMimic/ProtoMotions 风格的 tracking policy 生成全身 teacher 信号：

- walking + reach。
- carry posture。
- pre-contact hover。
- avoidance keypoints。
- posture hold。
- stop/brace with upper body task。

teacher 输出不能直接当 NorthStar action 使用，必须经过：

1. 形态 retarget。
2. action adapter。
3. ABI validator。
4. replay reader。
5. school sample builder。

### 7.2 Phase 2.5 对照矩阵扩展

Phase 2.5 的 candidate matrix 增加 Mimic 路线：

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

决策规则：

1. 如果 Mimic teacher-student 显著提升全身协调，但 public-only student 达不到 gate，Mimic 只保留为 teacher/oracle。
2. 如果 Mimic prior 提升动作自然性但引入 fall 或 fallback abruptness，不能作为 runtime candidate。
3. 如果 Mimic-style unified policy 在 Phase 2.5-A/B/C 全部通过，可作为 Phase 3 默认 follower candidate 之一。

## 8. Phase 3 小脑应用测试

Mimic 系列在 Phase 3 不直接替代小脑，而是给小脑提供运动先验、候选分布和对照信号。

### 8.1 Motion Prior

使用 Mimic rollout 学习：

- pose trend prior。
- velocity trend prior。
- contact transition prior。
- recovery motion prior。
- style/naturalness discriminator。

这些 prior 只能影响 generator 候选评分或训练 loss，不能直接绕过 selector/gate。

### 8.2 Teacher

高质量 Mimic follower 可提供：

- selected light-axis target。
- follower action target。
- confidence label。
- unreachable label。
- fallback-needed label。

但 teacher 的 privileged 信息必须记录在 `model_manifest` 和 `dataset_manifest` 中。

### 8.3 Expert

Mimic policy 可作为 MoE expert 候选，但必须声明能力边界：

```json
{
  "expert_id": "mimic_gmt_expert_v001",
  "source": "mimic_teacher",
  "role": "whole_body_motion_prior",
  "runtime_allowed": false,
  "teacher_allowed": true,
  "capability_boundary": {
    "terrain": "flat_or_known_heightmap",
    "max_payload_kg": 0.0,
    "requires_future_pose": true,
    "requires_privileged_observation": true
  }
}
```

只有当 `runtime_allowed=true` 且通过 release gate 时，expert 才能进入本地执行路径。

## 9. School System 扩展

### 9.1 Mimic Source Manifest

学校样本需要记录 Mimic 来源：

```json
{
  "schema_version": "mimic_source_manifest.v0",
  "source_id": "mimic_source_000001",
  "source_type": "mimic_teacher_rollout",
  "framework": "protomotions",
  "algorithm_family": "masked_mimic",
  "license_class": "research_or_review_required",
  "motion_dataset_ids": ["amass"],
  "robot_target": "unitree_g1_43dof_sim_v0",
  "retarget_pipeline_id": "retarget_g1_v001",
  "simulator": "isaaclab",
  "privileged_inputs": [
    "future_pose",
    "heightmap"
  ],
  "runtime_allowed": false
}
```

### 9.2 Dataset Split

Mimic-derived samples must be tagged so release gate and training split remain clean:

- `mimic_train`
- `mimic_distill`
- `mimic_validation`
- `mimic_release_gate`
- `non_mimic_release_gate`

Release gate 不能只由 Mimic-derived samples 构成。至少需要包含 NorthStar native rollout、stable model replay 和 offline/high-fidelity failure cases。

### 9.3 Priority 扩展

新增 priority signal：

```text
mimic_value_score =
  0.30 * teacher_success_score +
  0.25 * student_disagreement_score +
  0.20 * retargeting_gap_score +
  0.15 * rare_motion_score +
  0.10 * motion_quality_score
```

Mimic 样本总优先级：

```text
priority_score =
  0.70 * base_priority_score +
  0.30 * mimic_value_score
```

## 10. Gate 与 Fallback 约束

Mimic candidate 必须默认从 0 接管比例开始。

推荐 gate 初始规则：

```json
{
  "mimic_candidate_gate": {
    "initial_weight": 0.0,
    "max_shadow_only_weight": 0.0,
    "max_low_risk_takeover_weight": 0.10,
    "max_staged_takeover_weight": 0.25,
    "blocked_conditions": [
      "requires_privileged_observation == true",
      "runtime_allowed == false",
      "license_class != runtime_approved",
      "retargeting_gap_score > 0.25",
      "model_disagreement > 0.4",
      "dangerous_sig.fall_risk > 0.3",
      "fallback_abruptness_score > 0.2"
    ]
  }
}
```

Mimic candidate 不得覆盖：

- emergency stop。
- brace fallback。
- stable model fallback。
- action limit clipping。
- dangerous signal hard block。

## 11. Offline 与 Sim2Sim 应用测试

Mimic 系列尤其适合做 offline/sim2sim 偏差识别。

测试维度：

- IsaacLab -> Newton。
- IsaacLab -> MuJoCo。
- IsaacLab -> Genesis。
- SMPL -> Unitree G1 retarget。
- human mocap -> robot action。
- flat terrain -> heightmap terrain。
- no payload -> payload。
- no contact object -> object contact。

失败类型：

```text
mimic_sim2sim_gap
mimic_retargeting_gap
mimic_contact_mismatch
mimic_future_pose_dependency
mimic_runtime_observation_gap
mimic_license_blocked_release
```

这些失败默认进入：

- risk register。
- school sample pool。
- capability summary。
- Phase 2.5 或 Phase 3 release gate。

## 12. 许可证与合规边界

当前公开信息显示：

- MimicKit 提供 DeepMimic、AMP、ASE、ADD、SMP 等实现，仓库标注 Apache-2.0。
- ProtoMotions 标注 Apache-2.0，并提供 AMASS retarget、多机器人训练、sim2sim、G1 相关流程。
- MaskedMimic 预训练模型卡标注 `ncsl`，并声明模型为 SMPL humanoid、在 IsaacLab 中训练。
- AMASS license 面向 non-commercial scientific research，不能默认用于商业训练或产品化模型。

因此默认策略：

1. Apache-2.0 代码可以进入研究与内部工程验证。
2. AMASS、HumanML3D、MaskedMimic checkpoint 等数据或模型必须进入 license review。
3. 未通过许可证审查的模型只能标记 `research_only`、`teacher_only` 或 `oracle_only`。
4. 任何源自受限数据或 checkpoint 的 runtime release package 必须记录 lineage。

## 13. 推荐文件布局

后续实现时建议增加：

```text
configs/
  mimic/
    mimic_sources.yaml
    phase1_mimic_pretrain.yaml
    phase2_mimic_teacher.yaml
    phase3_mimic_prior.yaml
src/
  northstar/
    mimic/
      __init__.py
      source_manifest.py
      retarget_adapter.py
      rollout_importer.py
      teacher_dataset.py
      license_policy.py
tests/
  mimic/
    test_source_manifest.py
    test_rollout_importer.py
    test_teacher_dataset.py
```

该布局只表示后续 implementation plan 的方向，不要求 Phase 1 skeleton 立即实现。

## 14. 应用测试路线

### 14.1 Test A：Mimic Rollout Ingress

目标：验证外部 Mimic rollout 可以被转换为 NorthStar episode 和 school sample。

通过条件：

- schema validation pass rate 100%。
- replay validation pass rate 100%。
- school sample envelope validation pass rate 100%。
- source manifest 记录完整。

### 14.2 Test B：Phase 1 Mimic Teacher Student

目标：验证 Mimic pretrain 或 teacher distillation 是否提升 locomotion 学习效率。

通过条件：

- public-only student 达到 Phase 1 online gate。
- 与 scratch baseline 相比，训练步数或失败率有明确改善。
- teacher-only 模型不会进入 runtime release。

### 14.3 Test C：Phase 2 Whole-Body Teacher

目标：验证 Mimic teacher 是否能提升 whole-body reach/carry/hover 协调。

通过条件：

- wrist RMSE 下降。
- locomotion degradation ratio 不超过 Phase 2.5 门槛。
- fallback abruptness 不升高。
- 自碰撞和接触扰动不恶化。

### 14.4 Test D：Phase 2.5 Unified Policy Comparison

目标：把 Mimic-style unified policy 纳入统一策略对照矩阵。

通过条件：

- 与 unified/shared trunk/MoE/stitched baseline 产出同一指标表。
- 明确判断 Mimic 路线是 runtime candidate、teacher-only、expert-only 还是 rejected。

### 14.5 Test E：Phase 3 Mimic Prior for Light-Axis

目标：验证 Mimic motion prior 是否改善小脑候选光轴质量。

通过条件：

- candidate validity rate 提升。
- unreachable axis rate 下降。
- dangerous signal lead time 不下降。
- Phase 1/2 regression pass。

## 15. 完成定义

本规格完成后，后续实现或实验计划应满足：

1. 明确 Mimic 系列在 Phase 1/2/2.5/3/School/Gate/Offline 的应用测试位置。
2. 区分 Mimic 的 teacher、prior、expert、baseline、oracle 和 runtime candidate 角色。
3. 定义 `mimic_source_manifest.v0` 和 Mimic-derived school sample 标签。
4. 定义 Mimic candidate 的 gate/fallback 默认限制。
5. 明确许可证审查是 release package 的前置条件。
6. 明确 Mimic 模型不能绕过 NorthStar ABI 与 release gate。

## 16. 参考资料

- [MimicKit](https://github.com/xbpeng/MimicKit)
- [ProtoMotions](https://github.com/NVLabs/ProtoMotions)
- [ProtoMotions Documentation](https://nvlabs.github.io/ProtoMotions/)
- [MaskedMimic model card](https://huggingface.co/ctessler/MaskedMimic)
- [DeepMimic](https://github.com/xbpeng/DeepMimic)
- [ResMimic project](https://resmimic.github.io/)
- [AMASS license](https://amass.is.tue.mpg.de/license.html)
