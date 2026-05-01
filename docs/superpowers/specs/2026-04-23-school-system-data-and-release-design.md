# School System 数据、训练与发布设计

日期：2026-04-23

状态：设计草案，等待用户审阅

上游文档：

- [北极星具身智能技术架构蓝图](./2026-04-23-north-star-embodied-architecture-design.md)
- [Phase 0 ABI 与基础设施设计](./2026-04-23-phase-0-abi-and-infra-design.md)
- [Mimic Motion Prior 集成与应用测试设计](./2026-05-01-mimic-motion-prior-integration-design.md)

## 1. 目标

学校系统是北极星架构的横向核心层，从 Phase 1 开始贯穿所有阶段。它不是云端大脑，也不是后期附加模块。

学校系统的目标是：

1. 接收本地或仿真客户端精炼后的高价值经验，而不是接收全量原始数据。
2. 维护可查询、可回放、可分层采样的学校经验池。
3. 基于阶段目标训练、聚合和评测小脑 + whole-body follower 相关模型。
4. 发布候选模型版本，并支持本地双模型并行、gate、灰度接管和 fallback。
5. 汇总模型能力边界、失败模式和评测摘要，弱通信给云端大脑。
6. 把线上仿真和线下高保真/真实验证结果统一回流到训练与评测闭环中。

学校系统的价值不只是“存数据”，而是把高价值片段、训练任务、评测门槛和模型发布串成可审计的进化循环。

## 2. 范围

本文覆盖：

- 本地经验精炼与上传 envelope。
- 学校 ingress、校验、去重和优先级处理。
- 学校经验池、数据集版本和可回放样本。
- 训练任务、聚合任务和候选模型注册。
- 评测、回归、能力边界摘要和失败模式摘要。
- 模型发布、灰度、fallback 和回滚反馈。
- 学校系统与云端大脑的弱通信接口。

本文不覆盖：

- 云端大脑内部规划算法。
- 本地大脑 VLM/VLA/VLN 训练。
- 完整开放环境安全认证。
- 商业化用户、权限、计费、运维平台。
- 具体 Isaac Lab 训练脚本实现。
- 具体联邦优化算法的最终选择。

具体算法可以在阶段 implementation plan 或实验设计中细化，但学校系统的数据契约和发布契约应从 Phase 1 起稳定。

## 3. 设计原则

### 3.1 本地先精炼，学校再聚合

客户端不上传全量原始流。每个客户端先在本地筛选、压缩、评分和标注片段，再上传结构化样本。

### 3.2 样本必须可追溯

每个学校样本必须能追溯到 run、episode、step range、ABI version、embodiment id、模型版本和触发原因。

### 3.3 发布必须可回退

学校发布新模型时，必须绑定 fallback model、评测报告、schema hash 和 gate 建议。候选模型不能无条件覆盖本地稳定模型。

### 3.4 评测优先于训练指标

训练 loss 或平均 reward 不能直接作为发布依据。候选模型必须通过阶段回归、失败场景回放和能力边界检查。

### 3.5 学校与云端大脑弱通信

学校只向云端大脑提供能力摘要、版本摘要和失败模式摘要。云端大脑不控制训练循环，也不直接查询学校 replay buffer 做在线任务决策。

## 4. 核心模块

### 4.1 Local Refiner

运行在仿真客户端或机器人本地。负责从 episode log 中筛选高价值片段、计算优先级分数、压缩 payload，并生成上传 envelope。

### 4.2 Upload Queue

本地上传队列。负责断点续传、去重、重试、带宽限制和本地保留策略。

### 4.3 School Ingress

学校入口。负责校验 envelope、schema、artifact hash、ABI version、embodiment id 和模型版本依赖。

### 4.4 Experience Pool

学校经验池。保存结构化样本 metadata、payload artifact、指标、事件和索引。支持按 phase、任务、风险、模型版本、失败类型、形态和优先级查询。

### 4.5 Dataset Builder

从经验池构建训练数据集版本。负责采样策略、去重、分层、平衡和 dataset manifest。

### 4.6 Trainer/Aggregator

执行训练或聚合任务。Phase 1 可以先支持集中式训练或简单聚合；后续扩展 FedAvg、SCAFFOLD、FedProx、MoE expert 训练和 gate 训练。

### 4.7 Evaluator

执行候选模型评测、回放评测、阶段回归、消融对比和线下分支结果汇总。

### 4.8 Model Registry

记录模型 manifest、artifact、schema hash、父模型、fallback 模型、训练数据、评测报告和发布状态。

### 4.9 Release Manager

管理 candidate、staged、stable、rejected、rolled_back 状态转换，并生成本地部署包。

### 4.10 Capability Summarizer

把评测结果转化为能力边界、失败模式摘要和模型版本摘要，提供给云端大脑和本地大脑。

## 5. 端到端数据流

学校系统的数据流如下：

1. 客户端执行任务，生成 episode log。
2. Local Refiner 读取 episode log，发现候选片段。
3. Refiner 对片段计算 `priority_score` 和 `selection_reasons`。
4. Refiner 生成 `school_sample_envelope` 和 payload artifact。
5. Upload Queue 上传 envelope 和 artifact。
6. School Ingress 校验 schema、hash、ABI、版本依赖和权限。
7. Experience Pool 写入 metadata 和 artifact 索引。
8. Dataset Builder 从经验池构建 dataset version。
9. Trainer/Aggregator 训练或聚合候选模型。
10. Evaluator 对候选模型做阶段评测和回归评测。
11. Model Registry 注册候选模型与评测报告。
12. Release Manager 发布 candidate package。
13. 本地运行双模型并行、gate、灰度接管和 fallback。
14. 本地切换日志、回滚日志和失败样本回传学校。
15. Capability Summarizer 更新能力边界摘要，弱通信给云端大脑。

## 6. 本地经验精炼

### 6.1 输入

Local Refiner 输入来自 Phase 0 episode logging：

- `run_manifest.json`
- `episode_manifest.json`
- `steps.parquet`
- `events.jsonl`
- `metrics.json`
- `model_manifest.json`
- 评测报告或任务上下文。

### 6.2 片段类型

Phase 1 起支持以下 `segment_type`：

```text
failure
near_failure
high_uncertainty
high_prediction_error
high_td_error
model_disagreement
fallback_transition
rare_command
recovery_success
regression_case
clean_success_reference
offline_high_fidelity_case
```

说明：

- `failure`：已经失败，例如跌倒、严重碰撞、任务终止。
- `near_failure`：危险信号高但未失败，是最有价值的学习片段之一。
- `high_uncertainty`：confidence 低或波动大。
- `model_disagreement`：候选模型与稳定模型输出差异大。
- `fallback_transition`：进入或退出 fallback 的片段。
- `clean_success_reference`：高质量成功样本，用于避免训练只看失败。
- `offline_high_fidelity_case`：线下真实或高保真分支回流样本。

### 6.3 优先级评分

每个片段生成 `priority_score`，范围 `[0.0, 1.0]`。

推荐基础公式：

```text
priority_score =
  0.25 * risk_score +
  0.20 * learning_error_score +
  0.15 * novelty_score +
  0.15 * model_disagreement_score +
  0.10 * fallback_score +
  0.10 * task_relevance_score +
  0.05 * data_quality_score
```

字段解释：

- `risk_score`：由 fall risk、collision risk、overload risk、near-failure 事件计算。
- `learning_error_score`：由 TD error、prediction error、task error 或 value error 计算。
- `novelty_score`：由 command 稀有度、状态分布稀有度、接触模式稀有度计算。
- `model_disagreement_score`：由候选模型和稳定模型的 action/light-axis 差异计算。
- `fallback_score`：由 fallback 进入、退出、持续时间和恢复结果计算。
- `task_relevance_score`：由当前 Phase 目标任务相关性计算。
- `data_quality_score`：由日志完整性、传感器有效性、schema 校验结果计算。

Phase 1 可以先用规则分数。Phase 3 以后可以用学习到的 scorer。

### 6.4 片段窗口

片段窗口应包含事件前后上下文。

默认窗口：

- `pre_context_s = 1.0`
- `post_context_s = 1.0`
- 最小长度 `0.5s`
- 最大长度 `5.0s`

规则：

- 对 `failure` 和 `near_failure`，必须包含触发前上下文。
- 对 `fallback_transition`，必须包含 fallback 进入前、过程中、退出后的状态。
- 对 `clean_success_reference`，应包含完整短任务段，而不是只截取末尾成功状态。

## 7. School Sample Envelope

Phase 0 定义了最小 envelope。学校系统扩展为以下结构：

```json
{
  "schema_version": "school_sample_envelope.v0",
  "sample_id": "sample_000001",
  "created_at": "2026-04-23T00:00:00+08:00",
  "source": {
    "client_id": "sim_client_001",
    "client_type": "simulation",
    "run_id": "run_20260423_000001",
    "episode_id": "ep_000001",
    "step_start": 100,
    "step_end": 180
  },
  "versioning": {
    "abi_version": "abi.northstar.v0",
    "embodiment_id": "unitree_g1_43dof_sim_v0",
    "embodiment_manifest_hash": "sha256:...",
    "stable_model_id": "follower_stable_v001",
    "candidate_model_id": null,
    "adapter_id": null
  },
  "segment": {
    "segment_type": "near_failure",
    "phase": "phase_1",
    "task_family": "locomotion",
    "command_family": "velocity_tracking",
    "selection_reasons": [
      "near_fall",
      "high_action_delta"
    ],
    "priority_score": 0.8
  },
  "scores": {
    "risk_score": 0.9,
    "learning_error_score": 0.6,
    "novelty_score": 0.4,
    "model_disagreement_score": 0.0,
    "fallback_score": 0.3,
    "task_relevance_score": 1.0,
    "data_quality_score": 1.0
  },
  "metrics": {
    "fall_risk_peak": 0.9,
    "confidence_min": 0.4,
    "base_height_min_m": 0.42,
    "velocity_rmse_m_s": 0.22
  },
  "artifact": {
    "uri": "school://samples/sample_000001/payload.parquet",
    "sha256": "sha256:...",
    "format": "parquet_json_columns",
    "byte_size": 123456
  },
  "labels": {
    "human_reviewed": false,
    "usable_for_training": true,
    "usable_for_regression": true,
    "usable_for_release_gate": true
  }
}
```

规则：

- `sample_id` 全局唯一。
- `source` 必须能追溯到原始 run 和 episode。
- `versioning` 必须记录模型和 ABI 依赖。
- `scores` 必须保存各子分数，不能只保存最终 priority。
- `labels.usable_for_release_gate` 为 true 的样本，可以进入候选模型发布前回归集。

## 8. Payload 格式

Phase 1 默认 payload 使用 Parquet + JSON columns，与 Phase 0 step records 兼容。

推荐 payload 包含：

```text
payload/
  segment_manifest.json
  steps.parquet
  events.jsonl
  metrics.json
  source_model_manifest.json
```

### 8.1 Segment Manifest

```json
{
  "schema_version": "segment_manifest.v0",
  "sample_id": "sample_000001",
  "time_start_s": 2.0,
  "time_end_s": 3.6,
  "control_dt_s": 0.02,
  "step_count": 80,
  "contains_privileged": true,
  "contains_candidate_outputs": false,
  "contains_adapter_state": false,
  "compression": "none",
  "redaction": {
    "applied": false,
    "rules": []
  }
}
```

### 8.2 高频字段展开策略

Phase 1 可以继续使用 JSON columns。Phase 2 以后，如果训练吞吐成为瓶颈，Dataset Builder 可以将以下字段列式展开：

- `joint_pos`
- `joint_vel`
- `base_ang_vel`
- `projected_gravity`
- `foot_contact`
- `command.locomotion`
- `action.joint_pos_delta_rad`
- `confidence.overall`
- `dangerous_sig.overall_risk`
- `reward`
- `terminated`
- `truncated`

列式展开不改变上游 envelope，只改变 dataset version 的物理格式。

## 9. Experience Pool

Experience Pool 分成 metadata index 和 artifact store。

### 9.1 Metadata Index

必须支持按以下条件查询：

- phase。
- task family。
- command family。
- embodiment id。
- ABI version。
- stable model id。
- candidate model id。
- segment type。
- priority score range。
- risk score range。
- failure event type。
- usable labels。
- created_at 时间范围。

### 9.2 Artifact Store

artifact store 保存 payload 文件。对象路径建议：

```text
school://samples/<phase>/<task_family>/<sample_id>/
```

规则：

- artifact hash 必须在 ingress 时校验。
- artifact 不允许被就地修改；修正必须生成新 sample 或新 revision。
- metadata 可以补充 review labels，但需要记录审计日志。

### 9.3 数据保留策略

默认策略：

- 高优先级 failure / near_failure：长期保留。
- 普通 success：按 reservoir sampling 保留。
- 低质量或 schema invalid：拒收或隔离。
- 重复片段：保留最高质量版本，其余写入 duplicate group。
- 被模型发布回归集引用的样本：不能自动删除。

## 10. Dataset Version

训练不直接读取松散样本，而是读取 dataset version。

```json
{
  "schema_version": "dataset_manifest.v0",
  "dataset_id": "dataset_phase1_locomotion_0001",
  "created_at": "2026-04-23T00:00:00+08:00",
  "abi_versions": ["abi.northstar.v0"],
  "embodiment_ids": ["unitree_g1_43dof_sim_v0"],
  "phase": "phase_1",
  "purpose": "follower_locomotion_training",
  "sample_query": {
    "phase": "phase_1",
    "task_family": "locomotion",
    "min_priority_score": 0.3
  },
  "sample_count": 10000,
  "split": {
    "train": 0.8,
    "validation": 0.1,
    "release_gate": 0.1
  },
  "balancing": {
    "failure_min_fraction": 0.2,
    "clean_success_min_fraction": 0.2,
    "near_failure_min_fraction": 0.2
  },
  "artifact": {
    "uri": "school://datasets/dataset_phase1_locomotion_0001",
    "sha256": "sha256:..."
  }
}
```

规则：

- 训练任务必须引用 dataset id。
- release gate split 不能被训练污染。
- 数据集版本一旦用于候选模型评测，不允许就地修改。

## 11. 训练与聚合任务

学校系统支持多种 job 类型，但 Phase 1 只需实现最小子集。

### 11.1 Job 类型

```text
follower_training
follower_distillation
adapter_aggregation
cerebellum_generator_training
cerebellum_selector_training
gate_training
moe_expert_training
dangerous_signal_training
confidence_calibration
```

### 11.2 Phase 1 最小训练任务

Phase 1 推荐最小任务：

- `follower_training`：训练基础运动 follower。
- `dangerous_signal_training` 或规则校准：改进 near-fall、overload、limit clip 预测。
- `adapter_aggregation`：仅在多个客户端有本地 adapter 时启用。

### 11.3 Training Job Manifest

```json
{
  "schema_version": "training_job_manifest.v0",
  "job_id": "train_phase1_follower_0001",
  "job_type": "follower_training",
  "phase": "phase_1",
  "base_model_id": "follower_stable_v000",
  "dataset_id": "dataset_phase1_locomotion_0001",
  "abi_version": "abi.northstar.v0",
  "embodiment_ids": ["unitree_g1_43dof_sim_v0"],
  "objective": {
    "primary": "locomotion_stability",
    "auxiliary": [
      "velocity_tracking",
      "action_smoothness",
      "fall_risk_reduction"
    ]
  },
  "algorithm": {
    "family": "ppo_or_distillation",
    "config_uri": "school://configs/train_phase1_follower_0001.yaml"
  },
  "output": {
    "candidate_model_id": "follower_candidate_v001"
  }
}
```

规则：

- job manifest 记录 algorithm family，但具体训练参数由 implementation plan 决定。
- 每个 job 必须绑定 dataset id 和 base model id。
- 输出必须先成为 candidate，不能直接成为 stable。

## 12. 候选模型评测

候选模型发布前必须经过 Evaluator。

### 12.1 评测层级

1. Schema validation：输入输出 ABI、manifest、artifact hash。
2. Replay validation：在 release gate 样本上回放候选输出。
3. Scenario validation：在线仿真场景评测。
4. Regression validation：前序 Phase 能力不能明显退化。
5. Stress validation：扰动、边界 command、near-failure 场景。
6. Gate validation：候选模型与稳定模型双模型并行时，gate 能否平滑退回。

### 12.2 Evaluation Report

```json
{
  "schema_version": "school_evaluation_report.v0",
  "report_id": "eval_follower_candidate_v001",
  "candidate_model_id": "follower_candidate_v001",
  "stable_model_id": "follower_stable_v000",
  "dataset_id": "dataset_phase1_locomotion_0001",
  "phase": "phase_1",
  "summary": {
    "pass": true,
    "recommended_release_state": "candidate",
    "regression_detected": false,
    "fallback_required": true
  },
  "metrics": {
    "fall_rate": 0.005,
    "near_fall_rate": 0.02,
    "velocity_rmse_m_s": 0.12,
    "base_height_rmse_m": 0.025,
    "mean_action_delta_norm": 0.18,
    "fallback_recovery_success_rate": 0.98
  },
  "regression": {
    "compared_to": "follower_stable_v000",
    "worse_metrics": []
  },
  "known_failures": [
    {
      "failure_type": "high_yaw_rate_instability",
      "condition": "target_yaw_rate_rad_s > 1.2",
      "severity": 0.4
    }
  ]
}
```

### 12.3 发布门槛

候选模型进入 release candidate 至少需要：

- ABI schema validation 通过。
- replay validation 通过。
- 关键回归指标无显著退化。
- `fall_rate`、`near_fall_rate`、任务误差和动作平滑指标满足当前 Phase 门槛。
- 已生成 fallback model 绑定。
- 已生成 capability summary。

不通过的模型进入 `rejected`，不能被本地自动拉取。

## 13. Model Registry 与 Release State

模型状态：

```text
draft
candidate
staged
stable
rejected
rolled_back
archived
```

状态含义：

- `draft`：训练产物已生成，但未完成评测。
- `candidate`：通过基础评测，可供受控客户端下载。
- `staged`：允许小范围灰度。
- `stable`：当前推荐稳定版本。
- `rejected`：评测失败或风险过高。
- `rolled_back`：曾发布但因本地反馈或回归被回滚。
- `archived`：保留追溯，不再推荐使用。

状态转换规则：

- `draft -> candidate`：必须有 evaluation report。
- `candidate -> staged`：必须有 release manager 批准和 fallback 绑定。
- `staged -> stable`：必须有足够本地 gate 反馈或线上评测稳定记录。
- `staged/stable -> rolled_back`：本地回滚率、危险信号或回归超过门槛。
- `rejected` 不能直接转 `stable`，必须重新训练或重新评测生成新版本。

## 14. Release Package

本地拉取的不是裸模型，而是 release package。

```json
{
  "schema_version": "release_package.v0",
  "release_id": "release_follower_candidate_v001",
  "model_id": "follower_candidate_v001",
  "release_state": "candidate",
  "phase": "phase_1",
  "abi_version": "abi.northstar.v0",
  "artifact": {
    "uri": "school://models/follower_candidate_v001/model.pt",
    "sha256": "sha256:..."
  },
  "fallback": {
    "model_id": "follower_stable_v000",
    "required": true
  },
  "gate_recommendation": {
    "initial_takeover_ratio": 0.0,
    "max_takeover_ratio": 0.25,
    "allowed_task_families": ["locomotion"],
    "blocked_conditions": [
      "fall_risk > 0.5",
      "confidence < 0.4"
    ]
  },
  "evaluation_report_id": "eval_follower_candidate_v001",
  "capability_summary_id": "cap_follower_candidate_v001"
}
```

规则：

- 本地系统必须校验 artifact hash 和 ABI version。
- candidate 初始接管比例应为 0，通过影子推理或低风险试运行逐步增加。
- release package 必须包含 fallback 依赖。
- blocked conditions 必须能被本地 gate 解释。

## 15. 本地 Gate 反馈回流

本地使用候选模型后，需要回传 gate feedback。

```json
{
  "schema_version": "gate_feedback.v0",
  "feedback_id": "gate_feedback_000001",
  "client_id": "sim_client_001",
  "stable_model_id": "follower_stable_v000",
  "candidate_model_id": "follower_candidate_v001",
  "phase": "phase_1",
  "summary": {
    "shadow_steps": 10000,
    "active_steps": 2000,
    "fallback_count": 12,
    "rollback_triggered": false,
    "mean_takeover_ratio": 0.18
  },
  "metrics": {
    "candidate_better_count": 430,
    "candidate_worse_count": 38,
    "dangerous_sig_peak": 0.42,
    "confidence_min": 0.51
  },
  "events": [
    {
      "event_type": "fallback_entered",
      "count": 12
    }
  ],
  "sample_refs": [
    "sample_000001"
  ]
}
```

反馈用途：

- 决定 candidate 是否进入 staged 或 stable。
- 发现学校评测未覆盖的失败模式。
- 生成新的 high-priority samples。
- 校准 gate recommendation。

## 16. 能力摘要与云端大脑弱通信

学校向云端大脑暴露的是能力摘要，不是训练细节。

### 16.1 Capability Summary

```json
{
  "schema_version": "capability_summary.v0",
  "summary_id": "cap_follower_candidate_v001",
  "model_id": "follower_candidate_v001",
  "phase": "phase_1",
  "embodiment_id": "unitree_g1_43dof_sim_v0",
  "capabilities": [
    {
      "name": "velocity_tracking_flat",
      "status": "supported",
      "conditions": {
        "target_velocity_m_s_abs_max": 1.0,
        "target_yaw_rate_rad_s_abs_max": 1.0,
        "terrain": "flat"
      },
      "confidence": 0.85
    }
  ],
  "unsupported_or_risky": [
    {
      "name": "high_yaw_turn",
      "condition": "target_yaw_rate_rad_s > 1.2",
      "risk": "near_fall"
    }
  ],
  "known_failure_modes": [
    {
      "failure_type": "high_yaw_rate_instability",
      "severity": 0.4,
      "mitigation": "reduce yaw command or request brace"
    }
  ],
  "recommended_planning_constraints": {
    "max_velocity_m_s": 1.0,
    "max_yaw_rate_rad_s": 1.0,
    "requires_flat_terrain": true
  }
}
```

云端大脑使用该摘要调整任务规划，不直接干预学校训练。

### 16.2 本地大脑可用摘要

同一 capability summary 也可以被本地大脑读取，用于避免生成超出 follower 能力边界的语义意图。

## 17. Phase 集成

### 17.1 Phase 1

学校最小闭环：

- 收集 locomotion、跌倒、近跌倒、抗扰、brace、stop、fallback 片段。
- 训练或评测基础 follower。
- 发布候选 follower。
- 本地双模型影子推理和低比例 gate。
- 回传 fallback 和 near-failure 样本。

### 17.2 Phase 2

学校扩展：

- 收集上下肢冲突、reach 失败、姿态保持失败、接触前悬停失败。
- 构建全身协调数据集。
- 比较不同上肢/下肢耦合策略。

### 17.3 Phase 2.5

学校承担对照实验：

- unified policy。
- shared trunk + multi-head。
- MoE。
- 拼接式 baseline。
- Mimic prior。
- Mimic teacher-student。

学校输出：

- 对照报告。
- 统一 follower 是否成立的证据。
- 进入 Phase 3 小脑训练的 teacher 或 expert 候选。
- Mimic source manifest、retargeting gap、teacher/student disagreement 和 runtime_allowed 判定。

### 17.4 Phase 3

学校训练小脑相关模型：

- generator。
- selector/gate。
- confidence calibration。
- dangerous signal predictor。
- model disagreement scorer。

### 17.5 Phase 4

学校收集语义意图到执行失败的边界：

- 本地大脑输出意图。
- 小脑光轴生成结果。
- follower 执行结果。
- 能力边界反馈。

本地大脑仍不作为学校主替换目标。学校主要训练小脑 + follower。

## 18. API 边界

Phase 1 最小 API 可以是文件级接口，不必一开始做网络服务。

### 18.1 本地到学校

最小接口：

```text
submit_sample(envelope, artifact)
submit_gate_feedback(feedback)
submit_evaluation_branch_result(report)
```

### 18.2 学校内部

最小接口：

```text
validate_sample(sample_id)
build_dataset(query)
start_training_job(job_manifest)
register_model(model_manifest)
run_evaluation(model_id, evaluation_plan)
create_release_package(model_id)
```

### 18.3 学校到本地

最小接口：

```text
list_candidate_releases(client_capabilities)
download_release_package(release_id)
report_release_outcome(gate_feedback)
```

### 18.4 学校到云端大脑

最小接口：

```text
get_capability_summary(model_id)
list_current_stable_capabilities(embodiment_id)
list_known_failure_modes(embodiment_id, phase)
```

## 19. 关键指标

### 19.1 数据质量指标

- schema validation pass rate。
- artifact hash validation pass rate。
- duplicate sample rate。
- usable_for_training fraction。
- usable_for_release_gate fraction。
- average priority score。

### 19.2 学习效果指标

- 候选模型相对稳定模型的指标提升。
- 回归指标数量。
- release gate 通过率。
- high-priority sample 使用率。
- near-failure 复现和修复率。

### 19.3 发布安全指标

- fallback count。
- rollback rate。
- hard switch count。
- dangerous_sig peak under candidate。
- model disagreement distribution。
- candidate takeover ratio progression。

### 19.4 云端规划可用指标

- capability summary 覆盖率。
- known failure mode 新鲜度。
- unsupported condition 命中率。
- 云端规划违反能力边界的比例。

## 20. 风险与缓解

### 20.1 上传样本过多，学校变成数据垃圾场

缓解：必须有本地 priority scoring、去重、保留策略和 release gate 样本标签。

### 20.2 只收失败样本导致策略过度保守

缓解：dataset builder 必须保留 `clean_success_reference`，并设置最小比例。

### 20.3 候选模型平均指标提升但长尾退化

缓解：release gate 必须包含高风险、near-failure、边界 command 和前序 Phase 回归集。

### 20.4 学校训练和云端规划耦合过强

缓解：学校只暴露能力摘要和失败模式，不暴露 replay buffer 给云端在线决策。

### 20.5 本地 adapter 与学校主模型替换冲突

缓解：gate feedback 必须记录 adapter id；发布包必须声明 adapter 保留、迁移、重置或重新校准策略。具体策略在 gate/fallback 子规格中展开。

### 20.6 线下分支失败被忽略

缓解：offline_high_fidelity_case 必须成为 segment type，线下失败样本应默认获得较高 priority，并进入专门回归集。

## 21. 默认技术取舍

为保持推进速度，默认取舍如下：

1. Phase 1 学校系统先用文件级 API 和本地目录模拟 school object store。
2. Metadata index 可以先用 SQLite 或 Parquet metadata 表；后续再迁移到服务化数据库。
3. Artifact store 可以先用本地路径或对象存储兼容路径。
4. Dataset Builder 先实现 query + manifest + symlink/copy 形式，不急于做分布式数据服务。
5. Release Manager 先实现 manifest 状态机，不急于做完整部署平台。
6. Capability Summary 先以 JSON 文件输出，供云端大脑或本地大脑后续读取。
7. Mimic-derived samples 先以 source manifest + episode artifact 形式接入，不急于把外部 Mimic 框架嵌入学校 runtime。

这些取舍让 Phase 1 可以尽快获得学校最小闭环，同时不阻塞后续服务化。

## 22. 与后续子规格的关系

本规格提供学校系统的数据和发布骨架。后续文档应继续细化：

- Phase 1 基础运动底座与学校最小闭环：定义首个训练任务、reward、指标和学校最小落地。
- Gate、Fallback 与模型版本切换：定义本地 gate blending、adapter 迁移和回滚策略。
- 小脑光轴学习与消融：定义 generator/selector 训练数据如何从学校经验池构建。
- 线上/线下验证树与指标：定义学校如何消费线下高保真分支结果。
- Mimic Motion Prior 集成与应用测试：定义外部 motion prior、teacher、expert、oracle 和受限数据如何进入学校样本、数据集版本和 release gate。
