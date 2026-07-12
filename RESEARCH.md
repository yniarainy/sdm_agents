# A Scientific Agent Framework for Verifiable, Adaptive, and Knowledge-Grounded Species Distribution Modelling

> **Target**: Methods in Ecology and Evolution (MEE)
> **Positioning**: 不是"AI 工具"，而是一种**新的生态学研究方法**

---

## 核心洞察

### MEE 编辑问的唯一问题

> *"你的 Method，解决了生态学研究者以前解决不了的问题吗？"*

不是"AI 厉害了"，而是 **Ecology 变好了**。

### 两种叙事，两种命运

| 叙事 A: AI 工具 (✗) | 叙事 B: 科学方法 (✓) |
|---------------------|---------------------|
| 用 AI 自动化了 SDM | 提出了一种**可验证、自适应、知识驱动的**生态位建模新范式 |
| Agent 替代了手动操作 | Agent **改变了**研究者与数据的交互方式 |
| 更快、更省力 | 能回答**以前无法回答**的问题 |
| 输出：AUC=0.85 | 输出：**为什么** AUC=0.85，**哪里**还不确定，**下一步**该去哪采样 |

---

## 范式转变：Scientific Agent Framework

### 传统 SDM 工作流

```
Data ──→ Researcher (all decisions) ──→ Model ──→ Paper
              ↑
         黑箱：为什么选这个算法？为什么删这些点？
         为什么设这个阈值？预测可信吗？
```

### Scientific Agent Framework 工作流

```
Data ──→ Scientific Agent ──→ Researcher (审核与决策) ──→ Paper
              │                       ↑
              │  ① 数据质量诊断       │
              │  ② 假设生成           │  审查决策理由
              │  ③ 方法选择           │  验证反事实分析
              │  ④ 模型训练           │  确认采样建议
              │  ⑤ 不确定性分析       │
              │  ⑥ 反事实验证         │
              │  ⑦ 采样建议           │
              │  ⑧ 科学解释           │
              │                       │
              └─── Decision Graph ────┘
              (每一步: Decision · Evidence · Alternative · Confidence · Counterfactual)
```

**这不是"一个 SDM 工具"，这是生态学研究流程的范式升级。**

---

## 三大方法学创新

### Innovation 1: Scientific Decision Graph (SDG)

> **解决什么问题**: AI 黑箱 — 传统 SDM 每个决策都不可追溯，研究者无法判断模型是否可信
>
> **生态学意义**: 首次让 SDM 建模的每一个关键决策都附带**可审计的科学证据链**

#### SDG 的五要素

每个建模决策节点输出结构化决策记录：

```
Decision Node: Model Selection
├── Decision:   选择 XGBoost 作为最终模型
├── Evidence:   CV AUC: XGB=0.87, RF=0.84, LGBM=0.82, LogReg=0.76
│               XGB 在 Bathymetry-SST 非线性交互上表现最优
├── Alternative: 如果选 RF，AUC 降 0.03，TSS 从 0.62 降至 0.56
│               差异主要分布在近岸浅水区 (反事实空间图)
├── Confidence:  0.85 (基于 CV 方差和算法间一致性)
└── Counterfactual: 见 alternatives/counterfactual_rf_vs_xgb.png
```

#### 完整的 SDG 节点链

```
Occurrence Cleaning
├── 为什么删除了 23 个点？
├── 证据: 23 个点在陆地上/超出研究区/年份不匹配
└── 如果保留: AUC 会降 0.02 (噪声引入)

Feature Selection
├── 为什么保留 SST, Chl-a, Bathymetry, Salinity？
├── 证据: Permutation 重要性排序; 多重共线性 VIF < 5
└── 如果去掉 Bathymetry: AUC 降 0.06 (反事实)

Spatial CV Strategy
├── 为什么选 spatial_block_kfold 而非 random_kfold？
├── 证据: 空间自相关 Moran's I = 0.34; random CV 会高估 AUC 0.05
└── 如果选 random: 预测图在北部海域过度乐观

Pseudo-absence Generation
├── 为什么设 1:1 比例？为什么排除水深 < 10m 区域？
├── 证据: 文献建议底栖鱼类 PA 应排除不可栖息深度
└── 如果全随机: 会在陆地/浅滩生成无效 PA

Threshold Selection
├── 为什么 threshold = 0.53 而非默认 0.5？
├── 证据: 最大化 TSS; Sensitivity-Specificity 交点
└── 如果 0.5: Sensitivity 降 0.08

Prediction Uncertainty
├── 预测可信度分布: 高置信区占 62%, 低置信区占 15%
├── 不确定性来源: 60% 数据不足, 40% 模型分歧
└── 建议: 3 个低置信区需要补充采样
```

#### 实验设计

| 实验 | 方法 | 指标 |
|------|------|------|
| **专家盲审** | 5 位生态学家盲审 20 个决策节点，评分 1-10 | 科学合理性得分 |
| **信任度测试** | 有无 SDG 时，研究者对同一模型结果的采纳意愿 | 采纳率差异 |
| **反事实准确率** | 100 个物种，对比反事实预测 vs 实际重跑结果 | 误差 < 5% |
| **可复现性** | 同一配置跑 10 次 vs 手动 SDM 跑 10 次 | 结果方差对比 |

---

### Innovation 2: Adaptive Ecological Sampling

> **解决什么问题**: 传统 SDM 被动接受已有数据，无法指导**下一步该去哪采样**
>
> **生态学意义**: 首次将 SDM 从"描述已有分布"升级为"指导未来采样"，解决生态学最核心的 Sampling 问题

#### 这不是 AI，这是 Adaptive Ecological Sampling

生态学最稀缺的资源是什么？**野外采样数据。**

当前所有 SDM 都在回答"物种分布在哪"，但没有一个在回答**"下一趟船该往哪开"**。

#### 工作流

```
1. 用现有数据训练模型
        ↓
2. 计算三维覆盖缺口:
   ├── 地理空间: 哪些区域没有采样点？
   ├── 环境空间: 哪些环境组合没有观测？
   └── 预测不确定性: 哪些区域模型最不确定？
        ↓
3. Agent 输出采样建议:
   "建议在 (119.5°E, 34.2°N) 附近采集 5 个站位。
    理由: 该区域 SST > 26°C 且 Chl-a < 0.3 mg/m³，
    这一环境组合在当前数据中完全缺失。
    补充后可预期 AUC 提升 0.05-0.08，
    并降低近岸区域预测不确定性 30%。"
        ↓
4. 用户补充数据后重训练
        ↓
5. 对比前后: 预测图变化、不确定性缩减、AUC 提升
```

#### 为什么这是一个 Method？

| 传统 SDM | Adaptive Ecological Sampling |
|----------|---------------------------|
| 数据 → 预测 | 数据 → 预测 → **发现缺口** → **建议采样** → 新数据 → 更好的预测 |
| 被动 | **主动** |
| 一次性 | **迭代闭环** |
| 回答 "在哪" | 回答 "在哪" + **"下一步去哪"** |

#### 实验设计

| 实验 | 方法 | 预期结论 |
|------|------|---------|
| **数据削减模拟** | 从完整数据集随机删除 70% 点，对比: 主动学习补点 vs 随机补点 vs 均匀网格补点 | 主动学习用 50% 的点达到全量 95% AUC |
| **采样效率** | 达到 AUC=0.85 所需的采样点数 | 主动学习减少 40-60% |
| **真实案例** | 与合作课题组对接，实际指导一轮野外采样 | 野外验证预测准确性 |

---

### Innovation 3: Ecological Knowledge Integration

> **解决什么问题**: 当前 SDM 只用结构化环境因子，完全忽略了论文、遥感影像、季节动态中蕴含的生态学知识
>
> **生态学意义**: 首次将多源生态学知识（文献 + 遥感 + 时序）系统性地融入物种分布建模

#### 三个知识来源，一个生态学故事

```
Knowledge Source 1: Literature-Derived Ecological Priors
├── 不是 "NLP" 或 "LLM"
├── 而是: 从已发表的生态学文献中提取物种的环境偏好
│   作为贝叶斯先验，约束模型在生态学合理的范围内预测
├── 例: "根据 12 篇文献，黄鳍金枪鱼偏好 SST 18-28°C，
│   产卵期要求 SST > 24°C。将此作为先验约束纳入模型。"
└── 生态学意义: 模型预测不再纯数据驱动，而是受生态学理论约束

Knowledge Source 2: Habitat Structure from Remote Sensing
├── 不是 "Vision Transformer" 或 "计算机视觉"
├── 而是: 从卫星影像中提取对物种有生态学意义的生境结构特征
├── 例: "从 Landsat-8 提取的珊瑚礁覆盖率作为底栖鱼类的
│   生境复杂度指标，使近岸区域的预测 AUC 提升 0.06"
└── 生态学意义: 将遥感生态学参数纳入 SDM，而非仅用气候变量

Knowledge Source 3: Temporal Ecological Niche Dynamics
├── 不是 "Transformer" 或 "时序深度学习"
├── 而是: 捕捉物种对环境因子的季节性响应差异
├── 例: "该物种对 8 月 SST 的敏感性是 1 月的 3.2 倍，
│   与其夏秋季产卵的生态习性一致"
└── 生态学意义: 揭示生态位的时序动态，而非仅建模年均状态
```

#### 实验设计

| 知识来源 | 消融实验 | 预期 |
|---------|---------|------|
| 文献先验 | 有/无先验约束的 AUC 对比 | AUC 提升 3-8%，环境响应曲线更符合生态学预期 |
| 生境结构 | 有/无遥感生境特征的 AUC 对比 | 近岸物种增益最大 |
| 时序动态 | 月均序列 vs 年均值的 AUC 对比 | 季节性敏感物种提升显著 |

---

## 为什么 MEE 会喜欢这个框架？

### MEE 的发表标准

| MEE 关心什么 | 我们怎么回答 |
|-------------|------------|
| **方法解决了以前解决不了的问题吗？** | 以前 SDM 不可验证、无法指导采样、知识融合困难 → 现在三个创新逐一解决 |
| **方法改变了研究者的工作方式吗？** | 从"手动黑箱操作"到"Agent 生成 → 研究者审核 → 迭代优化" |
| **方法有生态学意义吗？** | SDG 让建模可审计，Adaptive Sampling 解决野外采样效率，Knowledge Integration 让模型受生态学理论约束 |
| **方法可复现吗？** | SDG 本身就是可复现性工具 — 每个决策都有记录 |

### 论文故事线

```
1. Introduction
   "SDM 建模面临三个根本性挑战:
    ① 决策不透明 — 研究者无法审计建模过程
    ② 数据被动 — 模型无法指导下一步采样
    ③ 知识割裂 — 文献/遥感/时序知识无法融入建模
    我们提出 Scientific Agent Framework 来解决这三个挑战。"

2. The Scientific Agent Framework
   2.1 范式转变: 从黑箱操作到可审计的科学决策
   2.2 Innovation 1 — Scientific Decision Graph
   2.3 Innovation 2 — Adaptive Ecological Sampling
   2.4 Innovation 3 — Ecological Knowledge Integration

3. Experiments
   3.1 SDG 验证: 专家盲审 + 反事实准确率 + 可复现性
   3.2 采样效率: 主动学习 vs 随机 vs 均匀网格
   3.3 知识融合消融: 文献 ± 遥感 ± 时序

4. Results
   4.1 SDG 使建模过程从"不可审计"变为"可追溯"
   4.2 Adaptive Sampling 用 50% 点数达到全量 95% AUC
   4.3 生态知识融合使模型更符合生态学预期

5. Discussion
   "Scientific Agent Framework 不仅是一个工具，
    而是一种新的生态学研究方法 —
    它改变了研究者与数据、模型、决策的关系。"
```

---

## 与现有代码的映射

| Innovation | 核心新文件 | 修改现有文件 | 当前基础 |
|-----------|-----------|-------------|---------|
| **SDG** | `rationale.py`, `counterfactual.py` | `agent_graph.py` 每个节点 | `step_status`, `error_events`, `_auto_repair` |
| **Adaptive Sampling** | `active_learning.py`, `sampling_recommender.py` | 新增 `sampling_advisor` 节点 | `committee_agreement.png`, UQ 雏形 |
| **Knowledge Integration** | `knowledge_extractor/` (文献), `gee_tools.py` 扩展 (生境) | `sdm_trainer/` (时序) | `datasets.json`, GEE 基础设施 |

---

## 实施路线

### Phase 1: SDG 原型 (2 周) → 可开始写 Methods
1. 实现 `RationaleNode` 数据类
2. 为 `training`, `split`, `ensemble`, `evaluate` 节点添加决策记录
3. HTML 报告增加"决策日志"板块
4. 产出: 一份带完整决策链的 demo 报告

### Phase 2: Adaptive Sampling 原型 (2 周) → 核心实验可跑
1. 环境空间覆盖分析
2. 不确定性热力图
3. 采样建议生成
4. 数据削减模拟实验脚本

### Phase 3: Knowledge Integration (3 周) → 消融实验
1. 文献知识提取 pipeline
2. 遥感生境特征提取
3. 时序生态位分析
4. 消融实验: 有/无各类知识

### Phase 4: 全文撰写 (4 周)
1. 基准实验: 10 物种 × 3 区域
2. 专家盲审: SDG 决策质量评估
3. 采样效率对比实验
4. 论文撰写 + 修改

---

## 最终定位

```
以前: "我们用 AI 自动化了 SDM"
现在: "我们提出了一种新的生态学建模方法 —
       Scientific Agent Framework —
       它让 SDM 从不可验证变为可审计，
       从被动描述变为主动指导，
       从数据驱动变为知识驱动。"

这不是一个软件。
这是一种新的 Scientific Workflow。
```
