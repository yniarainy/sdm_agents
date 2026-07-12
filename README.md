# 🌊 Scientific Agent Framework for SDM

> **A Verifiable, Adaptive, and Knowledge-Grounded Approach to Species Distribution Modelling**
>
> 不是"AI 自动化工具" — 而是一种**新的生态学研究方法**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![biomod2](https://img.shields.io/badge/biomod2-R_4.x-brightgreen.svg)](https://biomodhub.github.io/biomod2/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📌 项目状态

```
✅ 9节点 LangGraph 科学 Agent 图   ✅ 4+6 种算法 + Ensemble
✅ 模型可解释性 (SHAP/PDP/响应曲线) ✅ biomod2 R 引擎
✅ 现代 Web 控制台                  ✅ 研究路线图 (3 大方法学创新)
🔲 GEE 真实数据 (需 VPN)            🔲 基准实验 + 专家盲审 (论文阶段)
```

**当前生成产物 (16 个文件):**
```
roc_curve.png  ·  confusion_matrix.png  ·  variable_importance.png
shap_summary.png  ·  partial_dependence.png  ·  response_curves.png
prediction_map.png  ·  ensemble_prediction_map.png  ·  committee_agreement.png
evaluation_report.html  ·  run_summary.json  ·  best_model.joblib
```

---

## 🎯 这为什么是一个 Method（而不只是工具）

传统 SDM 的三个根本性挑战：

| 挑战 | 传统方法 | Scientific Agent Framework |
|------|---------|---------------------------|
| **决策不透明** | 算法选择、参数设置、数据清洗 — 全是黑箱 | **Scientific Decision Graph** — 每个决策附带 Evidence + Alternative + Confidence + Counterfactual |
| **数据被动** | 模型只能描述已有数据，无法指导下一步 | **Adaptive Ecological Sampling** — Agent 发现知识缺口，主动建议采样位置并解释原因 |
| **知识割裂** | 只用结构化环境因子，文献/遥感/时序知识被浪费 | **Ecological Knowledge Integration** — 论文先验 + 生境结构 + 生态位时序动态系统性融入 |

> 传统 SDM 回答 "物种分布在哪"。Scientific Agent Framework 进一步回答：
> **"为什么是这个分布？模型可信吗？下一步该去哪采样？"**

---

## 🏗️ 架构

```
┌──────────────────────────────────────────────────────┐
│            🖥️ Web Dashboard (FastAPI + LangServe)     │
│       对话式交互 · 拖拽上传 · SSE 实时日志            │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│       🧠 SDMAgentGraph — 9 节点 LangGraph 流水线      │
│                                                       │
│  planning ─→ data_acq ─→ split ─→ train ─→ biomod2   │
│                                  │                    │
│       ┌──────────────────────────┘                    │
│       ▼                                              │
│  ensemble ─→ evaluate ─→ predict ─→ report           │
│                                                       │
│  条件路由 · 错误自动检测 · 状态双向同步 · LLM可用切换   │
└──────────────────────┬───────────────────────────────┘
                       │
    ┌──────┬───────┬───┴───┬───────┬──────┐
    ▼      ▼       ▼       ▼       ▼      ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│ GBIF ││ OBIS ││ GEE  ││ biomod2││SHAP ││Ensemble│
│ 下载  ││ 下载  ││ 提取  ││ R引擎 ││解释 ││委员会 │
└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
```

**双引擎设计**: Python (scikit-learn/XGBoost/LightGBM) + R (biomod2)，自动选最优。

---

## 🔄 三种数据模式

| 模式 | 数据来源 | 适用场景 |
|------|---------|---------|
| `upload` | 用户提供完整 CSV（lon/lat/is_presence + 环境因子） | 已有提取好的数据，直接建模 |
| `gbif_obis` | 自动从 GBIF+OBIS 下载存在点 → GEE 提取环境因子 | 从零开始，全自动 |
| `gee_extract` | 用户提供存在点 CSV → 从 GEE 提取环境因子 | 有自己的存在点数据 |

```
upload:      [load_dataset] ──→ split ──→ train ──→ biomod2 ──→ ensemble ──→ evaluate ──→ predict ──→ report
gbif_obis:   [GBIF+OBIS下载] ──→ [GEE提取] ──→ split ──→ ... (同上)
gee_extract: [加载存在点CSV] ──→ [GEE提取(严格)] ──→ split ──→ ... (同上)
```

---

## 📊 完整流水线

| 节点 | 功能 | 亮点 |
|:----:|------|------|
| **planning** | 加载配置、构建计划、LLM路由决策 | 智能因子推荐（根据物种生态类型） |
| **data_acquisition** | 获取存在点 + 伪缺失生成 + 环境因子提取 | GBIF+OBIS 双源去重合并，8个 GEE 遥感数据集 |
| **split_data** | 5 种空间切分策略 + CV 折分配 | random / spatial_kfold / spatial_block / env_spatial_block |
| **training** | 4 算法对比训练，CV 选最优 | RF · XGBoost · LightGBM · LogisticRegression |
| **biomod2** | R 语言 biomod2 框架集成 | 6 算法 + TSS 筛选 + 3 种集成方式 |
| **ensemble** | AUC 加权多模型集成 | 委员会一致性评估 (committee agreement) |
| **evaluation** | 全量指标 + 可解释性分析 | SHAP · PDP · Permutation · 响应曲线 |
| **prediction** | 单模型 + Ensemble 双预测图 | 含委员会不确定性图 |
| **report** | HTML 报告 + JSON 元数据 | 数据质量标识 (真实/合成/用户) |

### 可解释性输出 (4 张图)

| 图表 | 说明 |
|------|------|
| **Permutation 重要性** | 每个环境因子对 ROC AUC 的边际贡献（带误差棒） |
| **SHAP 特征贡献** | TreeExplainer 全局特征影响排序（rf/xgb/lgbm） |
| **偏依赖图 (PDP)** | Top 3 因子的边际效应曲线 |
| **响应曲线** | 全部因子 vs 预测适宜度（其他因子固定中位数） |

### Ensemble 输出 (3 个文件)

| 产物 | 说明 |
|------|------|
| `ensemble_prediction_map.png` | 多模型 AUC 加权投票的集成预测图 |
| `committee_agreement.png` | 1 - 各模型预测 std，显示模型间一致性 |
| `ensemble_prediction.csv` | 集成预测网格数值 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/yniarainy/sdm_agents.git
cd sdm_agents
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
# 配置 DeepSeek API Key: 创建 .env，写入 DEEPSEEK_API_KEY="sk-xxx"
```

### 运行

```bash
# 方式1: 命令行（使用 config.yaml）
python main.py --auto

# 方式2: Web 控制台
start_sdm.bat   # Windows
# 或 python -m uvicorn apps.langserve_chat:app --host 0.0.0.0 --port 8000
# 然后访问 http://127.0.0.1:8000/console

# 方式3: Python API (单体)
from agents.orchestrator import SDMOrchestrator
state = SDMOrchestrator(config_path="config.yaml", interactive=False).run()

# 方式4: Python API (多Agent图 — 推荐)
from agents.orchestrator.agent_graph import SDMAgentGraph
result = SDMAgentGraph(config_path="config.yaml", enable_llm=False).run()
print(result["step_status"])      # 各步骤状态
print(result["metrics"]["ensemble"])  # 集成权重
print(result["artifacts"])        # 输出文件路径
```

---

## ⚙️ 核心配置

```yaml
species_name: "demersal_fish"
data_mode: "gbif_obis"          # upload | gbif_obis | gee_extract

# 环境因子（8个 GEE 海洋遥感数据集可用）
factors: [sst, chl_a, salinity, bathymetry]

# 算法（自动对比选最优）
algorithms: [rf, xgb, lgbm, logreg]

# 空间交叉验证（5种策略）
split_mode: "spatial_block_kfold"
n_splits: 5

# 时空范围
bbox: [110.0, 20.0, 125.0, 35.0]    # 南海
start_date: "2023-01-01"
end_date: "2023-12-31"

# GEE 设置（用户需自行认证）
use_gee: true
strict_gee: false                     # true = 无GEE即中止
```

### GEE 认证

```bash
# 方式1: 命令行认证（推荐）
python -c "import ee; ee.Authenticate(); ee.Initialize()"

# 方式2: 服务账号
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

若未认证且使用 `data_mode: gbif_obis`，系统会回退合成特征并在报告中标注 ⚠️。

---

## 📁 项目结构

```
sdm_agents/
├── main.py                            # CLI 入口
├── config.yaml                        # 默认配置
├── requirements.txt                   # Python 依赖
├── start_sdm.bat                      # Windows 一键启动
├── README.md                          # 项目文档
├── TODO.md                            # 待改进清单
├── RESEARCH.md                        # 研究创新路线图
│
├── agents/
│   ├── orchestrator/                  # 🔥 核心引擎
│   │   ├── agent.py                   #   SDMOrchestrator: 单体流水线 + 自动纠错
│   │   ├── agent_graph.py             #   🆕 SDMAgentGraph: 9节点 LangGraph 多Agent图
│   │   ├── state.py                   #   PlanConfig / PipelineState / AgentState
│   │   └── occurrence_tools.py        #   GBIF/OBIS 数据下载+清洗+去重合并
│   │
│   ├── gee_data_fetcher/              # 🌐 GEE 遥感数据提取
│   │   ├── agent.py                   #   LangGraph Agent (DeepSeek 驱动)
│   │   └── tools/
│   │       ├── gee_tools.py           #   GEE 搜索+批量提取
│   │       └── references/datasets.json # 8个海洋遥感数据集字典
│   │
│   ├── bg_generator/                  # 📊 背景点生成
│   │   └── tools/spatial_tools.py     #   伪缺失点空间随机生成
│   │
│   ├── sdm_trainer/                   # 🎯 模型训练
│   │   ├── tools/r_bridge.py          #   Python→R 桥接
│   │   └── scripts/run_biomod2.R      #   🆕 biomod2 完整训练脚本
│   │
│   ├── sdm_evaluator/                 # 🔬 模型评估
│   └── sdm_projector/                 # 🗺️ 空间投影
│       ├── tools/prediction_tools.py
│       └── scripts/project_map.R
│
├── apps/                              # Web 应用
│   ├── langserve_chat.py              #   FastAPI + LangServe + SSE 流式
│   ├── static/dashboard.html          #   🆕 现代 Web 控制台
│   └── uploads/                       #   文件上传存储
│
└── workspace/                         # 运行产物
    └── <species>_<timestamp>/
        ├── points_with_labels.csv     # 带标签点数据
        ├── training_dataset.csv       # 训练数据集
        ├── best_model.joblib          # 最优模型
        ├── roc_curve.png              # ROC 曲线
        ├── confusion_matrix.png       # 混淆矩阵
        ├── variable_importance.png    # 🆕 变量重要性
        ├── shap_summary.png           # 🆕 SHAP 贡献
        ├── partial_dependence.png     # 🆕 偏依赖图
        ├── response_curves.png        # 🆕 响应曲线
        ├── prediction_map.png         # 空间预测图
        ├── ensemble_prediction_map.png # 🆕 集成预测图
        ├── committee_agreement.png    # 🆕 委员会一致性
        ├── evaluation_report.html     # HTML 评估报告
        └── run_summary.json           # 运行元数据
```

---

## 🔬 模型可解释性

每次运行自动生成 4 类可解释性分析：

| 分析 | 问题 | 方法 |
|------|------|------|
| **Permutation 重要性** | 哪个环境因子对预测最重要？ | 随机打乱每个因子，测量 AUC 下降 |
| **SHAP 特征贡献** | 每个因子如何影响单个预测？ | TreeExplainer (rf/xgb/lgbm) |
| **偏依赖图** | 因子的边际效应是什么样的？ | PDP — Top 3 因子的平均预测曲线 |
| **响应曲线** | 每个因子的生态响应形状？ | 全部因子逐一变化，其他固定中位数 |

---

## 🎓 发表路线图

> 详见 `RESEARCH.md` — 三大方法学创新的完整论证、实验设计与 MEE 投稿策略。

### 三大方法学创新

| # | 创新 | 解决的生态学问题 | MEE 适配度 |
|:--:|------|-----------------|:--------:|
| **I1** | **Scientific Decision Graph** | SDM 决策不可追溯 → 每个决策附带可审计证据链 | ⭐⭐⭐⭐⭐ |
| **I2** | **Adaptive Ecological Sampling** | 模型被动 → 主动发现知识缺口，建议下一站去哪采样 | ⭐⭐⭐⭐⭐ |
| **I3** | **Ecological Knowledge Integration** | 只用结构化数据 → 论文+遥感+时序知识系统性融入 | ⭐⭐⭐⭐ |

### 论文叙事

> "我们提出了 Scientific Agent Framework — 一种新的生态位建模方法。
> 它解决了 SDM 的三个根本挑战：决策不可验证、数据被动、知识割裂。
> 它改变了研究者与建模过程的关系 — 从黑箱操作到可审计的科学决策。"

**当前进度**: 基础引擎完成 ✅，**下一步 I1.1 SDG 原型** → 可开始写 Methods 章节。

---

## 📦 依赖

```
Python 3.10+ · scikit-learn · XGBoost · LightGBM · SHAP
LangChain · LangGraph · FastAPI · LangServe
Google Earth Engine API (可选)
R 4.x + biomod2 (可选)
DeepSeek API (可选, 用于 LLM 增强)
```

---

## 📄 许可证

MIT License

---

<p align="center">
  <b>🌊 A Scientific Agent Framework — Verifiable · Adaptive · Knowledge-Grounded</b>
</p>
