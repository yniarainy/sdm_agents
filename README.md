# 🌊 SDM Agents — 基于多智能体的物种分布建模系统

> **LLM-Driven Multi-Agent Pipeline for Species Distribution Modeling**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📖 项目简介

**SDM Agents** 是一个基于 LangChain/LangGraph 构建的**多智能体协作系统**，旨在自动化完成物种分布模型（Species Distribution Model, SDM）的全流程建模工作。它将传统需要人工逐步操作的复杂生态建模任务转化为自然语言驱动的、智能体自主决策与执行的流水线。

### 为什么需要这个项目？

传统 SDM 建模流程通常需要研究者手动完成以下步骤：

1. 从 GBIF/OBIS 等数据库下载物种出现记录
2. 清洗、去重、时空过滤数据
3. 生成伪缺失/背景点
4. 从遥感数据（如 GEE）提取环境变量
5. 选择合适的算法（MaxEnt, Random Forest, GAM, GBM...）
6. 空间交叉验证
7. 模型评估与对比
8. 空间预测与制图

整个过程**耗时长、易出错、需要多种编程技能**（R/Python/SQL/GIS）。SDM Agents 的目标是让研究者只需**用自然语言描述需求**，智能体团队即可自动协作完成上述全流程。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                   🖥️ Web UI (FastAPI + LangServe)            │
│             自然语言交互 / 文件上传 / 实时任务流               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              🧠 SDM Orchestrator (主调度智能体)               │
│         规划 → 调度 → 自动纠错 → 状态追踪 → 报告生成           │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐   ┌────────▼───────┐   ┌───────▼──────┐
│ 🌐 GEE Data  │   │ 🎯 SDM Trainer │   │ 🗺️ Projector │
│   Fetcher    │   │  (R/biomod2)   │   │  (R/terra)   │
│ 遥感数据提取  │   │  模型训练/评估  │   │  空间预测制图 │
└──────────────┘   └────────────────┘   └──────────────┘
        │                   │                   │
┌───────▼──────┐   ┌────────▼───────┐   ┌───────▼──────┐
│ 📊 BG Gen    │   │ 🔬 Evaluator   │   │              │
│ 背景点生成    │   │ 模型评估诊断    │   │              │
└──────────────┘   └────────────────┘   └──────────────┘
```

---

## 🔄 完整流水线（8 步）

| 步骤 | 名称 | 功能 | 状态 |
|:----:|------|------|:----:|
| 1 | **prepare_points** | 从 GBIF/OBIS 下载存在点或上传文件；生成伪缺失点 | ✅ |
| 2 | **precheck_factors** | GEE 环境变量可用性预检 | ✅ |
| 3 | **build_dataset** | GEE 遥感数据提取（自动回退合成特征） | ✅ |
| 4 | **split_data** | 5 种空间切分策略 | ✅ |
| 5 | **train_models** | RF / LogisticRegression 训练+CV 选最优 | ✅ |
| 6 | **evaluate** | ROC AUC, PR AUC, TSS, F1, 混淆矩阵 | ✅ |
| 7 | **predict_map** | 研究区栖息地适宜度空间预测图 | ✅ |
| 8 | **build_report** | HTML 评估报告 + JSON 元数据汇总 | ✅ |

---

## ✨ 核心特性

### 🤖 智能体自主决策
- **自动纠错**：10+ 种常见配置错误自动检测与修复（非法参数、因子不匹配、切分策略冲突等）
- **智能因子推荐**：根据物种生态类型自动推荐环境变量（上层鱼类 → SST/Chl-a/洋流；底栖鱼类 → SST/盐度/水深）
- **自然语言交互**：通过 Web Chat 直接描述需求，LLM 自动解析意图和参数

### 🌐 真实数据集成
- **GBIF/OBIS 双源下载**：自动检索、去重、合并、时空过滤
- **Google Earth Engine**：从 8 个海洋遥感数据集中实时提取环境因子
- **自动回退机制**：GEE 不可用时自动使用合成特征，确保流水线不中断

### 📊 专业建模能力
- **5 种空间交叉验证策略**：
  - `random_holdout` — 随机留出法
  - `random_kfold` — 分层 K 折交叉验证
  - `spatial_kfold` — 基于 K-Means 聚类的空间 K 折
  - `spatial_block_kfold` — 空间分块 K 折
  - `env_spatial_block_kfold` — 环境分层+空间分块混合策略
- **多算法对比**：Random Forest / Logistic Regression
- **完整评估指标**：ROC AUC, PR AUC, TSS, Sensitivity, Specificity, F1

### 🖥️ Web 操作界面
- Chat 式自然语言交互（基于 LangServe）
- 存在点文件上传（CSV/TSV/GeoJSON）
- 任务后台运行 + SSE 实时日志推送
- 模板下载、参数查询

---

## 📦 安装

### 前置依赖

- Python 3.10+
- R 4.x（可选，用于 biomod2 集成）
- Google Earth Engine 账号（可选，用于真实遥感数据）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/sdm_agents.git
cd sdm_agents

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY

# 5. (可选) 初始化 GEE
python -c "import ee; ee.Authenticate(); ee.Initialize()"
```

### 环境变量

```bash
DEEPSEEK_API_KEY="sk-your-key-here"          # DeepSeek API 密钥
GOOGLE_APPLICATION_CREDENTIALS="/path/to/..."  # GEE 服务账号凭证（可选）
```

---

## 🚀 快速开始

### 方式 1：命令行（一键运行）

```bash
# 非交互模式（使用 config.yaml 默认参数）
python main.py --auto

# 交互模式（逐步配置参数）
python main.py
```

### 方式 2：Web 界面

```bash
# Windows
start_sdm.bat

# Linux/Mac
python -m uvicorn apps.langserve_chat:app --host 0.0.0.0 --port 8000 --reload
```

然后访问 `http://127.0.0.1:8000/console`，在对话框中输入：

> "帮我运行底栖鱼类的 SDM 建模，用 GBIF 和 OBIS 下载南海的存在点数据"

### 方式 3：Python API

```python
from agents.orchestrator import SDMOrchestrator

orchestrator = SDMOrchestrator(
    config_path="config.yaml",
    interactive=False,
)
state = orchestrator.run()
print(state.artifacts)
```

---

## ⚙️ 配置说明

编辑 `config.yaml`：

```yaml
species_name: "demersal_fish"      # 物种名称（用于 GBIF/OBIS 检索）
presence_source_mode: "gbif_obis"  # 存在点来源: upload/gbif/obis/gbif_obis
occurrence_download_limit: 1200    # 下载上限

# 时空范围
start_date: "2023-01-01"
end_date: "2023-12-31"
bbox: [110.0, 20.0, 125.0, 35.0]  # [min_lon, min_lat, max_lon, max_lat]

# 环境因子
factors:
  - sst           # 海表温度
  - chl_a         # 叶绿素-a
  - salinity      # 盐度
  - bathymetry    # 水深

# 算法
algorithms:
  - rf            # 随机森林
  - logreg        # 逻辑回归

# 建模参数
pseudo_absence_ratio: 1.0           # 伪缺失比例
test_size: 0.2                      # 测试集比例
split_mode: "random_holdout"        # 切分策略
n_splits: 5                         # 交叉验证折数
map_resolution: 140                 # 预测图分辨率

# GEE 设置
use_gee: true                       # 是否使用 GEE 真实数据
strict_gee: false                   # 严格模式（无 GEE 即失败）
enable_gee_precheck: true           # 变量可用性预检
```

---

## 📁 项目结构

```
sdm_agents/
├── main.py                          # CLI 入口
├── config.yaml                      # 默认配置文件
├── requirements.txt                 # Python 依赖
├── start_sdm.bat                    # Windows 一键启动脚本
│
├── agents/                          # 智能体模块
│   ├── orchestrator/                # 🔥 主调度器（核心）
│   │   ├── agent.py                 # SDMOrchestrator: 8步流水线
│   │   ├── state.py                 # PipelineState / PlanConfig 数据模型
│   │   └── occurrence_tools.py      # GBIF/OBIS 数据下载与清洗
│   │
│   ├── gee_data_fetcher/            # 🌐 遥感数据提取智能体
│   │   ├── agent.py                 # LangGraph Agent (DeepSeek)
│   │   ├── state.py                 # DataFetchState
│   │   └── tools/
│   │       ├── gee_tools.py         # GEE 搜索与提取工具
│   │       └── references/
│   │           └── datasets.json    # 海洋遥感数据集字典（8个数据集）
│   │
│   ├── bg_generator/                # 📊 背景点生成智能体
│   │   ├── agent.py                 # LangGraph Agent (DeepSeek)
│   │   ├── state.py                 # SDMState
│   │   └── tools/
│   │       └── spatial_tools.py     # 伪缺失点空间生成
│   │
│   ├── sdm_trainer/                 # 🎯 模型训练智能体
│   │   ├── agent.py                 # LangGraph Agent (DeepSeek)
│   │   ├── state.py                 # SDMState
│   │   ├── tools/
│   │   │   └── r_bridge.py          # R 语言桥接（biomod2）
│   │   └── scripts/
│   │       └── run_biomod2.R        # R 训练脚本
│   │
│   ├── sdm_evaluator/               # 🔬 模型评估智能体
│   │   └── tools/
│   │       └── r_eval_bridge.py     # R 评估桥接
│   │
│   └── sdm_projector/               # 🗺️ 空间投影智能体
│       ├── tools/
│       │   └── prediction_tools.py  # GEE 底图下载 + R 空间预测
│       └── scripts/
│           └── project_map.R        # R terra 包栅格预测
│
├── apps/                            # Web 应用
│   ├── langserve_chat.py            # FastAPI + LangServe Chat API
│   ├── static/
│   │   └── dashboard.html           # Web 控制台
│   └── uploads/                     # 上传文件存储
│
└── workspace/                       # 运行产物（每次运行一个子目录）
    └── <species>_<timestamp>/
        ├── points_with_labels.csv   # 带标签的点数据
        ├── training_dataset.csv     # 训练数据集
        ├── best_model.joblib        # 最优模型
        ├── roc_curve.png            # ROC 曲线
        ├── confusion_matrix.png     # 混淆矩阵
        ├── prediction_grid.csv      # 预测网格
        ├── prediction_map.png       # 空间预测图
        ├── evaluation_report.html   # HTML 评估报告
        ├── run_summary.json         # 运行元数据
        └── errors.json              # 错误日志
```

---

## 📊 运行产物展示

每次运行会生成完整的评估报告：

<!-- 以下为最近一次 demersal_fish 运行的实际指标 -->
| 指标 | 数值 |
|------|------|
| ROC AUC | 0.534 |
| PR AUC | 0.514 |
| TSS | 0.023 |
| F1 | 0.469 |
| 算法 | Random Forest (rf) |

> ⚠️ **注意**：上述指标基于合成特征（GEE 认证未通过时的数学模拟数据），不代表真实生态模型性能。在真实 GEE 遥感数据下，预期 AUC 可达到 0.70-0.90。

---

## 🔮 开发路线图

### 短期（1-2 周）
- [ ] 修复 GEE 认证，跑通真实遥感数据流水线
- [ ] 将 LangGraph 子 Agent 串联为真正的多 Agent 图
- [ ] 增加 XGBoost 和 MaxEnt 算法

### 中期（1-2 月）
- [ ] 完整接入 biomod2 框架（R 语言）
- [ ] 添加变量重要性、响应曲线、SHAP 可解释性分析
- [ ] 支持 CMIP6 未来气候情景投影
- [ ] 用 5 个以上真实物种做基准测试

### 长期（3-6 月）
- [ ] 多时相预测（季节/年际变化）
- [ ] 集成更多数据源（WorldClim, Bio-ORACLE, MARSPEC）
- [ ] 模型集成（Ensemble）与不确定性量化
- [ ] 用户研究：生态学研究者使用测试

---

## 🎓 学术发表路线图

### 研究定位

本工作位于 **AI for Science** × **Ecological Informatics** 的交叉领域。核心科学问题是：

> **"能否利用 LLM 驱动的多智能体协作系统，自动化完成物种分布建模全流程，降低生态建模的技术门槛，同时保证建模的科学严谨性？"**

### 与现有工具的对比定位

| 工具 | 类型 | 交互方式 | 自动化程度 | 算法覆盖 |
|------|------|----------|-----------|---------|
| **biomod2** (R) | R 包 | 脚本编程 | 低 | 10+ 种算法 |
| **Wallace** (R Shiny) | GUI | 图形界面引导 | 中 | MaxEnt 为主 |
| **sdmTMB** (R) | R 包 | 脚本编程 | 低 | 地统计模型 |
| **Google Earth Engine** | Web IDE | JavaScript/Python | 中 | 仅数据提取 |
| **SDM Agents** (本工作) | LLM Agent 系统 | 自然语言对话 | **高** | 可扩展 |

**核心创新点**：
1. **首次将 LLM Agent 架构引入 SDM 领域** — 用自然语言替代脚本编程
2. **自动纠错与自适应** — 智能体可检测配置错误并自动修复，无需人工介入
3. **多源数据无缝集成** — GBIF + OBIS + GEE（8个遥感数据集）自动融合
4. **端到端自动化** — 从数据获取到预测制图+报告生成，一个自然语言指令完成

### 建议投稿期刊

| 期刊 | IF (2024) | 特点 | 推荐度 |
|------|-----------|------|:----:|
| **Methods in Ecology and Evolution** | ~6.5 | 生态学方法学旗舰刊，非常适合 | ⭐⭐⭐⭐⭐ |
| **Ecological Informatics** | ~5.0 | 生态信息学专业期刊 | ⭐⭐⭐⭐ |
| **Environmental Modelling & Software** | ~5.5 | 环境建模软件类 | ⭐⭐⭐⭐ |
| **Diversity and Distributions** | ~4.5 | SDM 应用研究多 | ⭐⭐⭐ |
| **Scientific Data** | ~5.8 | 如果有 benchmark 数据集贡献 | ⭐⭐⭐ |

### 发表所需的核心实验

#### 实验 1：基准性能验证
- 选取 **10 个代表性海洋物种**（上层鱼类 ×3、底栖鱼类 ×3、甲壳类 ×2、头足类 ×2）
- 选取 **3 个研究区**（南海、地中海、东北太平洋）
- 与 biomod2 全手动流程对比：
  - 建模效率（人工耗时 vs 智能体耗时）
  - 模型性能（AUC, TSS, Boyce Index）
  - 结果可复现性（多次运行方差）

#### 实验 2：消融实验
- 对比有无自动纠错机制的成功率
- 对比有无 GEE 预检的失败率
- 对比 LLM 因子推荐 vs 专家手动选择的模型性能

#### 实验 3：用户研究
- 招募 10-15 名生态学研究生
- A/B 测试：传统 R 脚本 vs SDM Agents
- 记录：任务完成时间、错误次数、主观满意度

#### 实验 4：鲁棒性测试
- 故意制造数据缺失（模拟真实研究中常见的坑）
- 测试自动纠错覆盖率
- 边界情况：物种名拼写错误、极端少的出现点、跨日期变更线

### 论文结构建议

```
1. Introduction
   - 传统 SDM 流程的痛点（技术门槛高、步骤繁琐、易出错）
   - LLM Agent 在其他科学领域的成功应用
   - 本工作的目标和贡献

2. Methods
   2.1 系统架构（多 Agent 协作模式）
   2.2 数据获取智能体（GBIF/OBIS/GEE 集成）
   2.3 建模智能体（空间 CV、算法选择、自动纠错）
   2.4 评估与投影智能体
   2.5 自然语言交互层

3. Experiments
   3.1 基准对比（vs biomod2 手动流程）
   3.2 消融实验
   3.3 用户研究
   3.4 案例研究（2-3 个代表性物种的完整分析）

4. Results
   4.1 建模效率
   4.2 模型性能对比
   4.3 用户研究结果
   4.4 自动纠错覆盖率

5. Discussion
   5.1 LLM Agent 在生态建模中的优势与局限
   5.2 对"黑箱"担忧的回应（所有代码/模型可审计）
   5.3 未来展望（多模态、实时数据同化）

6. Conclusion
```

### 时间线建议

```
月份 1-2:  修复 GEE，跑通真实数据，补齐算法（biomod2 集成）
月份 3:    完成基准实验（10 物种 × 3 区域）
月份 4:    消融实验 + 用户研究
月份 5:    撰写初稿 + 补充实验
月份 6:    投稿 → 修改 → 再投
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

MIT License

---

<p align="center">
  <b>🌊 Built with LangChain + LangGraph + Earth Engine + biomod2 🌊</b>
</p>
