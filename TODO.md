# SDM Agents — 待改进清单

> 按优先级排列，每完成一项标记 `[x]`。建议按顺序逐个攻破。

---

## 🔴 P0 — 阻塞真实使用

### [ ] P0-1 GEE 认证失败

**现象**：所有运行日志显示 `GEE fallback: 初始化失败`，环境因子走 `synthetic_fallback`，模型用三角函数公式而非真实遥感数据训练

**根因**：`_check_gee_ready()` 调用 `ee.Initialize()` 超时 — 国内网络无法直连 `oauth2.googleapis.com`

**修复方案**：
- 方式1：代理/VPN 环境下运行 `python -c "import ee; ee.Authenticate(); ee.Initialize()"`
- 方式2：配置 GEE 服务账号的 `GOOGLE_APPLICATION_CREDENTIALS`
- 验证方法：`python -c "import ee; ee.Initialize(); print(ee.Image('NOAA/CDR/OISST/V2_1').getInfo()['id'])"`

**涉及文件**：`agents/orchestrator/agent.py` — `_check_gee_ready()`

---

### [x] P0-2 LangGraph 子 Agent 未串联到主流水线

**现象**：`agents/gee_data_fetcher/`、`bg_generator/`、`sdm_trainer/`、`sdm_evaluator/`、`sdm_projector/` 各自是独立的 LangGraph Agent，但 `SDMOrchestrator.run()` 完全没调用它们，而是把所有逻辑都内联在了 1300 行的 `agent.py` 里

**需要做的**：
- 设计多 Agent 图：Orchestrator → BG Generator → GEE Fetcher → Trainer → Evaluator → Projector
- 每个子 Agent 用 LangGraph `StateGraph` 编译为子图，Orchestrator 作为父图 `add_node` + `add_edge` 串联
- 统一 State schema（目前三个子 Agent 各自定义了不同的 `SDMState`/`DataFetchState`）

**涉及文件**：全部 `agents/*/agent.py`

---

## 🟡 P1 — 模型科学性

### [x] P1-1 算法太少

**当前**：只有 RandomForest + LogisticRegression（scikit-learn）
**目标**：加入 XGBoost、GAM（pyGAM）、GBM（LightGBM）
**涉及文件**：`agents/orchestrator/agent.py` — `_train_models()`

---

### [x] P1-2 R/biomod2 未真正接入

**当前**：`r_bridge.py` 调的是简陋的 `randomForest` R 脚本（`run_biomod2.R`），不是真正的 biomod2 框架
**目标**：重写 R 脚本，接入：
- `biomod2::BIOMOD_FormatingData()`
- `biomod2::BIOMOD_Modeling()`
- `biomod2::BIOMOD_EnsembleModeling()`
- `biomod2::BIOMOD_Projection()`

**涉及文件**：`agents/sdm_trainer/scripts/run_biomod2.R`、`agents/sdm_trainer/tools/r_bridge.py`

---

### [x] P1-3 无可解释性分析

**当前**：只输出 ROC 曲线和混淆矩阵
**需要加**：
- 变量重要性排名（Permutation importance / SHAP summary plot）
- 偏依赖图 (Partial Dependence Plots)
- 响应曲线（每个环境因子 vs 适宜度）
- SHAP 瀑布图（单样本解释）

**涉及文件**：`agents/orchestrator/agent.py` — `_evaluate()`、`_build_report()`

---

### [x] P1-4 无模型集成 (Ensemble)

**当前**：只选单个最优模型
**目标**：多算法加权/中位数集成，输出一致性图（committee averaging）
**涉及文件**：`agents/orchestrator/agent.py` — `_train_models()`、`_predict_map()`

---

### [x] P1-5 预测图同样依赖合成数据

**当前**：`_predict_map()` 对预测网格做 GEE 提取，GEE 失败就回退合成公式
**根因**：同 P0-1，GEE 通了即解决
**涉及文件**：`agents/orchestrator/agent.py` — `_predict_map()`

---

## 🟠 P2 — 工程健壮性

### [ ] P2-1 无单元测试

**当前**：1300+ 行 orchestrator/agent.py 零测试覆盖
**需要**：给以下方法写 pytest：
- `_build_plan()` — 验证各 data_mode 下配置正确
- `_split_data()` — 验证 5 种切分策略产出正确形状
- `_train_models()` — 验证 CV 评分和模型选择
- `_compute_cv_scores()` — 验证 fold 分配正确
- `occurrence_tools.py` — GBIF/OBIS API mock 测试

---

### [ ] P2-2 无示例数据

**当前**：新用户不知道 CSV 应该长什么样
**需要**：在 `examples/` 目录放：
- `example_full_dataset.csv`（data_mode=upload 示例）
- `example_presence_points.csv`（data_mode=gee_extract 示例）
- `example_config.yaml`（带注释的完整配置）

---

### [ ] P2-3 CV fold assignments 重复计算

**现象**：`_split_data()` 已经算好了 fold assignments 并保存到 `cv_fold_assignments.csv`，但 `_compute_cv_scores()` 又重新跑了一遍 KMeans/GroupKFold
**修复**：让 `_train_models()` 的 CV 分支直接读取 `state.artifacts["cv_assignments"]`

---

### [ ] P2-4 workspace 无清理机制

**现象**：19 次运行产物全部保留，占用空间
**需要**：加 `--keep N` 参数保留最近 N 次运行；或 `--clean` 标志删除旧运行

---

## 🔵 P3 — 功能扩展

### [ ] P3-1 无未来气候情景投影

**当前**：只能预测当前时段
**目标**：接入 CMIP6 未来气候数据（ssp126/ssp370/ssp585），做多时相预测对比
**数据源**：WorldClim future、Bio-ORACLE future

---

### [ ] P3-2 只支持海洋物种

**当前**：`datasets.json` 只有 8 个海洋遥感数据集
**目标**：加陆地数据集字典 — WorldClim 19 个生物气候变量 + SRTM 高程 + 土地利用
**涉及文件**：`agents/gee_data_fetcher/tools/references/datasets.json`

---

### [ ] P3-3 只输出静态 PNG

**目标**：生成 Leaflet/folium 交互式 HTML 地图，叠加存在点和适宜度图层

---

### [ ] P3-4 Web 界面功能有限

**当前**：`dashboard.html` 能聊天但不能可视化结果
**目标**：Dashboard 增加在线地图预览、指标仪表盘、历史运行对比

---

## ⚪ P4 — 发表准备

### [ ] P4-1 基准对比实验

- 选 10 个代表性海洋物种（上层 ×3、底栖 ×3、甲壳 ×2、头足 ×2）
- 选 3 个研究区（南海、地中海、东北太平洋）
- 与 biomod2 全手动流程对比：效率 / AUC / TSS / Boyce / 可复现性

---

### [ ] P4-2 消融实验

- 关/开 `auto_repair` → 成功率对比
- 关/开 `enable_gee_precheck` → 失败率对比
- LLM 因子推荐 vs 专家手动选择 → AUC 差异

---

### [ ] P4-3 用户研究

- 招募 10-15 名生态学研究生
- A/B 测试：传统 R 脚本 vs SDM Agents
- 指标：任务完成时间、错误次数、主观满意度

---

## 执行路线

```
本周            下周            2-4周            1-2月            3-6月
┌──────┐      ┌──────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│P0-1  │  →   │P0-2  │  →   │P1-1,2,3  │  →   │P2-1,2    │  →   │P4-1,2,3  │
│GEE   │      │Agent │      │算法补齐   │      │测试示例   │      │基准用户   │
│认证  │      │串联  │      │可解释性   │      │工程完善   │      │论文写作   │
└──────┘      └──────┘      └──────────┘      └──────────┘      └──────────┘
```
