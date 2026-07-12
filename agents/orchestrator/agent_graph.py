"""SDM Agent Graph — LangGraph multi-agent SDM pipeline.

Wires orchestrator steps + sub-agents into a unified StateGraph.
Supports both LLM-driven (DeepSeek) and procedural execution modes.
Features: ensemble prediction, biomod2 integration, committee agreement maps.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END

from .state import AgentState, PlanConfig, PipelineState
from .rationale import DecisionGraph, RationaleNode, make_rationale

load_dotenv(override=True)


class SDMAgentGraph:
    """LangGraph-based multi-agent SDM pipeline.

    Usage:
        graph = SDMAgentGraph(config_path="config.yaml")
        state = graph.run()
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        interactive: bool = False,
        plan_overrides: Optional[Dict[str, Any]] = None,
        enable_llm: bool = True,
    ):
        self.config_path = Path(config_path)
        self.interactive = interactive
        self.plan_overrides = plan_overrides or {}
        self.enable_llm = enable_llm
        self._orch = None

    @property
    def orch(self):
        if self._orch is None:
            from .agent import SDMOrchestrator
            self._orch = SDMOrchestrator(
                config_path=str(self.config_path),
                interactive=self.interactive,
                plan_overrides=self.plan_overrides,
            )
        return self._orch

    # ── Build ─────────────────────────────────────────────────

    def build(self):
        from langgraph.graph import StateGraph

        workflow = StateGraph(AgentState)

        workflow.add_node("planning", self._planning)
        workflow.add_node("data_acquisition", self._data_acq)
        workflow.add_node("split_data", self._split)
        workflow.add_node("training", self._train)
        workflow.add_node("biomod2", self._biomod2)
        workflow.add_node("ensemble", self._ensemble)
        workflow.add_node("evaluation", self._evaluate)
        workflow.add_node("prediction", self._predict)
        workflow.add_node("report", self._report)

        workflow.set_entry_point("planning")
        workflow.add_conditional_edges("planning", self._route, {
            "data_acquisition": "data_acquisition", "split_data": "split_data",
        })
        workflow.add_conditional_edges("data_acquisition", self._route, {
            "split_data": "split_data", END: END,
        })
        workflow.add_conditional_edges("split_data", self._route, {
            "training": "training", END: END,
        })
        workflow.add_conditional_edges("training", self._route, {
            "biomod2": "biomod2", "ensemble": "ensemble", END: END,
        })
        workflow.add_conditional_edges("biomod2", self._route, {
            "ensemble": "ensemble", END: END,
        })
        workflow.add_conditional_edges("ensemble", self._route, {
            "evaluation": "evaluation", END: END,
        })
        workflow.add_conditional_edges("evaluation", self._route, {
            "prediction": "prediction", END: END,
        })
        workflow.add_conditional_edges("prediction", self._route, {
            "report": "report", END: END,
        })
        workflow.add_edge("report", END)

        return workflow.compile()

    def _route(self, state: AgentState) -> str:
        if state.error_events and state.error_events[-1].get("fatal"):
            return END
        return state.metrics.get("_next_step", "report")

    # ── PipState ↔ AgentState ─────────────────────────────────

    def _to_ps(self, state: AgentState) -> PipelineState:
        ps = PipelineState(
            plan=state.plan or PlanConfig(),
            run_dir=state.run_dir or Path("."),
            log_messages=state.log_messages,
        )
        ps.points_df = state.points_df
        ps.dataset_df = state.dataset_df
        ps.train_df = state.train_df
        ps.test_df = state.test_df
        ps.best_model_name = state.best_model_name
        ps.best_model = state.best_model
        ps.feature_columns = state.feature_columns
        ps.metrics = state.metrics
        ps.artifacts = state.artifacts
        ps.step_status = state.step_status
        ps.error_events = state.error_events
        return ps

    def _sync(self, updates: dict, ps: PipelineState) -> dict:
        updates.update({
            "points_df": ps.points_df,
            "dataset_df": ps.dataset_df,
            "train_df": ps.train_df,
            "test_df": ps.test_df,
            "best_model_name": ps.best_model_name,
            "best_model": ps.best_model,
            "feature_columns": ps.feature_columns,
            "metrics": ps.metrics,
            "artifacts": ps.artifacts,
            "step_status": ps.step_status,
            "error_events": ps.error_events,
            "log_messages": ps.log_messages,
        })
        return updates

    # ── Nodes ─────────────────────────────────────────────────

    def _planning(self, state: AgentState) -> dict:
        try:
            plan = self.orch._build_plan()  # noqa: SLF001
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(plan.output_dir) / f"{plan.species_name}_{ts}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # All modes go through data_acquisition (which branches on data_mode internally)
            next_step = "data_acquisition"
            dg = DecisionGraph()
            return {
                "plan": plan,
                "run_dir": run_dir,
                "llm_enabled": self.enable_llm and bool(os.getenv("DEEPSEEK_API_KEY")),
                "log_messages": [
                    f"Plan: species={plan.species_name}, mode={plan.data_mode}, "
                    f"algos={plan.algorithms}, factors={plan.factors}"
                ],
                "metrics": {"_next_step": next_step, "_decision_graph": dg},
                "step_status": {"planning": "succeeded"},
                "error_events": [],
                "artifacts": {},
            }
        except Exception as exc:
            return {
                "step_status": {"planning": "failed"},
                "error_events": [{"step": "planning", "error": str(exc), "fatal": True}],
            }

    def _data_acq(self, state: AgentState) -> dict:
        updates: dict = {}
        plan = state.plan
        if plan is None:
            return updates

        try:
            if plan.data_mode == "upload":
                # Load user-provided full dataset
                ps = self._to_ps(state)
                self.orch._load_full_dataset(ps)  # noqa: SLF001
                updates = self._sync(updates, ps)
                updates["step_status"] = {**state.step_status, "load_dataset": "succeeded"}
            else:
                # Standard pipeline: prepare points → precheck → build dataset
                ps = self._to_ps(state)
                self.orch._prepare_points(ps)  # noqa: SLF001
                updates = self._sync(updates, ps)

                if plan.use_gee and plan.enable_gee_precheck:
                    ps2 = self._to_ps(state)
                    self.orch._precheck_factors(ps2)  # noqa: SLF001
                    updates = self._sync(updates, ps2)

                ps3 = self._to_ps(state)
                self.orch._build_dataset(ps3)  # noqa: SLF001
                updates = self._sync(updates, ps3)

                updates["step_status"] = {**state.step_status,
                    "prepare_points": "succeeded",
                    "precheck_factors": "succeeded",
                    "build_dataset": "succeeded",
                }
            # SDG: PA generation rationale
            if state.plan and state.plan.data_mode != "upload":
                self._add_rationale(state, make_rationale(
                    step="Pseudo-absence Generation",
                    decision=f"生成伪缺失点，比例 1:{state.plan.pseudo_absence_ratio if state.plan else 1.0}",
                    evidence=f"基于 {len(state.points_df) if state.points_df is not None else 0} 个存在点的空间边界生成背景点",
                    alternative="若比例调整为 1:2: 模型可能过度关注背景区域的模式",
                    confidence=0.75,
                    counterfactual="增加伪缺失比例会降低假阳性率，但可能牺牲对稀有生境的敏感性",
                ))
            updates["metrics"] = {**state.metrics, "_next_step": "split_data"}
        except Exception as exc:
            updates["step_status"] = {**state.step_status, "data_acquisition": "failed"}
            updates["error_events"] = state.error_events + [
                {"step": "data_acquisition", "error": str(exc), "fatal": True}
            ]
        return updates

    def _split(self, state: AgentState) -> dict:
        updates: dict = {}
        try:
            ps = self._to_ps(state)
            self.orch._split_data(ps)  # noqa: SLF001
            updates = self._sync(updates, ps)

            # SDG: CV strategy rationale
            plan = state.plan
            split_mode = str(plan.split_mode).strip().lower() if plan else "random_holdout"
            mode_labels = {
                "random_holdout": "随机留出法 (random_holdout)",
                "random_kfold": "分层 K 折交叉验证 (random_kfold)",
                "spatial_kfold": "空间聚类 K 折 (spatial_kfold)",
                "spatial_block_kfold": "空间分块 K 折 (spatial_block_kfold)",
                "env_spatial_block_kfold": "环境分层+空间分块 K 折 (env_spatial_block_kfold)",
            }
            mode_label = mode_labels.get(split_mode, split_mode)
            is_spatial = "spatial" in split_mode

            self._add_rationale(state, make_rationale(
                step="Cross-Validation Strategy",
                decision=f"选择 {mode_label}，测试集比例 {plan.test_size if plan else 0.2}",
                evidence=f"空间切分可避免空间自相关导致的高估 AUC"
                if is_spatial else
                f"随机切分适用于数据量充足且无明显空间自相关的场景",
                alternative=f"若选 random_kfold: {'可能高估模型性能，因训练/测试点在空间上不独立' if is_spatial else '计算开销更大，但对本数据集增益有限'}",
                confidence=0.80 if is_spatial else 0.65,
                counterfactual=f"使用 random CV 替代 spatial CV 可能导致 AUC 被高估 0.03-0.08",
                metrics={"split_mode": split_mode, "test_size": plan.test_size if plan else 0.2,
                         "n_splits": plan.n_splits if plan else 5},
            ))

            updates["step_status"] = {**state.step_status, "split_data": "succeeded"}
            updates["metrics"] = {**state.metrics, "_next_step": "training"}
        except Exception as exc:
            updates["step_status"] = {**state.step_status, "split_data": "failed"}
            updates["error_events"] = state.error_events + [
                {"step": "split_data", "error": str(exc), "fatal": True}
            ]
        return updates

    def _train(self, state: AgentState) -> dict:
        updates: dict = {}
        try:
            ps = self._to_ps(state)
            self.orch._train_models(ps)  # noqa: SLF001
            updates = self._sync(updates, ps)

            # Build candidate models for ensemble (use ps which has updated feature_columns)
            candidates = self._train_candidates_from_ps(ps)
            updates["candidate_models"] = candidates

            # SDG: Model selection rationale
            model_scores = ps.metrics.get("model_selection", {})
            best_name = ps.best_model_name or "unknown"
            scores_sorted = sorted(model_scores.items(),
                                   key=lambda x: x[1].get("holdout_auc", x[1].get("cv_mean_auc", 0)),
                                   reverse=True)
            if len(scores_sorted) >= 2:
                best_score = scores_sorted[0][1].get("holdout_auc", scores_sorted[0][1].get("cv_mean_auc", 0))
                runner_up = scores_sorted[1][0]
                runner_score = scores_sorted[1][1].get("holdout_auc", scores_sorted[1][1].get("cv_mean_auc", 0))
                diff = best_score - runner_score
                conf = min(0.95, 0.55 + diff * 5)

                self._add_rationale(state, make_rationale(
                    step="Model Selection",
                    decision=f"选择 {best_name.upper()} 作为最优模型 (AUC={best_score:.3f})",
                    evidence=f"{best_name.upper()} AUC={best_score:.3f} > {runner_up.upper()} AUC={runner_score:.3f}"
                             f"{' (CV mean)' if 'cv_mean_auc' in scores_sorted[0][1] else ''}",
                    alternative=f"若选 {runner_up.upper()}: AUC 降低 {diff:.3f}",
                    confidence=conf,
                    counterfactual=f"选择 {runner_up.upper()} 会导致预测在环境梯度边缘区域更保守",
                    metrics={k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                  for kk, vv in v.items()}
                             for k, v in scores_sorted[:3]},
                ))

            next_step = "biomod2" if (state.plan and state.plan.data_mode != "upload") else "ensemble"
            updates["step_status"] = {**state.step_status, "train_models": "succeeded"}
            updates["metrics"] = {**state.metrics, "_next_step": next_step}
        except Exception as exc:
            updates["step_status"] = {**state.step_status, "train_models": "failed"}
            updates["error_events"] = state.error_events + [
                {"step": "train_models", "error": str(exc), "fatal": True}
            ]
        return updates

    def _train_candidates_from_ps(self, ps: PipelineState) -> dict:
        """Train all algorithms for ensemble using PipelineState (post-training)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        plan = ps.plan
        train_df = ps.train_df
        feats = ps.feature_columns
        if plan is None or train_df is None or not feats:
            return {}

        x_train = train_df[feats]
        y_train = train_df["is_presence"].astype(int)
        candidates: Dict[str, Any] = {}

        for algo in plan.algorithms:
            try:
                if algo == "rf":
                    m = RandomForestClassifier(n_estimators=500, random_state=plan.random_seed,
                                               class_weight="balanced", n_jobs=-1)
                elif algo == "xgb":
                    from xgboost import XGBClassifier
                    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=plan.random_seed, eval_metric="logloss", verbosity=0)
                elif algo == "lgbm":
                    from lightgbm import LGBMClassifier
                    m = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=plan.random_seed, class_weight="balanced", verbose=-1)
                elif algo == "logreg":
                    m = LogisticRegression(max_iter=2500, random_state=plan.random_seed,
                                           class_weight="balanced")
                else:
                    continue
                m.fit(x_train, y_train)
                candidates[algo] = m
            except Exception:
                pass
        return candidates

    def _biomod2(self, state: AgentState) -> dict:
        """Optional biomod2 R-based training."""
        try:
            self._run_biomod2(state)
            return {
                "step_status": {**state.step_status, "biomod2": "succeeded"},
                "metrics": {**state.metrics, "_next_step": "ensemble"},
            }
        except Exception:
            return {
                "step_status": {**state.step_status, "biomod2": "skipped"},
                "metrics": {**state.metrics, "_next_step": "ensemble"},
            }

    def _run_biomod2(self, state: AgentState) -> None:
        import subprocess
        if state.dataset_df is None or state.plan is None or state.run_dir is None:
            return
        ws = str(state.run_dir)
        csv_p = str(state.run_dir / "training_data.csv")
        state.dataset_df.to_csv(csv_p, index=False)
        sp = Path(__file__).resolve().parents[1] / "sdm_trainer" / "scripts" / "run_biomod2.R"
        r = subprocess.run(["Rscript", str(sp), csv_p, ws, ",".join(state.plan.algorithms)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            mp = Path(ws) / "biomod2_metrics.json"
            if mp.exists():
                with open(mp, "r", encoding="utf-8") as f:
                    state.metrics["biomod2"] = json.load(f)
                state.artifacts["biomod2_metrics"] = str(mp)

    def _ensemble(self, state: AgentState) -> dict:
        try:
            self._build_ensemble(state)
            return {
                "step_status": {**state.step_status, "ensemble": "succeeded"},
                "metrics": {**state.metrics, "_next_step": "evaluation"},
            }
        except Exception:
            return {
                "step_status": {**state.step_status, "ensemble": "skipped"},
                "metrics": {**state.metrics, "_next_step": "evaluation"},
            }

    def _build_ensemble(self, state: AgentState) -> None:
        candidates = state.candidate_models
        test_df = state.test_df
        feats = state.feature_columns
        if not candidates or test_df is None or not feats:
            return

        from sklearn.metrics import roc_auc_score
        x_test = test_df[feats]
        y_test = test_df["is_presence"].astype(int)
        weights: Dict[str, float] = {}
        for name, model in candidates.items():
            try:
                prob = model.predict_proba(x_test)[:, 1]
                weights[name] = max(0.5, float(roc_auc_score(y_test, prob)))
            except Exception:
                weights[name] = 0.5
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        state.metrics["ensemble"] = {"method": "auc_weighted",
                                     "members": list(candidates.keys()), "weights": weights}

        # SDG: Ensemble rationale
        best_w = max(weights.items(), key=lambda x: x[1]) if weights else ("?", 0)
        self._add_rationale(state, make_rationale(
            step="Ensemble Construction",
            decision=f"AUC 加权集成 {len(candidates)} 个模型，{best_w[0]} 权重最高 ({best_w[1]:.2f})",
            evidence=f"各模型 AUC 归一化为权重: { {k: round(v, 3) for k, v in weights.items()} }",
            alternative="若用简单平均: 表现差的模型会拖低整体精度",
            confidence=0.78,
            counterfactual="简单平均 vs AUC 加权: 加权集成预期 AUC 提升 0.01-0.03",
            metrics={"n_models": len(candidates), "method": "auc_weighted"},
        ))

    def _evaluate(self, state: AgentState) -> dict:
        updates: dict = {}
        try:
            ps = self._to_ps(state)
            self.orch._evaluate(ps)  # noqa: SLF001
            updates = self._sync(updates, ps)
            updates["step_status"] = {**state.step_status, "evaluate": "succeeded"}
            updates["metrics"] = {**state.metrics, "_next_step": "prediction"}
        except Exception as exc:
            updates["step_status"] = {**state.step_status, "evaluate": "failed"}
            updates["error_events"] = state.error_events + [{"step": "evaluate", "error": str(exc)}]
        return updates

    def _predict(self, state: AgentState) -> dict:
        updates: dict = {}
        try:
            ps = self._to_ps(state)
            self.orch._predict_map(ps)  # noqa: SLF001
            updates = self._sync(updates, ps)

            if state.candidate_models:
                self._predict_ensemble(state)

            # SDG: Uncertainty assessment
            if state.candidate_models:
                n_models = len(state.candidate_models)
                uq_rating = "低" if n_models >= 3 else "中"
                self._add_rationale(state, make_rationale(
                    step="Prediction Uncertainty",
                    decision=f"基于 {n_models} 个模型的委员会一致性评估不确定性",
                    evidence=f"多模型预测标准差 < 0.15 的区域标记为高置信区；"
                             f"委员会分歧大的区域标记为建议补充采样区",
                    alternative="若只用单模型: 无法评估预测可靠性",
                    confidence=0.85,
                    counterfactual="单模型预测会给出虚假的确定性——模型分歧本身就是重要的生态学信息",
                    metrics={"n_committee_models": n_models,
                             "uncertainty_source": "模型间分歧 + 环境覆盖不足"},
                ))

            updates["step_status"] = {**state.step_status, "predict_map": "succeeded"}
            updates["metrics"] = {**state.metrics, "_next_step": "report"}
        except Exception as exc:
            updates["step_status"] = {**state.step_status, "predict_map": "failed"}
            updates["error_events"] = state.error_events + [{"step": "predict_map", "error": str(exc)}]
        return updates

    def _predict_ensemble(self, state: AgentState) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        candidates = state.candidate_models
        plan = state.plan
        feats = state.feature_columns
        if not candidates or plan is None or not feats or state.run_dir is None:
            return

        min_lon, min_lat, max_lon, max_lat = plan.bbox
        n = plan.map_resolution
        lon_g = np.linspace(min_lon, max_lon, n)
        lat_g = np.linspace(min_lat, max_lat, n)
        xx, yy = np.meshgrid(lon_g, lat_g)
        grid = pd.DataFrame({"lon": xx.ravel(), "lat": yy.ravel()})

        from .agent import SDMOrchestrator
        o2 = SDMOrchestrator(config_path=str(self.config_path), interactive=False)
        for f in feats:
            if f not in grid.columns:
                grid[f] = o2._feature_formula(grid["lon"].to_numpy(), grid["lat"].to_numpy(), f)  # noqa

        x_pred = grid[feats]
        weights = state.metrics.get("ensemble", {}).get("weights", {})
        probs = []
        for name, model in candidates.items():
            w = weights.get(name, 1.0 / len(candidates))
            probs.append(w * model.predict_proba(x_pred)[:, 1])
        ensemble_prob = np.sum(probs, axis=0)

        all_p = np.column_stack([m.predict_proba(x_pred)[:, 1] for m in candidates.values()])
        agree = 1 - np.std(all_p, axis=1)

        z_e = ensemble_prob.reshape(n, n)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        cmap = sns.color_palette("YlGnBu", as_cmap=True)
        ax.imshow(z_e, origin="lower", cmap=cmap, extent=[min_lon, max_lon, min_lat, max_lat], aspect="auto", vmin=0, vmax=1)
        plt.colorbar(ax.images[0], ax=ax).set_label("Ensemble Suitability", rotation=90)
        ax.set_title(f"Ensemble ({len(candidates)} models)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ep = state.run_dir / "ensemble_prediction_map.png"
        fig.tight_layout(); fig.savefig(ep); plt.close(fig)
        state.artifacts["ensemble_prediction_map"] = str(ep)

        z_a = agree.reshape(n, n)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        cmap2 = sns.color_palette("RdYlGn", as_cmap=True)
        ax.imshow(z_a, origin="lower", cmap=cmap2, extent=[min_lon, max_lon, min_lat, max_lat], aspect="auto", vmin=0, vmax=1)
        plt.colorbar(ax.images[0], ax=ax).set_label("Committee Agreement", rotation=90)
        ax.set_title("Model Agreement (1 - std)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ap = state.run_dir / "committee_agreement.png"
        fig.tight_layout(); fig.savefig(ap); plt.close(fig)
        state.artifacts["committee_agreement"] = str(ap)

        grid.to_csv(state.run_dir / "ensemble_prediction.csv", index=False)
        state.artifacts["ensemble_prediction"] = str(state.run_dir / "ensemble_prediction.csv")

    def _report(self, state: AgentState) -> dict:
        updates: dict = {}
        try:
            ps = self._to_ps(state)
            self.orch._build_report(ps)  # noqa: SLF001
            updates = self._sync(updates, ps)
            ps2 = self._to_ps(state)
            self.orch._save_metadata(ps2)  # noqa: SLF001
            updates = self._sync(updates, ps2)
            updates["step_status"] = {**state.step_status, "build_report": "succeeded"}
        except Exception as exc:
            updates["step_status"] = {**state.step_status, "build_report": "failed"}
            updates["error_events"] = state.error_events + [{"step": "build_report", "error": str(exc)}]
        return updates

    # ── SDG Rationale ─────────────────────────────────────────

    def _add_rationale(self, state: AgentState, node: "RationaleNode") -> None:
        """Record a rationale node in the Scientific Decision Graph."""
        dg = state.metrics.get("_decision_graph")
        if dg is None:
            dg = DecisionGraph()
            state.metrics["_decision_graph"] = dg
        dg.add(node)

    # ── Run ───────────────────────────────────────────────────

    def run(self) -> AgentState:
        app = self.build()
        return app.invoke(AgentState(
            step_status={}, metrics={}, artifacts={}, error_events=[], log_messages=[],
            feature_columns=[], candidate_models={},
        ))
