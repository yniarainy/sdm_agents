from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .occurrence_tools import download_occurrence_points, load_presence_points_file, normalize_presence_dataframe
from .state import PipelineState, PlanConfig


class SDMOrchestrator:
    def __init__(
        self,
        config_path: str = "config.yaml",
        interactive: bool = True,
        plan_overrides: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.config_path = Path(config_path)
        self.interactive = interactive
        self.plan_overrides = plan_overrides or {}
        self.progress_callback = progress_callback
        self._gee_ready_cache: Optional[bool] = None

    def run(self) -> PipelineState:
        plan = self._build_plan()
        run_dir = self._create_run_dir(plan.output_dir, plan.species_name)
        state = PipelineState(plan=plan, run_dir=run_dir, progress_callback=self.progress_callback)

        state.log("Pipeline started")
        fatal_error: Optional[Exception] = None

        # Build step list based on data mode
        if plan.data_mode == "upload":
            steps: List[Tuple[str, Callable[[PipelineState], None]]] = [
                ("load_dataset", self._load_full_dataset),
                ("split_data", self._split_data),
                ("train_models", self._train_models),
                ("evaluate", self._evaluate),
                ("predict_map", self._predict_map),
                ("build_report", self._build_report),
            ]
        else:
            steps = [
                ("prepare_points", self._prepare_points),
                ("precheck_factors", self._precheck_factors),
                ("build_dataset", self._build_dataset),
                ("split_data", self._split_data),
                ("train_models", self._train_models),
                ("evaluate", self._evaluate),
                ("predict_map", self._predict_map),
                ("build_report", self._build_report),
            ]

        for step_name, fn in steps:
            try:
                self._run_step(state, step_name, fn)
            except Exception as exc:
                fatal_error = exc
                state.step_status[step_name] = "failed"
                state.log(f"Pipeline abort: {exc}")
                break

        state.log("Pipeline finished" if fatal_error is None else "Pipeline finished with failure")
        try:
            self._save_metadata(state)
        except Exception as exc:
            state.step_status["save_metadata"] = "failed"
            state.log(f"[save_metadata] failed: {exc}")
            raise

        if fatal_error is not None:
            raise RuntimeError(str(fatal_error))
        return state

    def _run_step(
        self,
        state: PipelineState,
        step_name: str,
        fn: Callable[[PipelineState], None],
    ) -> None:
        max_attempts = max(1, int(state.plan.max_retries) + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                state.log(f"[{step_name}] started (attempt {attempt}/{max_attempts})")
                fn(state)
                state.step_status[step_name] = "succeeded"
                state.log(f"[{step_name}] succeeded")
                return
            except Exception as exc:
                state.error_events.append(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "step": step_name,
                        "attempt": str(attempt),
                        "error": str(exc),
                    }
                )
                state.step_status[step_name] = "retrying"
                state.log(f"[{step_name}] failed on attempt {attempt}: {exc}")
                if attempt >= max_attempts:
                    state.step_status[step_name] = "failed"
                    raise RuntimeError(f"步骤 {step_name} 连续失败 {max_attempts} 次，流程终止。") from exc
                self._auto_repair(state, step_name, exc)
                state.log(f"[{step_name}] auto-fix applied; retrying...")

    def _auto_repair(self, state: PipelineState, step_name: str, exc: Exception) -> None:
        plan = state.plan
        if step_name == "prepare_points" and plan.presence_points_path:
            plan.presence_points_path = ""
            state.log("Auto-fix: invalid presence file path, switched to GBIF/OBIS download.")
            return

        if step_name == "load_dataset":
            if plan.full_dataset_path:
                plan.full_dataset_path = ""
                state.log("Auto-fix: cleared invalid full_dataset_path. 请手动提供有效路径或切换 data_mode。")
                return
            # Fallback: switch to gbif_obis mode
            plan.data_mode = "gbif_obis"
            state.log("Auto-fix: data_mode 已从 'upload' 切换为 'gbif_obis'，将自动下载存在点。")
            return

        if step_name == "build_dataset":
            # If in gee_extract mode, try switching to gbif_obis (allows fallback)
            if str(plan.data_mode).strip().lower() == "gee_extract":
                plan.data_mode = "gbif_obis"
                state.log("Auto-fix: data_mode 从 'gee_extract' 切换为 'gbif_obis'（允许合成特征回退）。")
                return

            registry = self._load_gee_registry()
            valid_factors = [f for f in plan.factors if f in registry]
            if valid_factors and valid_factors != plan.factors:
                plan.factors = valid_factors
                state.log("Auto-fix: removed unsupported factors from plan.")
                return

        if step_name == "split_data" and plan.test_size >= 0.5:
            plan.test_size = 0.2
            state.log("Auto-fix: test_size too large, reset to 0.2.")
            return

        if step_name == "split_data" and plan.n_splits < 2:
            plan.n_splits = 5
            state.log("Auto-fix: n_splits too small, reset to 5.")
            return

        if step_name == "split_data" and plan.split_mode == "spatial_kfold" and plan.spatial_clusters < plan.n_splits:
            plan.spatial_clusters = max(plan.n_splits, 8)
            state.log(f"Auto-fix: spatial_clusters increased to {plan.spatial_clusters}.")
            return

        if step_name == "split_data" and plan.split_mode == "spatial_block_kfold":
            if plan.spatial_block_bins_lon < 2:
                plan.spatial_block_bins_lon = 4
                state.log("Auto-fix: spatial_block_bins_lon reset to 4.")
                return
            if plan.spatial_block_bins_lat < 2:
                plan.spatial_block_bins_lat = 4
                state.log("Auto-fix: spatial_block_bins_lat reset to 4.")
                return
            if plan.spatial_block_bins_lon * plan.spatial_block_bins_lat < plan.n_splits:
                side = int(np.ceil(np.sqrt(plan.n_splits)))
                plan.spatial_block_bins_lon = max(plan.spatial_block_bins_lon, side)
                plan.spatial_block_bins_lat = max(plan.spatial_block_bins_lat, side)
                state.log(
                    f"Auto-fix: spatial block grid increased to {plan.spatial_block_bins_lon}x{plan.spatial_block_bins_lat}."
                )
                return

        if step_name == "split_data" and plan.split_mode == "env_spatial_block_kfold":
            if plan.env_strata_bins < 2:
                plan.env_strata_bins = 4
                state.log("Auto-fix: env_strata_bins reset to 4.")
                return
            if plan.spatial_block_bins_lon < 2:
                plan.spatial_block_bins_lon = 4
                state.log("Auto-fix: spatial_block_bins_lon reset to 4.")
                return
            if plan.spatial_block_bins_lat < 2:
                plan.spatial_block_bins_lat = 4
                state.log("Auto-fix: spatial_block_bins_lat reset to 4.")
                return

        if step_name == "train_models":
            plan.algorithms = ["rf"]
            state.log("Auto-fix: reduced algorithms to rf only.")
            return

        if step_name == "predict_map":
            if plan.map_resolution > 180:
                plan.map_resolution = 120
            else:
                plan.map_resolution = max(80, int(plan.map_resolution * 0.8))
            state.log(f"Auto-fix: reduced map_resolution to {plan.map_resolution}.")
            return

        raise RuntimeError(f"Unable to auto-fix step '{step_name}': {exc}") from exc

    def _build_plan(self) -> PlanConfig:
        defaults = self._load_yaml_defaults()
        plan = PlanConfig(**defaults)
        registry = self._load_gee_registry()

        if self.interactive:
            plan = self._ask_plan_questions(plan, registry)

        # Upload mode: user brings their own env data — skip factor validation against GEE registry
        if plan.data_mode == "upload":
            for key, value in self.plan_overrides.items():
                if hasattr(plan, key):
                    setattr(plan, key, value)
            plan.factors = list(dict.fromkeys(plan.factors))
            return plan

        # Non-interactive mode still validates user factors and records missing ones.
        requested = plan.factors[:] if plan.factors else []
        valid, missing = self._validate_factors(requested, registry)
        if missing:
            print(f"Planner 提示: 以下环境变量缺失并将忽略: {', '.join(missing)}")
        if valid:
            plan.factors = valid
        elif not requested:
            plan.factors = self._recommend_factors(plan.species_name, registry)

        if plan.required_factors:
            required_missing = [f for f in plan.required_factors if f not in plan.factors]
            if required_missing:
                raise ValueError(f"必需环境变量缺失: {required_missing}")

        for key, value in self.plan_overrides.items():
            if hasattr(plan, key):
                setattr(plan, key, value)

        plan.factors = list(dict.fromkeys(plan.factors))
        return plan

    def _load_yaml_defaults(self) -> Dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "bbox" in data:
            data["bbox"] = tuple(data["bbox"])

        # Normalize data_mode
        data.setdefault("data_mode", "gbif_obis")
        data.setdefault("full_dataset_path", "")
        return data

    def _ask_plan_questions(self, plan: PlanConfig, registry: Dict[str, Dict[str, Any]]) -> PlanConfig:
        print("\n=== SDM Planner Agent: 请输入或回车采用默认值 ===")
        plan.species_name = self._ask_str("物种名", plan.species_name)
        plan.data_mode = self._ask_str(
            "数据来源模式 (upload 自带完整数据 / gbif_obis 下载存在点+GEE提取 / gee_extract 上传存在点+GEE提取)",
            plan.data_mode,
        )

        if plan.data_mode == "upload":
            plan.full_dataset_path = self._ask_str("完整数据集路径 (CSV, 含 lon/lat/is_presence + 环境因子)", plan.full_dataset_path)
            plan.output_dir = self._ask_str("输出目录", plan.output_dir)
            plan.algorithms = [x.strip().lower() for x in self._ask_str("算法(逗号分隔: rf,xgb,lgbm,logreg)", ",".join(plan.algorithms)).split(",") if x.strip()]
            plan.pseudo_absence_ratio = float(self._ask_str("伪缺失比例(相对 presence)", str(plan.pseudo_absence_ratio)))
            plan.test_size = float(self._ask_str("测试集比例", str(plan.test_size)))
            plan.split_mode = self._ask_str(
                "切分策略(random_holdout/random_kfold/spatial_kfold/spatial_block_kfold)",
                plan.split_mode,
            )
            plan.n_splits = int(self._ask_str("交叉验证折数(n_splits)", str(plan.n_splits)))
            plan.spatial_clusters = int(self._ask_str("空间切分聚类数(spatial_clusters)", str(plan.spatial_clusters)))
            plan.map_resolution = int(self._ask_str("预测图网格分辨率(建议80-180)", str(plan.map_resolution)))
            plan.random_seed = int(self._ask_str("随机种子", str(plan.random_seed)))
            plan.max_retries = int(self._ask_str("单步骤最大自动重试次数", str(plan.max_retries)))
            return plan

        plan.presence_source_mode = self._ask_str(
            "存在点来源(upload/gbif/obis/gbif_obis)",
            plan.presence_source_mode,
        )
        plan.occurrence_download_limit = int(self._ask_str("GBIF/OBIS 下载上限", str(plan.occurrence_download_limit)))
        plan.presence_points_path = self._ask_str("出现点文件路径(CSV, 含 lon/lat，可留空)", plan.presence_points_path)
        plan.output_dir = self._ask_str("输出目录", plan.output_dir)
        plan.start_date = self._ask_str("起始日期(YYYY-MM-DD)", plan.start_date)
        plan.end_date = self._ask_str("结束日期(YYYY-MM-DD)", plan.end_date)

        bbox_raw = self._ask_str(
            "研究区 bbox(min_lon,min_lat,max_lon,max_lat)",
            ",".join([str(x) for x in plan.bbox]),
        )
        plan.bbox = tuple(float(x.strip()) for x in bbox_raw.split(","))  # type: ignore[assignment]

        recommended = self._recommend_factors(plan.species_name, registry)
        print(f"Planner 推荐环境变量: {', '.join(recommended)}")
        plan.auto_factor_selection = self._ask_bool("是否使用 Planner 推荐变量", plan.auto_factor_selection)

        if plan.auto_factor_selection:
            selected = recommended
        else:
            selected = [
                x.strip() for x in self._ask_str("环境因子(逗号分隔)", ",".join(plan.factors)).split(",") if x.strip()
            ]

        valid_factors, missing_factors = self._validate_factors(selected, registry)
        if missing_factors:
            print(f"提示: 以下变量在数据字典中不存在，将跳过: {', '.join(missing_factors)}")
        if not valid_factors:
            print("提示: 可用变量为空，自动回退到推荐变量。")
            valid_factors = recommended
        plan.factors = valid_factors

        req_raw = self._ask_str("必需变量(逗号分隔，可空)", ",".join(plan.required_factors))
        plan.required_factors = [x.strip() for x in req_raw.split(",") if x.strip()]
        plan.algorithms = [x.strip().lower() for x in self._ask_str("算法(逗号分隔: rf,xgb,lgbm,logreg)", ",".join(plan.algorithms)).split(",") if x.strip()]

        plan.pseudo_absence_ratio = float(self._ask_str("伪缺失比例(相对 presence)", str(plan.pseudo_absence_ratio)))
        plan.test_size = float(self._ask_str("测试集比例", str(plan.test_size)))
        plan.split_mode = self._ask_str(
            "切分策略(random_holdout/random_kfold/spatial_kfold/spatial_block_kfold)",
            plan.split_mode,
        )
        plan.n_splits = int(self._ask_str("交叉验证折数(n_splits)", str(plan.n_splits)))
        plan.spatial_clusters = int(self._ask_str("空间切分聚类数(spatial_clusters)", str(plan.spatial_clusters)))
        plan.spatial_block_bins_lon = int(self._ask_str("空间分块Lon网格数(spatial_block_bins_lon)", str(plan.spatial_block_bins_lon)))
        plan.spatial_block_bins_lat = int(self._ask_str("空间分块Lat网格数(spatial_block_bins_lat)", str(plan.spatial_block_bins_lat)))
        plan.env_strata_bins = int(self._ask_str("环境分层箱数(env_strata_bins)", str(plan.env_strata_bins)))
        plan.map_resolution = int(self._ask_str("预测图网格分辨率(建议80-180)", str(plan.map_resolution)))
        plan.random_seed = int(self._ask_str("随机种子", str(plan.random_seed)))
        plan.max_retries = int(self._ask_str("单步骤最大自动重试次数", str(plan.max_retries)))
        plan.use_gee = self._ask_bool("是否启用真实 GEE 抽样(当前默认使用稳定内置特征)", plan.use_gee)
        plan.strict_gee = self._ask_bool("是否启用严格 GEE 模式(变量非 GEE 来源即失败)", plan.strict_gee)
        plan.enable_gee_precheck = self._ask_bool("是否启用 GEE 变量可用性预检", plan.enable_gee_precheck)

        return plan

    def _recommend_factors(self, species_name: str, registry: Dict[str, Dict[str, Any]]) -> List[str]:
        name = species_name.lower()
        if any(k in name for k in ["tuna", "金枪", "鲔", "pelagic", "上层"]):
            base = ["sst", "chl_a", "current_u", "current_v", "ssh"]
        elif any(k in name for k in ["cod", "鳕", "demersal", "底栖", "flounder", "比目"]):
            base = ["sst", "salinity", "bathymetry", "chl_a"]
        elif any(k in name for k in ["shrimp", "虾", "crab", "蟹"]):
            base = ["sst", "salinity", "bathymetry", "chl_a"]
        else:
            base = ["sst", "chl_a", "salinity", "bathymetry"]
        return [f for f in base if f in registry]

    def _validate_factors(self, factors: List[str], registry: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        normalized = [f.strip() for f in factors if f.strip()]
        valid = [f for f in normalized if f in registry]
        missing = [f for f in normalized if f not in registry]
        return valid, missing

    def _ask_str(self, prompt: str, default: str) -> str:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default

    def _ask_bool(self, prompt: str, default: bool) -> bool:
        default_str = "y" if default else "n"
        user_input = input(f"{prompt} [y/n, default={default_str}]: ").strip().lower()
        if not user_input:
            return default
        return user_input in {"y", "yes", "true", "1"}

    def _create_run_dir(self, base_output: str, species_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_output) / f"{species_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _prepare_points(self, state: PipelineState) -> None:
        plan = state.plan
        np.random.seed(plan.random_seed)

        if plan.presence_points_path:
            state.log(f"读取上传存在点: {plan.presence_points_path}")
            presence = load_presence_points_file(plan.presence_points_path, species_name=plan.species_name, source_label="upload")
            state.metrics["presence_source"] = {
                "mode": "upload",
                "path": plan.presence_points_path,
                "rows": int(len(presence)),
                "years": [int(presence["year"].min()), int(presence["year"].max())] if not presence.empty else [],
                "months": [int(presence["month"].dropna().min()), int(presence["month"].dropna().max())] if presence.get("month") is not None and presence["month"].notna().any() else [],
            }
        else:
            source_mode = str(plan.presence_source_mode or "gbif_obis").strip().lower()
            try:
                download_result = download_occurrence_points(
                    species_name=plan.species_name,
                    source_mode=source_mode,
                    start_date=plan.start_date,
                    end_date=plan.end_date,
                    limit=int(plan.occurrence_download_limit),
                    timeout=max(10, int(plan.gee_auth_timeout_ms / 1000)),
                )
                presence = download_result.dataframe
                state.metrics["presence_source"] = download_result.source_stats
                state.log(
                    f"存在点已从 {source_mode} 下载并合并: {len(presence)} 条, 年份 {state.metrics['presence_source'].get('year_range', [])}"
                )
            except Exception as exc:
                state.log(f"存在点下载失败，将回退到合成示例点: {exc}")
                min_lon, min_lat, max_lon, max_lat = plan.bbox
                n_presence = 220
                presence = pd.DataFrame(
                    {
                        "lon": np.random.uniform(min_lon, max_lon, n_presence),
                        "lat": np.random.uniform(min_lat, max_lat, n_presence),
                        "year": np.random.randint(pd.to_datetime(plan.start_date).year, pd.to_datetime(plan.end_date).year + 1, n_presence),
                        "month": np.random.randint(1, 13, n_presence),
                        "source": "synthetic_fallback",
                        "species_name": plan.species_name,
                    }
                )
                state.metrics["presence_source"] = {
                    "mode": "synthetic_fallback",
                    "reason": str(exc),
                    "rows": int(len(presence)),
                }

        if "is_presence" not in presence.columns:
            presence["is_presence"] = 1

        n_bg = max(20, int(len(presence) * plan.pseudo_absence_ratio))
        bg = self._generate_pseudo_absence(presence, n_bg, plan.bbox, plan.random_seed)

        points = pd.concat([presence, bg], ignore_index=True)
        state.points_df = points

        points.to_csv(state.run_dir / "points_with_labels.csv", index=False)
        state.artifacts["points"] = str(state.run_dir / "points_with_labels.csv")

    def _load_full_dataset(self, state: PipelineState) -> None:
        """Load a user-provided CSV that already contains is_presence + all environmental factors."""
        plan = state.plan
        path = plan.full_dataset_path
        if not path:
            raise ValueError("data_mode='upload' 但未指定 full_dataset_path，请在 config.yaml 或交互模式中提供")

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"找不到数据集文件: {path}")

        suffix = file_path.suffix.lower()
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)

        required = {"lon", "lat", "is_presence"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"上传的数据集缺少必要列: {missing}\n需要包含: lon, lat, is_presence, 以及所有环境因子列")

        # Detect which factors are present as columns
        auto_factors = [c for c in df.columns if c not in {"lon", "lat", "is_presence", "species_name", "year", "month", "date", "source", "occurrence_id"}]
        if not auto_factors:
            raise ValueError("数据集中未检测到环境因子列（除 lon/lat/is_presence 外的数值列）")

        if not plan.factors:
            plan.factors = auto_factors
            state.log(f"自动检测环境因子: {', '.join(auto_factors)}")
        else:
            missing_factors = [f for f in plan.factors if f not in df.columns]
            if missing_factors:
                raise ValueError(f"配置的环境因子在数据集中不存在: {missing_factors}\n数据集可用列: {list(df.columns)}")

        # Ensure is_presence is 0/1
        df["is_presence"] = df["is_presence"].astype(int)

        # Fill in missing metadata columns
        if "year" not in df.columns:
            df["year"] = pd.NA
        if "month" not in df.columns:
            df["month"] = pd.NA
        if "species_name" not in df.columns:
            df["species_name"] = plan.species_name

        state.points_df = df
        state.dataset_df = df.copy()
        state.metrics["presence_source"] = {
            "mode": "upload",
            "path": path,
            "rows": int(len(df)),
            "presence_count": int(df["is_presence"].sum()),
            "background_count": int((df["is_presence"] == 0).sum()),
        }
        state.metrics["feature_source"] = {f: "user_upload" for f in plan.factors}

        # Save artifacts
        df.to_csv(state.run_dir / "points_with_labels.csv", index=False)
        state.artifacts["points"] = str(state.run_dir / "points_with_labels.csv")
        df.to_csv(state.run_dir / "training_dataset.csv", index=False)
        state.artifacts["dataset"] = str(state.run_dir / "training_dataset.csv")

        state.log(f"已加载用户数据: {len(df)} 行, 存在点 {state.metrics['presence_source']['presence_count']} 个, 环境因子 {len(plan.factors)} 个")

    def _model_feature_columns(self, plan: PlanConfig, df: pd.DataFrame) -> List[str]:
        feature_cols = []
        for factor in plan.factors:
            if factor in df.columns:
                feature_cols.append(factor)
        return feature_cols

    def _generate_pseudo_absence(
        self,
        presence_df: pd.DataFrame,
        n_bg: int,
        bbox: Tuple[float, float, float, float],
        seed: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        min_lon, min_lat, max_lon, max_lat = bbox

        existing = set(
            zip(
                np.round(presence_df["lon"].to_numpy(), 4),
                np.round(presence_df["lat"].to_numpy(), 4),
            )
        )

        bg_points: List[Tuple[float, float]] = []
        max_tries = n_bg * 30
        tries = 0
        while len(bg_points) < n_bg and tries < max_tries:
            tries += 1
            lon = float(np.round(rng.uniform(min_lon, max_lon), 4))
            lat = float(np.round(rng.uniform(min_lat, max_lat), 4))
            if (lon, lat) not in existing:
                bg_points.append((lon, lat))

        if len(bg_points) < n_bg:
            raise RuntimeError("背景点生成不足，请扩大 bbox 或减少比例。")

        bg_df = pd.DataFrame(bg_points, columns=["lon", "lat"])
        bg_df["is_presence"] = 0
        return bg_df

    def _precheck_factors(self, state: PipelineState) -> None:
        plan = state.plan
        if not plan.use_gee or not plan.enable_gee_precheck:
            state.metrics["gee_precheck"] = {"status": "skipped"}
            return

        precheck: Dict[str, Any] = {
            "status": "ok",
            "available": [],
            "unavailable": {},
        }

        if not self._check_gee_ready(state):
            precheck["status"] = "gee_not_ready"
            precheck["unavailable"] = {factor: "gee_auth_failed" for factor in plan.factors}
            state.metrics["gee_precheck"] = precheck
            if plan.strict_gee:
                raise RuntimeError("严格 GEE 模式下认证失败，流程终止。")
            return

        registry = self._load_gee_registry()
        for factor in plan.factors:
            config = registry.get(factor)
            if not config:
                precheck["unavailable"][factor] = "not_in_registry"
                continue
            try:
                if self._gee_factor_has_data(config, plan.start_date, plan.end_date):
                    precheck["available"].append(factor)
                else:
                    precheck["unavailable"][factor] = "no_data_in_date_range"
            except Exception as exc:
                precheck["unavailable"][factor] = f"check_failed: {exc}"

        if precheck["unavailable"]:
            state.log(f"GEE 预检提示: 不可用变量 {list(precheck['unavailable'].keys())}")

        missing_required = [f for f in plan.required_factors if f in precheck["unavailable"]]
        state.metrics["gee_precheck"] = precheck
        if missing_required:
            raise RuntimeError(f"必需变量预检不可用: {missing_required}")

        if plan.strict_gee and precheck["unavailable"]:
            raise RuntimeError("严格 GEE 模式下存在不可用变量，流程终止。")

    def _gee_factor_has_data(self, config: Dict[str, Any], start_date: str, end_date: str) -> bool:
        import ee  # type: ignore

        if config.get("is_static", False):
            return True
        size = ee.ImageCollection(config["id"]).filterDate(start_date, end_date).select(config["band"]).size().getInfo()
        return int(size) > 0

    def _build_dataset(self, state: PipelineState) -> None:
        if state.points_df is None:
            raise ValueError("points_df is empty")

        df = state.points_df.copy()
        data_mode = str(state.plan.data_mode or "gbif_obis").strip().lower()

        source_map: Dict[str, str] = {}
        gee_ok = state.plan.use_gee and self._check_gee_ready(state)

        if gee_ok:
            source_map.update(self._extract_gee_features(state, df))

        # For any factor missing from GEE (or failed), decide based on data_mode
        for factor in state.plan.factors:
            if factor not in df.columns:
                if data_mode == "gee_extract":
                    # gee_extract mode: GEE must provide the data
                    raise RuntimeError(
                        f"环境因子 '{factor}' 无法从 GEE 提取。\n"
                        f"请检查:\n"
                        f"  1. GEE 认证是否正常\n"
                        f"  2. 因子名称是否在 datasets.json 中注册\n"
                        f"  3. 所选时间范围内是否有可用影像\n"
                        f"或改用 data_mode='gbif_obis' 以允许回退到合成特征。"
                    )
                # gbif_obis mode: auto-fallback to synthetic with warning
                df[factor] = self._feature_formula(df["lon"].to_numpy(), df["lat"].to_numpy(), factor)
                source_map[factor] = source_map.get(factor, "synthetic_fallback")
                state.log(f"⚠️  '{factor}' 使用合成特征（非真实遥感数据）— 模型评估可能不准确")

        # In gee_extract mode, all factors must come from GEE
        if data_mode == "gee_extract":
            non_gee = [f for f, src in source_map.items() if src not in {"gee_live", "gee_cache"}]
            if non_gee:
                raise RuntimeError(
                    f"gee_extract 模式下以下因子非 GEE 来源: {non_gee}\n"
                    f"来源映射: {source_map}\n"
                    f"请确保 GEE 认证正常且因子名称正确。"
                )

        if state.plan.use_gee and state.plan.strict_gee:
            non_gee = [f for f, src in source_map.items() if src not in {"gee_live", "gee_cache"}]
            if non_gee:
                raise RuntimeError(f"严格 GEE 模式失败，以下变量非 GEE 来源: {non_gee}")

        for factor in state.plan.factors:
            df[factor] = pd.to_numeric(df[factor], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        state.dataset_df = df
        state.metrics["feature_source"] = source_map
        df.to_csv(state.run_dir / "training_dataset.csv", index=False)
        state.artifacts["dataset"] = str(state.run_dir / "training_dataset.csv")

    def _check_gee_ready(self, state: PipelineState) -> bool:
        if self._gee_ready_cache is not None:
            return self._gee_ready_cache

        cred_path = Path.home() / ".config" / "earthengine" / "credentials"
        service_account_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

        if not cred_path.exists() and not service_account_key:
            msg = (
                "\n╔══════════════════════════════════════════════════════════════╗\n"
                "║  ⚠️  Google Earth Engine 未认证                            ║\n"
                "╠══════════════════════════════════════════════════════════════╣\n"
                "║  请选择以下方式之一完成认证:                                 ║\n"
                "║                                                            ║\n"
                "║  方式1 (推荐): 命令行认证                                   ║\n"
                "║    python -c \"import ee; ee.Authenticate(); ee.Initialize()\" ║\n"
                "║                                                            ║\n"
                "║  方式2: 服务账号密钥                                        ║\n"
                "║    设置环境变量:                                            ║\n"
                "║    GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json         ║\n"
                "║                                                            ║\n"
                "║  详细文档: https://developers.google.com/earth-engine/      ║\n"
                "║            guides/auth                                     ║\n"
                "╚══════════════════════════════════════════════════════════════╝"
            )
            state.log(msg)
            self._gee_ready_cache = False
            return False

        try:
            import ee  # type: ignore
        except Exception:
            state.log("GEE 未安装: pip install earthengine-api")
            self._gee_ready_cache = False
            return False

        try:
            if hasattr(ee, "data") and hasattr(ee.data, "setDeadline"):
                ee.data.setDeadline(int(state.plan.gee_auth_timeout_ms))
            ee.Initialize()
            state.log("✅ GEE 认证成功 — 将从遥感数据集中提取真实环境变量")
            self._gee_ready_cache = True
            return True
        except Exception as exc:
            state.log(
                f"\n╔══════════════════════════════════════════════════════════════╗\n"
                f"║  ❌ GEE 认证失败                                            ║\n"
                f"╠══════════════════════════════════════════════════════════════╣\n"
                f"║  错误: {str(exc)[:60]:<60}║\n"
                f"╠══════════════════════════════════════════════════════════════╣\n"
                f"║  排查步骤:                                                   ║\n"
                f"║  1. 确认网络可访问 googleapis.com                            ║\n"
                f"║  2. 重新运行: python -c \"import ee; ee.Authenticate()\"       ║\n"
                f"║  3. 检查 GOOGLE_APPLICATION_CREDENTIALS 是否正确             ║\n"
                f"╚══════════════════════════════════════════════════════════════╝"
            )
            self._gee_ready_cache = False
            return False

    def _extract_gee_features(self, state: PipelineState, df: pd.DataFrame) -> Dict[str, str]:
        import ee  # type: ignore

        registry = self._load_gee_registry()
        cache_dir = Path(state.plan.output_dir) / ".gee_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        source_map: Dict[str, str] = {}
        point_sig = self._points_signature(df)

        for factor in state.plan.factors:
            config = registry.get(factor)
            if not config:
                state.log(f"GEE fallback: datasets.json 未找到因子 {factor}，改用内置特征。")
                source_map[factor] = "synthetic_fallback"
                continue

            cache_key = hashlib.md5(
                f"{factor}|{state.plan.start_date}|{state.plan.end_date}|{point_sig}|{config.get('id')}|{config.get('band')}".encode("utf-8")
            ).hexdigest()[:16]
            cache_path = cache_dir / f"{factor}_{cache_key}.csv"

            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                if len(cached) == len(df) and "value" in cached.columns:
                    df[factor] = cached["value"].to_numpy()
                    source_map[factor] = "gee_cache"
                    continue

            try:
                values = self._sample_gee_factor(ee, df, config, state.plan.start_date, state.plan.end_date)
                df[factor] = values
                pd.DataFrame({"value": values}).to_csv(cache_path, index=False)
                source_map[factor] = "gee_live"
            except Exception as exc:
                state.log(f"GEE fallback: 因子 {factor} 拉取失败({exc})，改用内置特征。")
                df[factor] = self._feature_formula(df["lon"].to_numpy(), df["lat"].to_numpy(), factor)
                source_map[factor] = "synthetic_fallback"

        return source_map

    def _sample_gee_factor(
        self,
        ee: object,
        df: pd.DataFrame,
        config: Dict,
        start_date: str,
        end_date: str,
    ) -> np.ndarray:
        points = [
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"id": idx})
            for idx, (lon, lat) in enumerate(zip(df["lon"], df["lat"]))
        ]
        fc = ee.FeatureCollection(points)

        if config.get("is_static", False):
            image = ee.Image(config["id"]).select(config["band"])
        else:
            collection = ee.ImageCollection(config["id"]).filterDate(start_date, end_date).select(config["band"])
            if collection.size().getInfo() == 0:
                raise RuntimeError("所选时段无可用影像")
            image = collection.mean()

        if "scale_factor" in config:
            image = image.multiply(config["scale_factor"])
        if "offset" in config:
            image = image.add(config["offset"])

        sampled = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=int(config.get("scale", 5000)),
        )
        features = sampled.getInfo()["features"]

        values: List[Optional[float]] = [None] * len(df)
        for feature in features:
            props = feature.get("properties", {})
            idx = int(props.get("id", -1))
            if idx < 0 or idx >= len(values):
                continue
            raw = props.get("first", props.get(config["band"]))
            values[idx] = None if raw is None else float(raw)

        return np.array(values, dtype=float)

    def _load_gee_registry(self) -> Dict:
        registry_path = Path(__file__).resolve().parents[1] / "gee_data_fetcher" / "tools" / "references" / "datasets.json"
        with registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _points_signature(self, df: pd.DataFrame) -> str:
        rounded = df[["lon", "lat"]].round(5)
        payload = "|".join(f"{x:.5f},{y:.5f}" for x, y in zip(rounded["lon"], rounded["lat"]))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    def _feature_formula(self, lon: np.ndarray, lat: np.ndarray, factor: str) -> np.ndarray:
        lon_r = np.deg2rad(lon)
        lat_r = np.deg2rad(lat)
        noise = np.random.normal(0, 0.15, size=len(lon))

        rules = {
            "sst": 24 + 4 * np.sin(lat_r) - 1.5 * np.cos(lon_r) + noise,
            "chl_a": 1.2 + 0.7 * np.cos(lat_r * 2) + 0.3 * np.sin(lon_r * 1.3) + noise,
            "salinity": 33 + 1.2 * np.sin(lon_r * 0.8) - 0.8 * np.sin(lat_r * 0.6) + noise,
            "bathymetry": -1200 + 500 * np.sin(lon_r * 1.4) * np.cos(lat_r * 1.1) + noise * 20,
            "current_u": 0.2 * np.sin(lon_r * 1.1) + noise / 10,
            "current_v": 0.2 * np.cos(lat_r * 1.1) + noise / 10,
            "ssh": 0.3 * np.sin(lon_r + lat_r) + noise / 10,
        }
        if factor in rules:
            return rules[factor]
        return np.sin(lon_r * 0.7) + np.cos(lat_r * 0.9) + noise

    def _split_data(self, state: PipelineState) -> None:
        if state.dataset_df is None:
            raise ValueError("dataset_df is empty")
        plan = state.plan

        split_mode = str(plan.split_mode).strip().lower()
        if split_mode not in {
            "random_holdout",
            "random_kfold",
            "spatial_kfold",
            "spatial_block_kfold",
            "env_spatial_block_kfold",
        }:
            raise ValueError(f"不支持的切分策略: {plan.split_mode}")
        if plan.n_splits < 2:
            raise ValueError("n_splits 必须 >= 2")

        train_df, test_df = train_test_split(
            state.dataset_df,
            test_size=plan.test_size,
            random_state=plan.random_seed,
            stratify=state.dataset_df["is_presence"],
        )

        state.train_df = train_df.reset_index(drop=True)
        state.test_df = test_df.reset_index(drop=True)

        state.train_df.to_csv(state.run_dir / "train_split.csv", index=False)
        state.test_df.to_csv(state.run_dir / "test_split.csv", index=False)
        state.artifacts["train_split"] = str(state.run_dir / "train_split.csv")
        state.artifacts["test_split"] = str(state.run_dir / "test_split.csv")

        split_meta: Dict[str, Any] = {
            "mode": split_mode,
            "test_size": float(plan.test_size),
            "n_splits": int(plan.n_splits),
        }

        if split_mode == "random_kfold":
            kf = StratifiedKFold(n_splits=plan.n_splits, shuffle=True, random_state=plan.random_seed)
            y_all = state.dataset_df["is_presence"].astype(int).to_numpy()
            assignment = np.full(len(state.dataset_df), -1, dtype=int)
            fold_sizes = []
            for fold, (_, val_idx) in enumerate(kf.split(np.zeros(len(y_all)), y_all), start=1):
                assignment[val_idx] = fold
                fold_sizes.append(int(len(val_idx)))
            fold_df = pd.DataFrame({
                "row_id": np.arange(len(state.dataset_df)),
                "cv_fold": assignment,
            })
            fold_path = state.run_dir / "cv_fold_assignments.csv"
            fold_df.to_csv(fold_path, index=False)
            state.artifacts["cv_assignments"] = str(fold_path)
            split_meta["fold_sizes"] = fold_sizes

        if split_mode == "spatial_kfold":
            if plan.spatial_clusters < plan.n_splits:
                raise ValueError("spatial_clusters 必须 >= n_splits")
            coords = state.dataset_df[["lon", "lat"]].to_numpy()
            kmeans = KMeans(n_clusters=plan.spatial_clusters, random_state=plan.random_seed, n_init=10)
            clusters = kmeans.fit_predict(coords)
            gkf = GroupKFold(n_splits=plan.n_splits)
            assignment = np.full(len(state.dataset_df), -1, dtype=int)
            y_all = state.dataset_df["is_presence"].astype(int).to_numpy()
            fold_sizes = []
            for fold, (_, val_idx) in enumerate(gkf.split(coords, y_all, groups=clusters), start=1):
                assignment[val_idx] = fold
                fold_sizes.append(int(len(val_idx)))
            fold_df = pd.DataFrame({
                "row_id": np.arange(len(state.dataset_df)),
                "cluster": clusters,
                "cv_fold": assignment,
            })
            fold_path = state.run_dir / "cv_fold_assignments.csv"
            fold_df.to_csv(fold_path, index=False)
            state.artifacts["cv_assignments"] = str(fold_path)
            split_meta["spatial_clusters"] = int(plan.spatial_clusters)
            split_meta["fold_sizes"] = fold_sizes

        if split_mode == "spatial_block_kfold":
            groups, x_edges, y_edges = self._spatial_block_groups(
                state.dataset_df,
                bins_lon=plan.spatial_block_bins_lon,
                bins_lat=plan.spatial_block_bins_lat,
            )
            unique_groups = np.unique(groups)
            if len(unique_groups) < plan.n_splits:
                raise ValueError("空间分块后有效组数不足，无法进行所需折数的 GroupKFold")

            gkf = GroupKFold(n_splits=plan.n_splits)
            assignment = np.full(len(state.dataset_df), -1, dtype=int)
            y_all = state.dataset_df["is_presence"].astype(int).to_numpy()
            fold_sizes = []
            for fold, (_, val_idx) in enumerate(
                gkf.split(state.dataset_df[["lon", "lat"]].to_numpy(), y_all, groups=groups),
                start=1,
            ):
                assignment[val_idx] = fold
                fold_sizes.append(int(len(val_idx)))

            fold_df = pd.DataFrame(
                {
                    "row_id": np.arange(len(state.dataset_df)),
                    "block_group": groups,
                    "cv_fold": assignment,
                }
            )
            fold_path = state.run_dir / "cv_fold_assignments.csv"
            fold_df.to_csv(fold_path, index=False)
            state.artifacts["cv_assignments"] = str(fold_path)
            split_meta["spatial_block_bins_lon"] = int(plan.spatial_block_bins_lon)
            split_meta["spatial_block_bins_lat"] = int(plan.spatial_block_bins_lat)
            split_meta["spatial_block_edges_lon"] = [float(v) for v in x_edges]
            split_meta["spatial_block_edges_lat"] = [float(v) for v in y_edges]
            split_meta["fold_sizes"] = fold_sizes

        if split_mode == "env_spatial_block_kfold":
            groups, x_edges, y_edges, env_edges, env_pc1 = self._env_spatial_block_groups(
                state.dataset_df,
                bins_lon=plan.spatial_block_bins_lon,
                bins_lat=plan.spatial_block_bins_lat,
                env_bins=plan.env_strata_bins,
            )
            unique_groups = np.unique(groups)
            if len(unique_groups) < plan.n_splits:
                raise ValueError("混合分层+分块后有效组数不足，无法进行所需折数")

            gkf = GroupKFold(n_splits=plan.n_splits)
            assignment = np.full(len(state.dataset_df), -1, dtype=int)
            y_all = state.dataset_df["is_presence"].astype(int).to_numpy()
            fold_sizes = []
            for fold, (_, val_idx) in enumerate(
                gkf.split(state.dataset_df[["lon", "lat"]].to_numpy(), y_all, groups=groups),
                start=1,
            ):
                assignment[val_idx] = fold
                fold_sizes.append(int(len(val_idx)))

            fold_df = pd.DataFrame(
                {
                    "row_id": np.arange(len(state.dataset_df)),
                    "env_pc1": env_pc1,
                    "block_group": groups,
                    "cv_fold": assignment,
                }
            )
            fold_path = state.run_dir / "cv_fold_assignments.csv"
            fold_df.to_csv(fold_path, index=False)
            state.artifacts["cv_assignments"] = str(fold_path)

            split_meta["spatial_block_bins_lon"] = int(plan.spatial_block_bins_lon)
            split_meta["spatial_block_bins_lat"] = int(plan.spatial_block_bins_lat)
            split_meta["env_strata_bins"] = int(plan.env_strata_bins)
            split_meta["spatial_block_edges_lon"] = [float(v) for v in x_edges]
            split_meta["spatial_block_edges_lat"] = [float(v) for v in y_edges]
            split_meta["env_edges_pc1"] = [float(v) for v in env_edges]
            split_meta["fold_sizes"] = fold_sizes

        state.metrics["split_strategy"] = split_meta

    def _train_models(self, state: PipelineState) -> None:
        if state.train_df is None or state.test_df is None:
            raise ValueError("train/test data is empty")

        x_cols = self._model_feature_columns(state.plan, state.train_df)
        if not x_cols:
            raise ValueError("没有可用于训练的环境特征列")

        x_train = state.train_df[x_cols]
        y_train = state.train_df["is_presence"].astype(int)
        x_test = state.test_df[x_cols]
        y_test = state.test_df["is_presence"].astype(int)

        candidates: Dict[str, Pipeline] = {}
        if "rf" in state.plan.algorithms:
            candidates["rf"] = Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            random_state=state.plan.random_seed,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    )
                ]
            )

        if "xgb" in state.plan.algorithms:
            from xgboost import XGBClassifier
            candidates["xgb"] = Pipeline(
                [
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=state.plan.random_seed,
                            eval_metric="logloss",
                            verbosity=0,
                        ),
                    )
                ]
            )

        if "lgbm" in state.plan.algorithms:
            from lightgbm import LGBMClassifier
            candidates["lgbm"] = Pipeline(
                [
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=300,
                            max_depth=-1,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=state.plan.random_seed,
                            class_weight="balanced",
                            verbose=-1,
                        ),
                    )
                ]
            )

        if "logreg" in state.plan.algorithms:
            candidates["logreg"] = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2500,
                            random_state=state.plan.random_seed,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

        if not candidates:
            raise ValueError("未配置可用算法")

        best_auc = -1.0
        best_name = None
        best_model = None

        model_scores = {}
        split_mode = str(state.plan.split_mode).strip().lower()
        if split_mode == "random_holdout":
            for name, model in candidates.items():
                model.fit(x_train, y_train)
                prob = model.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, prob))
                model_scores[name] = {"holdout_auc": auc}
                if auc > best_auc:
                    best_auc = auc
                    best_name = name
                    best_model = model
        else:
            cv_scores = self._compute_cv_scores(state, candidates)
            for name, stats in cv_scores.items():
                model = candidates[name]
                model.fit(x_train, y_train)
                prob = model.predict_proba(x_test)[:, 1]
                holdout_auc = float(roc_auc_score(y_test, prob))
                model_scores[name] = {
                    "cv_mean_auc": float(stats["mean_auc"]),
                    "cv_std_auc": float(stats["std_auc"]),
                    "cv_fold_aucs": [float(v) for v in stats["fold_aucs"]],
                    "holdout_auc": holdout_auc,
                }
                if float(stats["mean_auc"]) > best_auc:
                    best_auc = float(stats["mean_auc"])
                    best_name = name
                    best_model = model

        if best_name is None or best_model is None:
            raise RuntimeError("模型训练失败，未选出最佳模型")

        state.best_model_name = best_name
        state.best_model = best_model
        state.feature_columns = x_cols
        state.metrics["model_selection"] = model_scores

        model_path = state.run_dir / "best_model.joblib"
        joblib.dump(best_model, model_path)
        state.artifacts["model"] = str(model_path)

    def _compute_cv_scores(self, state: PipelineState, candidates: Dict[str, Pipeline]) -> Dict[str, Dict[str, Any]]:
        if state.dataset_df is None:
            raise ValueError("dataset_df is empty")

        df = state.dataset_df
        x_cols = self._model_feature_columns(state.plan, df)
        x_all = df[x_cols].to_numpy()
        y_all = df["is_presence"].astype(int).to_numpy()

        split_mode = str(state.plan.split_mode).strip().lower()
        splits = []
        if split_mode == "random_kfold":
            kf = StratifiedKFold(
                n_splits=state.plan.n_splits,
                shuffle=True,
                random_state=state.plan.random_seed,
            )
            splits = list(kf.split(x_all, y_all))
        elif split_mode == "spatial_kfold":
            coords = df[["lon", "lat"]].to_numpy()
            kmeans = KMeans(
                n_clusters=state.plan.spatial_clusters,
                random_state=state.plan.random_seed,
                n_init=10,
            )
            clusters = kmeans.fit_predict(coords)
            gkf = GroupKFold(n_splits=state.plan.n_splits)
            splits = list(gkf.split(x_all, y_all, groups=clusters))
        elif split_mode == "spatial_block_kfold":
            groups, _, _ = self._spatial_block_groups(
                df,
                bins_lon=state.plan.spatial_block_bins_lon,
                bins_lat=state.plan.spatial_block_bins_lat,
            )
            if len(np.unique(groups)) < state.plan.n_splits:
                raise ValueError("空间分块后有效组数不足，无法进行所需折数的 GroupKFold")
            gkf = GroupKFold(n_splits=state.plan.n_splits)
            splits = list(gkf.split(x_all, y_all, groups=groups))
        elif split_mode == "env_spatial_block_kfold":
            groups, _, _, _, _ = self._env_spatial_block_groups(
                df,
                bins_lon=state.plan.spatial_block_bins_lon,
                bins_lat=state.plan.spatial_block_bins_lat,
                env_bins=state.plan.env_strata_bins,
            )
            if len(np.unique(groups)) < state.plan.n_splits:
                raise ValueError("混合分层+分块后有效组数不足，无法进行所需折数的 GroupKFold")
            gkf = GroupKFold(n_splits=state.plan.n_splits)
            splits = list(gkf.split(x_all, y_all, groups=groups))

        out: Dict[str, Dict[str, Any]] = {}
        for name, model in candidates.items():
            fold_aucs: List[float] = []
            for train_idx, val_idx in splits:
                model.fit(x_all[train_idx], y_all[train_idx])
                prob = model.predict_proba(x_all[val_idx])[:, 1]
                fold_aucs.append(float(roc_auc_score(y_all[val_idx], prob)))
            out[name] = {
                "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else 0.0,
                "std_auc": float(np.std(fold_aucs)) if fold_aucs else 0.0,
                "fold_aucs": fold_aucs,
            }

        return out

    def _spatial_block_groups(self, df: pd.DataFrame, bins_lon: int, bins_lat: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bins_lon < 2 or bins_lat < 2:
            raise ValueError("空间分块网格数必须 >= 2")

        lon = df["lon"].to_numpy()
        lat = df["lat"].to_numpy()

        x_edges = np.linspace(float(np.min(lon)), float(np.max(lon)), bins_lon + 1)
        y_edges = np.linspace(float(np.min(lat)), float(np.max(lat)), bins_lat + 1)

        x_bin = np.clip(np.digitize(lon, x_edges[1:-1], right=False), 0, bins_lon - 1)
        y_bin = np.clip(np.digitize(lat, y_edges[1:-1], right=False), 0, bins_lat - 1)
        groups = y_bin * bins_lon + x_bin
        return groups.astype(int), x_edges, y_edges

    def _env_spatial_block_groups(
        self,
        df: pd.DataFrame,
        bins_lon: int,
        bins_lat: int,
        env_bins: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        groups_spatial, x_edges, y_edges = self._spatial_block_groups(df, bins_lon=bins_lon, bins_lat=bins_lat)
        if env_bins < 2:
            raise ValueError("env_strata_bins 必须 >= 2")

        env_cols = [c for c in df.columns if c not in {"lon", "lat", "is_presence"}]
        if not env_cols:
            raise ValueError("无法构建环境分层：缺少环境变量列")

        env_matrix = df[env_cols].to_numpy(dtype=float)
        scaler = StandardScaler()
        env_scaled = scaler.fit_transform(env_matrix)
        pc1 = PCA(n_components=1, random_state=42).fit_transform(env_scaled).reshape(-1)

        env_edges = np.quantile(pc1, q=np.linspace(0.0, 1.0, env_bins + 1))
        env_edges = np.unique(env_edges)
        if len(env_edges) < 3:
            # Degenerated case: very low variance in env features.
            env_edges = np.linspace(float(np.min(pc1)), float(np.max(pc1)) + 1e-9, env_bins + 1)

        env_bin = np.clip(np.digitize(pc1, env_edges[1:-1], right=False), 0, len(env_edges) - 2)
        groups = groups_spatial.astype(int) * max(env_bins, len(env_edges) - 1) + env_bin.astype(int)
        return groups.astype(int), x_edges, y_edges, env_edges, pc1

    def _evaluate(self, state: PipelineState) -> None:
        if state.best_model is None or state.test_df is None:
            raise ValueError("缺少模型或测试集")

        y_true = state.test_df["is_presence"].astype(int).to_numpy()
        x_test = state.test_df[state.feature_columns]
        prob = state.best_model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        tss = sensitivity + specificity - 1

        eval_metrics = {
            "best_model": state.best_model_name,
            "roc_auc": float(roc_auc_score(y_true, prob)),
            "pr_auc": float(average_precision_score(y_true, prob)),
            "f1": float(f1_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tss": float(tss),
            "threshold": 0.5,
        }
        state.metrics["evaluation"] = eval_metrics

        # ROC
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, color="#005f73", lw=2.5, label=f"ROC AUC = {eval_metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        roc_path = state.run_dir / "roc_curve.png"
        fig.tight_layout()
        fig.savefig(roc_path)
        plt.close(fig)
        state.artifacts["roc_curve"] = str(roc_path)

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, pred), display_labels=[0, 1])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        cm_path = state.run_dir / "confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(cm_path)
        plt.close(fig)
        state.artifacts["confusion_matrix"] = str(cm_path)

        # --- Model Interpretability ---
        self._compute_interpretability(state, x_test, y_true)

    def _compute_interpretability(
        self,
        state: PipelineState,
        x_test: pd.DataFrame,
        y_true: np.ndarray,
    ) -> None:
        """Generate variable importance, SHAP, PDP, and response curves."""
        feature_names = list(x_test.columns)
        if not feature_names:
            return

        model = state.best_model
        # Unwrap from Pipeline if needed
        if isinstance(model, Pipeline):
            estimator = model.named_steps.get("model", model)
        else:
            estimator = model

        # --- 1. Permutation Importance ---
        try:
            from sklearn.inspection import permutation_importance
            perm_result = permutation_importance(
                model, x_test, y_true, n_repeats=10,
                random_state=state.plan.random_seed, scoring="roc_auc",
            )
            perm_df = pd.DataFrame({
                "factor": feature_names,
                "importance_mean": perm_result.importances_mean,
                "importance_std": perm_result.importances_std,
            }).sort_values("importance_mean", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
            ax.barh(perm_df["factor"], perm_df["importance_mean"],
                    xerr=perm_df["importance_std"], color="#0a9396", capsize=3)
            ax.set_xlabel("ROC AUC Decrease")
            ax.set_title("Permutation Importance", fontsize=14, fontweight="bold")
            fig.tight_layout()
            imp_path = state.run_dir / "variable_importance.png"
            fig.savefig(imp_path)
            plt.close(fig)
            state.artifacts["variable_importance"] = str(imp_path)

            # Store top factors
            top_factors = perm_df.tail(5)["factor"].tolist()[::-1]
            state.metrics["evaluation"]["top_factors"] = top_factors
            state.metrics["evaluation"]["permutation_importance"] = {
                row["factor"]: {"mean": float(row["importance_mean"]), "std": float(row["importance_std"])}
                for _, row in perm_df.iterrows()
            }
        except Exception as exc:
            state.log(f"Permutation importance skipped: {exc}")

        # --- 2. SHAP Summary (tree-based models only) ---
        try:
            is_tree = state.best_model_name in {"rf", "xgb", "lgbm"}
            if is_tree:
                import shap
                # Sample to avoid OOM on large test sets
                n_shap = min(200, len(x_test))
                x_sample = x_test.sample(n=n_shap, random_state=42) if len(x_test) > n_shap else x_test

                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(x_sample)
                # shap_values may be a list [neg, pos] for binary classification
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # positive class

                fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
                shap.summary_plot(shap_values, x_sample, feature_names=feature_names,
                                  show=False, max_display=10)
                fig.tight_layout()
                shap_path = state.run_dir / "shap_summary.png"
                fig.savefig(shap_path, bbox_inches="tight")
                plt.close("all")
                state.artifacts["shap_summary"] = str(shap_path)

                # Store mean SHAP values
                shap_means = np.abs(shap_values).mean(axis=0)
                state.metrics["evaluation"]["shap_importance"] = {
                    feature_names[i]: float(shap_means[i])
                    for i in np.argsort(shap_means)[::-1][:10]
                }
        except Exception as exc:
            state.log(f"SHAP analysis skipped: {exc}")

        # --- 3. Partial Dependence Plots (top 3 factors) ---
        top3 = state.metrics["evaluation"].get("top_factors", feature_names[:3])[:3]
        try:
            from sklearn.inspection import PartialDependenceDisplay
            fig, ax = plt.subplots(1, len(top3), figsize=(5 * len(top3), 5), dpi=140)
            if len(top3) == 1:
                ax = [ax]
            PartialDependenceDisplay.from_estimator(
                model, x_test, top3, kind="average",
                ax=ax, line_kw={"color": "#005f73", "lw": 2},
            )
            for i, factor in enumerate(top3):
                ax[i].set_title(f"PDP: {factor}", fontsize=12, fontweight="bold")
                ax[i].set_ylabel("Predicted Suitability")
            fig.tight_layout()
            pdp_path = state.run_dir / "partial_dependence.png"
            fig.savefig(pdp_path)
            plt.close(fig)
            state.artifacts["partial_dependence"] = str(pdp_path)
        except Exception as exc:
            state.log(f"PDP plots skipped: {exc}")

        # --- 4. Response Curves (probability vs each factor, others at median) ---
        try:
            n_factors = len(feature_names)
            n_cols = min(3, n_factors)
            n_rows = int(np.ceil(n_factors / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=140)
            axes_flat = axes.flatten() if n_factors > 1 else [axes]

            x_median = x_test.median().to_frame().T
            for i, factor in enumerate(feature_names):
                ax = axes_flat[i]
                vals = np.linspace(x_test[factor].min(), x_test[factor].max(), 50)
                x_curve = pd.concat([x_median] * len(vals), ignore_index=True)
                x_curve[factor] = vals
                prob_curve = model.predict_proba(x_curve)[:, 1]
                ax.plot(vals, prob_curve, color="#005f73", lw=2)
                ax.fill_between(vals, prob_curve, alpha=0.15, color="#005f73")
                ax.set_xlabel(factor)
                ax.set_ylabel("Suitability")
                ax.set_title(f"Response: {factor}", fontsize=11, fontweight="bold")
                # Mark actual data distribution
                ax.scatter(x_test[factor].sample(min(100, len(x_test))),
                          [0.02] * min(100, len(x_test)),
                          s=3, alpha=0.3, color="#ee9b00")

            # Hide unused subplots
            for j in range(n_factors, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle("Environmental Response Curves", fontsize=15, fontweight="bold", y=1.01)
            fig.tight_layout()
            resp_path = state.run_dir / "response_curves.png"
            fig.savefig(resp_path)
            plt.close(fig)
            state.artifacts["response_curves"] = str(resp_path)
        except Exception as exc:
            state.log(f"Response curves skipped: {exc}")

    def _predict_map(self, state: PipelineState) -> None:
        if state.best_model is None:
            raise ValueError("缺少训练完成的模型")

        min_lon, min_lat, max_lon, max_lat = state.plan.bbox
        n = state.plan.map_resolution

        lon_grid = np.linspace(min_lon, max_lon, n)
        lat_grid = np.linspace(min_lat, max_lat, n)
        xx, yy = np.meshgrid(lon_grid, lat_grid)

        grid_df = pd.DataFrame({"lon": xx.ravel(), "lat": yy.ravel()})

        pred_source: Dict[str, str] = {}
        if state.plan.use_gee and self._check_gee_ready(state):
            pred_source.update(self._extract_gee_features(state, grid_df))

        for factor in state.plan.factors:
            if factor not in grid_df.columns:
                grid_df[factor] = self._feature_formula(grid_df["lon"].to_numpy(), grid_df["lat"].to_numpy(), factor)
                pred_source[factor] = pred_source.get(factor, "synthetic_fallback")

        if state.plan.use_gee and state.plan.strict_gee:
            non_gee = [f for f, src in pred_source.items() if src not in {"gee_live", "gee_cache"}]
            if non_gee:
                raise RuntimeError(f"严格 GEE 模式失败，预测变量非 GEE 来源: {non_gee}")

        state.metrics["prediction_feature_source"] = pred_source

        x_pred = grid_df[state.feature_columns]
        grid_df["suitability"] = state.best_model.predict_proba(x_pred)[:, 1]

        out_csv = state.run_dir / "prediction_grid.csv"
        grid_df.to_csv(out_csv, index=False)
        state.artifacts["prediction_grid"] = str(out_csv)

        z = grid_df["suitability"].to_numpy().reshape(n, n)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        cmap = sns.color_palette("YlGnBu", as_cmap=True)
        heat = ax.imshow(
            z,
            origin="lower",
            cmap=cmap,
            extent=[min_lon, max_lon, min_lat, max_lat],
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        cbar = fig.colorbar(heat, ax=ax)
        cbar.set_label("Habitat Suitability", rotation=90)
        ax.set_title("Predicted Habitat Suitability Map", fontsize=15, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if state.points_df is not None:
            pres = state.points_df[state.points_df["is_presence"] == 1]
            ax.scatter(pres["lon"], pres["lat"], s=9, c="#ffb703", edgecolors="none", alpha=0.45, label="Presence")
            ax.legend(loc="upper right")

        map_path = state.run_dir / "prediction_map.png"
        fig.tight_layout()
        fig.savefig(map_path)
        plt.close(fig)
        state.artifacts["prediction_map"] = str(map_path)

    def _build_report(self, state: PipelineState) -> None:
        report_path = state.run_dir / "evaluation_report.html"
        metrics = state.metrics.get("evaluation", {})
        gee_precheck = state.metrics.get("gee_precheck", {})
        feature_source = state.metrics.get("feature_source", {})
        pred_source = state.metrics.get("prediction_feature_source", {})
        presence_source = state.metrics.get("presence_source", {})
        data_mode = state.plan.data_mode or "gbif_obis"

        # Determine data quality flag
        sources_used = set(feature_source.values()) if feature_source else set()
        if not sources_used:
            data_quality = "unknown"
            data_quality_label = "未知"
        elif sources_used <= {"user_upload"}:
            data_quality = "user"
            data_quality_label = "✅ 用户自有数据"
        elif sources_used <= {"gee_live", "gee_cache"}:
            data_quality = "real"
            data_quality_label = "✅ 真实遥感数据 (GEE)"
        elif sources_used & {"gee_live", "gee_cache"}:
            data_quality = "mixed"
            data_quality_label = "⚠️ 混合来源（部分GEE + 部分合成）"
        else:
            data_quality = "synthetic"
            data_quality_label = "⚠️ 合成模拟数据（非真实遥感）— 模型结果不可用于科研"

        step_status_html = "".join(
            f"<li><strong>{k}</strong>: {v}</li>" for k, v in state.step_status.items()
        ) or "<li>暂无</li>"
        error_html = "".join(
            f"<li>[{e.get('time','')}] {e.get('step','')} (attempt {e.get('attempt','')}): {e.get('error','')}</li>"
            for e in state.error_events[-10:]
        ) or "<li>无错误事件</li>"
        train_source_html = "".join(
            f"<li>{k}: {v}</li>" for k, v in feature_source.items()
        ) or "<li>暂无</li>"
        pred_source_html = "".join(
            f"<li>{k}: {v}</li>" for k, v in pred_source.items()
        ) or "<li>暂无</li>"
        presence_source_html = json.dumps(presence_source, ensure_ascii=False, indent=2) if presence_source else "{}"
        gee_precheck_html = "".join(
            [
                f"<li>status: {gee_precheck.get('status', 'na')}</li>",
                f"<li>available: {', '.join(gee_precheck.get('available', [])) or 'none'}</li>",
                f"<li>unavailable: {', '.join(list((gee_precheck.get('unavailable') or {}).keys())) or 'none'}</li>",
            ]
        )

        html = f"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>SDM 评估报告</title>
  <style>
    :root {{
      --bg: #f4f7f9;
      --card: #ffffff;
      --text: #0f172a;
      --sub: #334155;
      --primary: #005f73;
      --accent: #ee9b00;
      --border: #dbe4ea;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: radial-gradient(circle at 20% 10%, #d9f0ff, var(--bg));
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 28px auto;
      padding: 0 18px 30px;
    }}
    .hero {{
      background: linear-gradient(120deg, #023047, #0a9396);
      color: white;
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(2,48,71,0.06);
    }}
    .k {{ color: var(--sub); font-size: 13px; }}
    .v {{ color: var(--primary); font-size: 22px; font-weight: 700; margin-top: 5px; }}
    .section-title {{ margin: 26px 0 10px; font-size: 20px; }}
    .img-box img {{ width: 100%; border-radius: 10px; border: 1px solid var(--border); }}
    .footer {{ margin-top: 20px; color: var(--sub); font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"hero\">
      <h1 style=\"margin:0\">SDM 生产级评估报告</h1>
      <p style=\"opacity:0.92\">物种: {state.plan.species_name} | 模型: {metrics.get("best_model", "NA")} | 数据模式: {data_mode}</p>
      <p style=\"opacity:0.92; margin-top:6px\">数据质量: {data_quality_label}</p>
    </div>

    <h2 class=\"section-title\">核心指标</h2>
    <div class=\"grid\">
      <div class=\"card\"><div class=\"k\">ROC AUC</div><div class=\"v\">{metrics.get("roc_auc", 0):.3f}</div></div>
      <div class=\"card\"><div class=\"k\">PR AUC</div><div class=\"v\">{metrics.get("pr_auc", 0):.3f}</div></div>
      <div class=\"card\"><div class=\"k\">TSS</div><div class=\"v\">{metrics.get("tss", 0):.3f}</div></div>
      <div class=\"card\"><div class=\"k\">F1</div><div class=\"v\">{metrics.get("f1", 0):.3f}</div></div>
      <div class=\"card\"><div class=\"k\">Precision</div><div class=\"v\">{metrics.get("precision", 0):.3f}</div></div>
      <div class=\"card\"><div class=\"k\">Recall</div><div class=\"v\">{metrics.get("recall", 0):.3f}</div></div>
    </div>

    <h2 class=\"section-title\">模型判别能力</h2>
    <div class=\"img-box\"><img src=\"roc_curve.png\" alt=\"ROC Curve\" /></div>

    <h2 class=\"section-title\">混淆矩阵</h2>
    <div class=\"img-box\"><img src=\"confusion_matrix.png\" alt=\"Confusion Matrix\" /></div>

    <h2 class=\"section-title\">变量重要性 (Permutation)</h2>
    <div class=\"img-box\"><img src=\"variable_importance.png\" alt=\"Variable Importance\" /></div>

    <h2 class=\"section-title\">环境响应曲线</h2>
    <div class=\"img-box\"><img src=\"response_curves.png\" alt=\"Response Curves\" /></div>

    <h2 class=\"section-title\">SHAP 特征贡献</h2>
    <div class=\"img-box\"><img src=\"shap_summary.png\" alt=\"SHAP Summary\" /></div>

    <h2 class=\"section-title\">偏依赖图 (PDP)</h2>
    <div class=\"img-box\"><img src=\"partial_dependence.png\" alt=\"Partial Dependence\" /></div>


    <h2 class=\"section-title\">空间预测图</h2>
    <div class=\"img-box\"><img src=\"prediction_map.png\" alt=\"Prediction Map\" /></div>

        <h2 class="section-title">存在点来源</h2>
        <div class="card"><pre style="margin:0; white-space:pre-wrap; word-break:break-word;">{presence_source_html}</pre></div>

        <h2 class="section-title">数据源与执行状态</h2>
        <div class="grid">
            <div class="card"><div class="k">训练变量来源</div><ul>{train_source_html}</ul></div>
            <div class="card"><div class="k">预测变量来源</div><ul>{pred_source_html}</ul></div>
            <div class="card"><div class="k">步骤状态</div><ul>{step_status_html}</ul></div>
            <div class="card"><div class="k">GEE 变量预检</div><ul>{gee_precheck_html}</ul></div>
        </div>

        <h2 class="section-title">错误与自动纠错日志(最近10条)</h2>
        <div class="card"><ul>{error_html}</ul></div>

    <p class=\"footer\">生成时间: {datetime.now().isoformat(timespec="seconds")}</p>
  </div>
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")
        state.artifacts["html_report"] = str(report_path)

    def _save_metadata(self, state: PipelineState) -> None:
        state.step_status["save_metadata"] = "succeeded"
        out_path = state.run_dir / "run_summary.json"
        state.artifacts["summary"] = str(out_path)

        error_path = state.run_dir / "errors.json"
        error_payload = {
            "step_status": state.step_status,
            "error_events": state.error_events,
        }
        error_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        state.artifacts["errors"] = str(error_path)

        meta = {
            "plan": asdict(state.plan),
            "metrics": state.metrics,
            "artifacts": state.artifacts,
            "step_status": state.step_status,
            "error_events": state.error_events,
            "logs": state.log_messages,
        }
        out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
