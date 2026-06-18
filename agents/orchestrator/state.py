from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PlanConfig:
    species_name: str = "target_species"
    # --- data mode ---
    data_mode: str = "gbif_obis"  # "upload" | "gbif_obis" | "gee_extract"
    full_dataset_path: str = ""   # used when data_mode == "upload"

    presence_points_path: str = ""
    presence_source_mode: str = "gbif_obis"
    occurrence_download_limit: int = 1200
    output_dir: str = "./workspace"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    bbox: Tuple[float, float, float, float] = (110.0, 20.0, 125.0, 35.0)
    factors: List[str] = field(
        default_factory=lambda: [
            "sst",
            "chl_a",
            "salinity",
            "bathymetry",
        ]
    )
    algorithms: List[str] = field(default_factory=lambda: ["rf", "logreg"])
    pseudo_absence_ratio: float = 1.0
    test_size: float = 0.2
    split_mode: str = "random_holdout"
    n_splits: int = 5
    spatial_clusters: int = 10
    spatial_block_bins_lon: int = 4
    spatial_block_bins_lat: int = 4
    env_strata_bins: int = 4
    random_seed: int = 42
    map_resolution: int = 140
    use_gee: bool = False
    strict_gee: bool = False
    enable_gee_precheck: bool = True
    gee_auth_timeout_ms: int = 8000
    max_retries: int = 2
    auto_factor_selection: bool = True
    required_factors: List[str] = field(default_factory=list)


@dataclass
class PipelineState:
    plan: PlanConfig
    run_dir: Path
    log_messages: List[str] = field(default_factory=list)

    points_df: Optional[pd.DataFrame] = None
    dataset_df: Optional[pd.DataFrame] = None

    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None

    best_model_name: Optional[str] = None
    best_model: Any = None
    feature_columns: List[str] = field(default_factory=list)

    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    step_status: Dict[str, str] = field(default_factory=dict)
    error_events: List[Dict[str, str]] = field(default_factory=list)
    progress_callback: Optional[Callable[[str], None]] = None

    def log(self, message: str) -> None:
        self.log_messages.append(message)
        if self.progress_callback:
            self.progress_callback(message)
