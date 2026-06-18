from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OccurrenceDownloadResult:
    dataframe: pd.DataFrame
    source_mode: str
    source_stats: Dict[str, Any]


def _fetch_json(url: str, timeout: int = 30) -> Any:
    request = Request(url, headers={"User-Agent": "SDM-Orchestrator/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in {None, ""}:
            return mapping[key]
    return None


def _to_int_or_none(value: Any) -> Optional[int]:
    if value in {None, "", "NA", "NaN"}:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _to_float_or_none(value: Any) -> Optional[float]:
    if value in {None, "", "NA", "NaN"}:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def _to_iso_date(value: Any) -> Optional[str]:
    if value in {None, "", "NA", "NaN"}:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: Dict[str, str] = {}
    lower_map = {str(col).strip().lower(): col for col in df.columns}

    aliases = {
        "lon": ["lon", "longitude", "decimallongitude", "decimal_longitude", "x"],
        "lat": ["lat", "latitude", "decimallatitude", "decimal_latitude", "y"],
        "year": ["year", "yyyy", "occurrence_year", "date_year"],
        "month": ["month", "mm", "occurrence_month", "date_month"],
        "date": ["date", "eventdate", "occurrencedate", "verbatimeventdate", "datetime"],
        "species_name": ["species_name", "species", "scientificname", "scientific_name"],
        "source": ["source", "dataset_source", "record_source"],
        "occurrence_id": ["occurrence_id", "occurrenceid", "id", "gbifid"],
    }

    for target, names in aliases.items():
        for name in names:
            if name in lower_map:
                renamed[lower_map[name]] = target
                break

    out = df.rename(columns=renamed).copy()
    if "date" in out.columns:
        out["date"] = out["date"].map(_to_iso_date)

    if "year" not in out.columns and "date" in out.columns:
        out["year"] = pd.to_datetime(out["date"], errors="coerce").dt.year
    if "month" not in out.columns and "date" in out.columns:
        out["month"] = pd.to_datetime(out["date"], errors="coerce").dt.month

    if "year" in out.columns:
        out["year"] = out["year"].map(_to_int_or_none)
    if "month" in out.columns:
        out["month"] = out["month"].map(_to_int_or_none)

    if "lon" in out.columns:
        out["lon"] = out["lon"].map(_to_float_or_none)
    if "lat" in out.columns:
        out["lat"] = out["lat"].map(_to_float_or_none)

    if "source" not in out.columns:
        out["source"] = None

    return out


def normalize_presence_dataframe(
    df: pd.DataFrame,
    species_name: str,
    source_label: str,
) -> pd.DataFrame:
    out = _normalize_dataframe_columns(df)

    if "lon" not in out.columns or "lat" not in out.columns:
        raise ValueError("文件必须包含 lon 和 lat 两列")

    if out[["lon", "lat"]].isna().any().any():
        raise ValueError("lon/lat 不能为空")

    if "year" not in out.columns or out["year"].isna().all():
        raise ValueError("文件必须包含 year 或可解析的 date/eventDate")

    if "species_name" not in out.columns:
        out["species_name"] = species_name
    else:
        out["species_name"] = out["species_name"].fillna(species_name)
    out["source"] = out["source"].fillna(source_label)
    out["year"] = out["year"].astype("Int64")
    if "month" not in out.columns:
        out["month"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["month"] = out["month"].astype("Int64")

    if "date" not in out.columns:
        out["date"] = pd.NA

    out["is_presence"] = 1
    out["lon"] = out["lon"].astype(float)
    out["lat"] = out["lat"].astype(float)

    ordered = [
        "species_name",
        "lon",
        "lat",
        "year",
        "month",
        "date",
        "source",
        "occurrence_id",
        "is_presence",
    ]
    front = [col for col in ordered if col in out.columns]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].reset_index(drop=True)


def load_presence_points_file(path: str, species_name: str, source_label: str = "upload") -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到存在点文件: {path}")

    suffix = file_path.suffix.lower()
    if suffix == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        df = pd.read_csv(file_path)
    return normalize_presence_dataframe(df, species_name=species_name, source_label=source_label)


def resolve_species_name(species_name: str, timeout: int = 30) -> str:
    raw = species_name.strip()
    if not raw:
        return raw
    try:
        url = f"https://api.gbif.org/v1/species/match?q={quote_plus(raw)}"
        data = _fetch_json(url, timeout=timeout)
        scientific = data.get("scientificName") or data.get("canonicalName")
        if scientific and int(data.get("confidence", 0) or 0) >= 70:
            return str(scientific)
    except Exception:
        pass
    return raw


def _records_to_frame(records: List[Dict[str, Any]], source_label: str, species_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        lon = _first_present(record, ["decimalLongitude", "lon", "longitude"])
        lat = _first_present(record, ["decimalLatitude", "lat", "latitude"])
        if lon is None or lat is None:
            continue

        date_raw = _first_present(record, ["eventDate", "verbatimEventDate", "date"])
        year = _first_present(record, ["year", "date_year"])
        month = _first_present(record, ["month", "date_month"])
        if (year is None or year == "") and date_raw is not None:
            parsed = pd.to_datetime(date_raw, errors="coerce")
            if not pd.isna(parsed):
                year = parsed.year
                if month in {None, ""}:
                    month = parsed.month

        row = {
            "species_name": _first_present(record, ["scientificName", "species", "species_name"]) or species_name,
            "lon": lon,
            "lat": lat,
            "year": year,
            "month": month,
            "date": _to_iso_date(date_raw),
            "source": source_label,
            "occurrence_id": _first_present(record, ["occurrenceID", "gbifID", "id", "key"]),
            "dataset_key": _first_present(record, ["datasetKey", "dataset_id", "datasetID"]),
            "basis_of_record": _first_present(record, ["basisOfRecord", "basis_of_record"]),
            "country": _first_present(record, ["country", "countryName"]),
            "marine": _first_present(record, ["marine"]),
            "event_type": _first_present(record, ["eventType"]),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["species_name", "lon", "lat", "year", "month", "date", "source", "occurrence_id"])

    return pd.DataFrame(rows)


def _extract_gbif_records(species_name: str, start_date: str, end_date: str, limit: int, timeout: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    offset = 0
    while len(records) < limit:
        page_size = min(300, limit - len(records))
        url = (
            "https://api.gbif.org/v1/occurrence/search?"
            f"scientificName={quote_plus(species_name)}&hasCoordinate=true&limit={page_size}&offset={offset}"
        )
        data = _fetch_json(url, timeout=timeout)
        page = data.get("results", []) if isinstance(data, dict) else []
        if not page:
            break
        records.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return records[:limit]


def _extract_obis_records(species_name: str, start_date: str, end_date: str, limit: int, timeout: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    page = 1
    while len(records) < limit:
        page_size = min(500, limit - len(records))
        url = (
            "https://api.obis.org/v3/occurrence?"
            f"scientificname={quote_plus(species_name)}&size={page_size}&page={page}"
        )
        data = _fetch_json(url, timeout=timeout)
        page_records = data.get("results", []) if isinstance(data, dict) else []
        if not page_records:
            break
        records.extend(page_records)
        if len(page_records) < page_size:
            break
        page += 1
    return records[:limit]


def _filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
    else:
        parsed = pd.Series([pd.NaT] * len(df), index=df.index)

    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")

    mask = pd.Series(True, index=df.index)
    if not pd.isna(start):
        mask &= (parsed.isna() | (parsed >= start))
    if not pd.isna(end):
        mask &= (parsed.isna() | (parsed <= end))

    if "year" in df.columns:
        start_year = int(start.year) if not pd.isna(start) else None
        end_year = int(end.year) if not pd.isna(end) else None
        if start_year is not None:
            mask &= df["year"].isna() | (df["year"].astype("Int64") >= start_year)
        if end_year is not None:
            mask &= df["year"].isna() | (df["year"].astype("Int64") <= end_year)

    return df.loc[mask].reset_index(drop=True)


def _combine_sources(frames: List[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()

    merged = pd.concat(valid, ignore_index=True, sort=False)
    merged = merged.copy()

    merged["lon_round"] = merged["lon"].round(5)
    merged["lat_round"] = merged["lat"].round(5)
    if "year" not in merged.columns:
        merged["year"] = pd.Series([pd.NA] * len(merged), dtype="Int64")
    if "month" not in merged.columns:
        merged["month"] = pd.Series([pd.NA] * len(merged), dtype="Int64")

    merged["year"] = merged["year"].astype("Int64")
    merged["month"] = merged["month"].astype("Int64")
    merged["merge_key"] = (
        merged["species_name"].fillna("")
        + "|"
        + merged["lon_round"].astype(str)
        + "|"
        + merged["lat_round"].astype(str)
        + "|"
        + merged["year"].astype(str)
        + "|"
        + merged["month"].astype(str)
    )

    keep_columns = [col for col in merged.columns if col not in {"lon_round", "lat_round", "merge_key"}]
    rows: List[Dict[str, Any]] = []
    for _, group in merged.groupby("merge_key", dropna=False, sort=False):
        row: Dict[str, Any] = {}
        for column in keep_columns:
            if column == "source":
                sources = sorted({str(v) for v in group[column].dropna().tolist() if str(v).strip()})
                row[column] = "|".join(sources)
                continue
            non_null = group[column].dropna()
            row[column] = non_null.iloc[0] if not non_null.empty else pd.NA
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def download_occurrence_points(
    species_name: str,
    source_mode: str,
    start_date: str,
    end_date: str,
    limit: int = 1200,
    timeout: int = 30,
) -> OccurrenceDownloadResult:
    resolved_name = resolve_species_name(species_name, timeout=timeout)
    mode = str(source_mode or "gbif_obis").strip().lower()
    if mode in {"upload", "", "auto"}:
        mode = "gbif_obis"

    frames: List[pd.DataFrame] = []
    stats: Dict[str, Any] = {
        "requested_species_name": species_name,
        "resolved_species_name": resolved_name,
        "source_mode": mode,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
        "sources": {},
    }

    if mode in {"gbif", "gbif_obis", "auto"}:
        gbif_records = _extract_gbif_records(resolved_name, start_date, end_date, limit, timeout)
        gbif_df = normalize_presence_dataframe(
            _records_to_frame(gbif_records, "gbif", resolved_name),
            species_name=resolved_name,
            source_label="gbif",
        )
        gbif_df = _filter_date_range(gbif_df, start_date, end_date)
        frames.append(gbif_df)
        stats["sources"]["gbif"] = {
            "records": int(len(gbif_df)),
        }

    if mode in {"obis", "gbif_obis", "auto"}:
        obis_records = _extract_obis_records(resolved_name, start_date, end_date, limit, timeout)
        obis_df = normalize_presence_dataframe(
            _records_to_frame(obis_records, "obis", resolved_name),
            species_name=resolved_name,
            source_label="obis",
        )
        obis_df = _filter_date_range(obis_df, start_date, end_date)
        frames.append(obis_df)
        stats["sources"]["obis"] = {
            "records": int(len(obis_df)),
        }

    merged = _combine_sources(frames)
    if merged.empty:
        raise RuntimeError("GBIF/OBIS 未下载到有效存在点")

    merged = normalize_presence_dataframe(merged, species_name=resolved_name, source_label=mode)
    merged = merged.drop_duplicates(subset=["species_name", "lon", "lat", "year", "month", "source"], keep="first").reset_index(drop=True)
    stats["total_records"] = int(len(merged))
    stats["year_range"] = [int(merged["year"].min()), int(merged["year"].max())]
    if "month" in merged.columns and merged["month"].notna().any():
        stats["month_range"] = [int(merged["month"].dropna().min()), int(merged["month"].dropna().max())]
    stats["sources"]["merged"] = {"records": int(len(merged))}

    return OccurrenceDownloadResult(dataframe=merged, source_mode=mode, source_stats=stats)