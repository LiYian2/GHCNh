#!/usr/bin/env python3
"""
GHCNh CONUS hourly precipitation pipeline.

This script downloads NOAA GHCNh per-station/year Parquet files, applies
source-aware precipitation QC, and writes one consolidated Parquet file.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


BASE_URL = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
DEFAULT_YEARS = (2020, 2021, 2022)
DEFAULT_SAMPLE_STATION = "USW00094728"
DEFAULT_SAMPLE_YEAR = 2020

CONUS = {
    "lat_min": 24.0,
    "lat_max": 50.0,
    "lon_min": -125.0,
    "lon_max": -66.0,
}

STATION_COLS = {
    "GHCN_ID": "station_id",
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "ELEVATION": "elevation_m",
    "STATE": "state",
    "NAME": "station_name",
    "ISO_CODE": "iso_code",
}

SOURCE_COLUMN_ALIASES = {
    "station_id": ("Station_ID", "STATION"),
    "date_raw": ("DATE",),
    "lat": ("Latitude", "LATITUDE"),
    "lon": ("Longitude", "LONGITUDE"),
    "elevation_m": ("Elevation", "ELEVATION"),
    "precip_mm": ("precipitation",),
    "measurement_code": ("precipitation_Measurement_Code",),
    "quality_code": ("precipitation_Quality_Code",),
    "report_type": ("precipitation_Report_Type",),
    "source_code": ("precipitation_Source_Code",),
}

OUTPUT_COLUMNS = [
    "station_id",
    "time_utc",
    "report_time_utc",
    "lat",
    "lon",
    "elevation_m",
    "precip_mm",
    "measurement_code",
    "quality_code",
    "source_code",
    "report_type",
]

COMPACT_OUTPUT_COLUMNS = [col for col in OUTPUT_COLUMNS if col != "time_utc"]

# Documentation page VI, Table 3.
QC_GOOD_BY_SOURCE = {
    "220": {"1"},
    "221": {"1"},
    "222": {"1"},
    "223": {"1"},
    "347": {"1"},
    "348": {"1"},
    "313": {"1", "5"},
    "314": {"1", "5"},
    "315": {"1", "5"},
    "322": {"1", "5"},
    "335": {"1", "5"},
    "343": {"1", "5"},
    "344": {"1", "5"},
    "346": {"1", "5"},
    "382": {""},
}

TRACE_BY_SOURCE = {
    "220": {"2", "T"},
    "221": {"2", "T"},
    "222": {"2", "T"},
    "223": {"2", "T"},
    "347": {"2", "T"},
    "348": {"2", "T"},
    "313": {"2", "T"},
    "314": {"2", "T"},
    "315": {"2", "T"},
    "322": {"2", "T"},
    "335": {"2", "T"},
    "343": {"2", "T"},
    "344": {"2", "T"},
    "346": {"2", "T"},
    "382": {"T"},
}

# Strict hourly table policy: blank means no measurement qualifier, trace is
# converted to 0. All other documented precipitation qualifiers are excluded.
MEASUREMENT_GOOD_BY_SOURCE = {
    source: {""} | TRACE_BY_SOURCE[source] for source in QC_GOOD_BY_SOURCE
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ghcnh_precip")


@dataclass(frozen=True)
class StationYear:
    station_id: str
    year: int
    lat: float
    lon: float
    elevation_m: float
    station_name: str
    state: str


@dataclass
class DownloadResult:
    station_id: str
    year: int
    status: str
    path: str
    url: str
    size_bytes: int = 0
    error: str = ""


@dataclass
class ProcessResult:
    station_id: str
    year: int
    status: str
    input_rows: int = 0
    output_rows: int = 0
    error: str = ""


def parquet_url(station_id: str, year: int) -> str:
    return f"{BASE_URL}/access/by-year/{year}/parquet/GHCNh_{station_id}_{year}.parquet"


def raw_path(raw_dir: Path, station_id: str, year: int) -> Path:
    return raw_dir / str(year) / f"GHCNh_{station_id}_{year}.parquet"


def normalize_code(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def ensure_manifest_dir(output: Path, manifest_dir: Path | None) -> Path:
    target = manifest_dir or output.parent / "manifests"
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_station_years(docs_dir: Path, years: Iterable[int]) -> list[StationYear]:
    station_file = docs_dir / "ghcnh-station-list.csv"
    inventory_file = docs_dir / "ghcnh-inventory.txt"
    years_set = {int(y) for y in years}

    if not station_file.exists():
        raise FileNotFoundError(f"station list not found: {station_file}")
    if not inventory_file.exists():
        raise FileNotFoundError(f"inventory not found: {inventory_file}")

    stations = pd.read_csv(station_file)
    missing = sorted(set(STATION_COLS) - set(stations.columns))
    if missing:
        raise ValueError(f"station list is missing columns: {missing}")

    stations = stations.rename(columns=STATION_COLS)
    stations = stations[
        (stations["iso_code"] == "US")
        & stations["lat"].between(CONUS["lat_min"], CONUS["lat_max"])
        & stations["lon"].between(CONUS["lon_min"], CONUS["lon_max"])
    ].copy()

    inventory = pd.read_csv(inventory_file, sep=r"\s+")
    inventory = inventory[inventory["YEAR"].isin(years_set)].copy()
    inventory = inventory[["GHCNh_ID", "YEAR"]].drop_duplicates()

    merged = inventory.merge(
        stations,
        left_on="GHCNh_ID",
        right_on="station_id",
        how="inner",
    ).sort_values(["YEAR", "station_id"])

    station_years = [
        StationYear(
            station_id=str(row.station_id),
            year=int(row.YEAR),
            lat=float(row.lat),
            lon=float(row.lon),
            elevation_m=float(row.elevation_m),
            station_name=str(row.station_name),
            state="" if pd.isna(row.state) else str(row.state),
        )
        for row in merged.itertuples(index=False)
    ]

    log.info(
        "Station-year manifest: %s CONUS stations, %s station-years for %s",
        stations["station_id"].nunique(),
        len(station_years),
        ",".join(str(y) for y in sorted(years_set)),
    )
    return station_years


def write_station_year_manifest(station_years: list[StationYear], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "year",
                "lat",
                "lon",
                "elevation_m",
                "station_name",
                "state",
            ],
        )
        writer.writeheader()
        for item in station_years:
            writer.writerow(item.__dict__)


def download_one(
    item: StationYear,
    raw_dir: Path,
    timeout: int,
    retries: int,
) -> DownloadResult:
    url = parquet_url(item.station_id, item.year)
    dst = raw_path(raw_dir, item.station_id, item.year)
    if dst.exists() and dst.stat().st_size > 0:
        return DownloadResult(
            item.station_id,
            item.year,
            "cached",
            str(dst),
            url,
            dst.stat().st_size,
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    for attempt in range(1, retries + 2):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                if response.status_code == 404:
                    return DownloadResult(item.station_id, item.year, "404", str(dst), url)
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dst)
                return DownloadResult(
                    item.station_id,
                    item.year,
                    "ok",
                    str(dst),
                    url,
                    dst.stat().st_size,
                )
        except Exception as exc:  # noqa: BLE001 - recorded in manifest
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt <= retries:
                time.sleep(min(2**attempt, 20))
                continue
            return DownloadResult(
                item.station_id,
                item.year,
                "error",
                str(dst),
                url,
                error=repr(exc),
            )

    return DownloadResult(item.station_id, item.year, "error", str(dst), url)


def download_all(
    station_years: list[StationYear],
    raw_dir: Path,
    manifest_dir: Path,
    max_workers: int,
    timeout: int,
    retries: int,
) -> Counter:
    manifest_path = manifest_dir / "download_manifest.csv"
    counts: Counter = Counter()
    total = len(station_years)

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "year",
                "status",
                "path",
                "url",
                "size_bytes",
                "error",
            ],
        )
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(download_one, item, raw_dir, timeout, retries)
                for item in station_years
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                counts[result.status] += 1
                writer.writerow(result.__dict__)
                if index == total or index % 100 == 0:
                    log.info(
                        "Download progress %s/%s cached=%s ok=%s 404=%s error=%s",
                        index,
                        total,
                        counts["cached"],
                        counts["ok"],
                        counts["404"],
                        counts["error"],
                    )
    return counts


def resolve_source_columns(columns: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    available = set(columns)
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in SOURCE_COLUMN_ALIASES.items():
        match = next((alias for alias in aliases if alias in available), None)
        if match is None:
            missing.append(canonical)
        else:
            resolved[canonical] = match
    return resolved, missing


def read_precip_frame(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema_arrow.names
    except Exception as exc:  # noqa: BLE001
        return None, f"read_error:{exc!r}"

    resolved, missing = resolve_source_columns(columns)
    if missing:
        return None, "missing_columns:" + ",".join(missing)

    try:
        selected = list(dict.fromkeys(resolved.values()))
        frame = pd.read_parquet(path, columns=selected)
        frame = frame.rename(columns={source: canonical for canonical, source in resolved.items()})
        return frame, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"read_error:{exc!r}"


def qc_keep_mask(df: pd.DataFrame) -> tuple[pd.Series, Counter]:
    counts: Counter = Counter()
    source = df["source_code"].map(normalize_code)
    quality = df["quality_code"].map(normalize_code)
    keep_values: list[bool] = []

    for src, qc in zip(source, quality, strict=True):
        if src not in QC_GOOD_BY_SOURCE:
            counts[f"qc_unknown_source:{src or '<blank>'}"] += 1
            keep_values.append(False)
        elif qc in QC_GOOD_BY_SOURCE[src]:
            keep_values.append(True)
        else:
            counts[f"qc_reject:{src}:{qc or '<blank>'}"] += 1
            keep_values.append(False)

    return pd.Series(keep_values, index=df.index), counts


def measurement_keep_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, Counter]:
    counts: Counter = Counter()
    source = df["source_code"].map(normalize_code)
    measurement = df["measurement_code"].map(normalize_code)
    trace_values: list[bool] = []
    keep_values: list[bool] = []

    for src, meas in zip(source, measurement, strict=True):
        is_trace = src in TRACE_BY_SOURCE and meas in TRACE_BY_SOURCE[src]
        trace_values.append(is_trace)
        if src not in MEASUREMENT_GOOD_BY_SOURCE:
            counts[f"measurement_unknown_source:{src or '<blank>'}"] += 1
            keep_values.append(False)
        elif meas in MEASUREMENT_GOOD_BY_SOURCE[src]:
            keep_values.append(True)
        else:
            counts[f"measurement_reject:{src}:{meas or '<blank>'}"] += 1
            keep_values.append(False)

    return (
        pd.Series(keep_values, index=df.index),
        pd.Series(trace_values, index=df.index),
        counts,
    )


def process_station_year(
    item: StationYear,
    path: Path,
) -> tuple[pd.DataFrame | None, ProcessResult, Counter]:
    df, error = read_precip_frame(path)
    if df is None:
        return None, ProcessResult(item.station_id, item.year, "skip", error=error), Counter()

    input_rows = len(df)
    reject_counts: Counter = Counter()
    df["source_code"] = df["source_code"].map(normalize_code)
    df["measurement_code"] = df["measurement_code"].map(normalize_code)
    df["quality_code"] = df["quality_code"].map(normalize_code)
    df["report_type"] = df["report_type"].map(normalize_code)

    missing_precip = df["precip_mm"].isna()
    reject_counts["missing_precipitation"] += int(missing_precip.sum())
    df = df.loc[~missing_precip].copy()
    if df.empty:
        return (
            None,
            ProcessResult(item.station_id, item.year, "empty", input_rows=input_rows),
            reject_counts,
        )

    qc_mask, qc_counts = qc_keep_mask(df)
    reject_counts.update(qc_counts)
    df = df.loc[qc_mask].copy()
    if df.empty:
        return (
            None,
            ProcessResult(item.station_id, item.year, "empty", input_rows=input_rows),
            reject_counts,
        )

    meas_mask, trace_mask, meas_counts = measurement_keep_mask(df)
    reject_counts.update(meas_counts)
    df.loc[trace_mask, "precip_mm"] = 0.0
    df = df.loc[meas_mask].copy()
    if df.empty:
        return (
            None,
            ProcessResult(item.station_id, item.year, "empty", input_rows=input_rows),
            reject_counts,
        )

    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    invalid_precip = df["precip_mm"].isna() | (df["precip_mm"] < 0)
    reject_counts["invalid_or_negative_precipitation"] += int(invalid_precip.sum())
    df = df.loc[~invalid_precip].copy()
    if df.empty:
        return (
            None,
            ProcessResult(item.station_id, item.year, "empty", input_rows=input_rows),
            reject_counts,
        )

    df["report_time_utc"] = pd.to_datetime(df["date_raw"], utc=True, errors="coerce")
    invalid_time = df["report_time_utc"].isna()
    reject_counts["invalid_time"] += int(invalid_time.sum())
    df = df.loc[~invalid_time].copy()
    if df.empty:
        return (
            None,
            ProcessResult(item.station_id, item.year, "empty", input_rows=input_rows),
            reject_counts,
        )

    df["report_time_utc"] = df["report_time_utc"].dt.floor("min")
    df["time_utc"] = df["report_time_utc"].dt.floor("h")
    df["station_id"] = item.station_id
    df["lat"] = np.float32(item.lat)
    df["lon"] = np.float32(item.lon)
    df["elevation_m"] = np.float32(item.elevation_m)

    before_dedupe = len(df)
    df = df.sort_values(["station_id", "time_utc", "report_time_utc"])
    df = df.drop_duplicates(["station_id", "time_utc"], keep="last")
    reject_counts["duplicate_reports_within_hour"] += before_dedupe - len(df)

    df["precip_mm"] = df["precip_mm"].astype(np.float32)
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)

    return (
        df,
        ProcessResult(
            item.station_id,
            item.year,
            "ok",
            input_rows=input_rows,
            output_rows=len(df),
        ),
        reject_counts,
    )


def write_reject_counts(counter: Counter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in sorted(counter.items()):
            writer.writerow({"reason": reason, "count": count})


def append_processing_manifest(results: list[ProcessResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "year",
                "status",
                "input_rows",
                "output_rows",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_output_batch(
    buffer: list[pd.DataFrame],
    writer: pq.ParquetWriter | None,
    output: Path,
    output_columns: list[str],
    sort_output: bool,
    compression: str | None,
) -> tuple[pq.ParquetWriter | None, int]:
    if not buffer:
        return writer, 0

    batch = pd.concat(buffer, ignore_index=True)
    if sort_output:
        sort_cols = ["report_time_utc", "station_id"]
        if "time_utc" in batch.columns:
            sort_cols.insert(0, "time_utc")
        batch = batch.sort_values(sort_cols, kind="mergesort")
    batch = batch[output_columns]

    table = pa.Table.from_pandas(batch, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output, table.schema, compression=compression)
    writer.write_table(table)
    return writer, len(batch)


def process_all(
    station_years: list[StationYear],
    raw_dir: Path,
    output: Path,
    manifest_dir: Path,
    omit_time_utc: bool,
    sort_output: bool,
    write_batch_rows: int,
    compression: str | None,
) -> tuple[int, Counter]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    accepted_rows = 0
    buffered_rows = 0
    output_buffer: list[pd.DataFrame] = []
    output_columns = COMPACT_OUTPUT_COLUMNS if omit_time_utc else OUTPUT_COLUMNS
    reject_counts: Counter = Counter()
    process_results: list[ProcessResult] = []

    try:
        for index, item in enumerate(station_years, start=1):
            path = raw_path(raw_dir, item.station_id, item.year)
            if not path.exists():
                process_results.append(
                    ProcessResult(item.station_id, item.year, "missing_raw")
                )
                reject_counts["missing_raw_file"] += 1
                continue

            chunk, result, counts = process_station_year(item, path)
            process_results.append(result)
            reject_counts.update(counts)

            if chunk is not None and not chunk.empty:
                accepted_rows += len(chunk)
                output_buffer.append(chunk)
                buffered_rows += len(chunk)
                if buffered_rows >= write_batch_rows:
                    writer, written_rows = write_output_batch(
                        output_buffer,
                        writer,
                        output,
                        output_columns,
                        sort_output,
                        compression,
                    )
                    total_rows += written_rows
                    output_buffer = []
                    buffered_rows = 0

            if index == len(station_years) or index % 100 == 0:
                log.info(
                    "Process progress %s/%s output_rows=%s",
                    index,
                    len(station_years),
                    accepted_rows,
                )
    finally:
        if output_buffer:
            writer, written_rows = write_output_batch(
                output_buffer,
                writer,
                output,
                output_columns,
                sort_output,
                compression,
            )
            total_rows += written_rows
        if writer is not None:
            writer.close()

    append_processing_manifest(process_results, manifest_dir / "processing_manifest.csv")
    write_reject_counts(reject_counts, manifest_dir / "reject_counts.csv")
    return total_rows, reject_counts


def inspect_sample(
    station_id: str,
    year: int,
    raw_dir: Path,
    timeout: int,
    retries: int,
) -> None:
    item = StationYear(station_id, year, np.nan, np.nan, np.nan, "", "")
    result = download_one(item, raw_dir, timeout, retries)
    if result.status not in {"cached", "ok"}:
        raise RuntimeError(f"sample download failed: {result}")

    df = pd.read_parquet(result.path)
    precip_cols = [col for col in df.columns if "precipitation" in col.lower()]
    print("Columns:")
    print(list(df.columns))
    print("\nPrecipitation columns:")
    print(precip_cols)
    print("\nPrecipitation sample:")
    print(df[precip_cols].head(20).to_string())
    for col in [
        "precipitation_Source_Code",
        "precipitation_Quality_Code",
        "precipitation_Measurement_Code",
        "precipitation_Report_Type",
    ]:
        if col in df.columns:
            print(f"\n{col} distribution:")
            print(df[col].map(normalize_code).value_counts(dropna=False).head(30).to_string())


def validate_output(output: Path, omit_time_utc: bool = False) -> None:
    if not output.exists():
        raise FileNotFoundError(f"output parquet not found: {output}")

    expected_columns = COMPACT_OUTPUT_COLUMNS if omit_time_utc else OUTPUT_COLUMNS
    table = pq.read_table(output)
    missing = [col for col in expected_columns if col not in table.column_names]
    if missing:
        raise AssertionError(f"output missing columns: {missing}")

    df = table.to_pandas()
    if not df.empty:
        if "time_utc" in df.columns:
            duplicated = df.duplicated(["station_id", "time_utc"]).sum()
            if duplicated:
                raise AssertionError(f"duplicate station/hour rows: {duplicated}")
            if str(df["time_utc"].dt.tz) != "UTC":
                raise AssertionError("time_utc is not UTC")
        if (df["precip_mm"] < 0).any():
            raise AssertionError("negative precipitation values found")
        if str(df["report_time_utc"].dt.tz) != "UTC":
            raise AssertionError("report_time_utc is not UTC")
    log.info("Output validation passed: rows=%s columns=%s", len(df), len(df.columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/final/ghcnh_conus_precip_2020_2022.parquet"),
    )
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument(
        "--omit-time-utc",
        action="store_true",
        help="omit the hourly time_utc column from final output; station-hour dedupe still uses it internally",
    )
    parser.add_argument(
        "--sort-output-by-report-time",
        action="store_true",
        help="sort buffered output batches by report_time_utc/station_id before writing for better timestamp compression",
    )
    parser.add_argument(
        "--write-batch-rows",
        type=int,
        default=1_000_000,
        help="target accepted rows per output write batch",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "zstd", "gzip", "brotli", "none"],
        help="Parquet compression codec",
    )
    parser.add_argument("--inspect-sample", action="store_true")
    parser.add_argument("--inspect-station", default=DEFAULT_SAMPLE_STATION)
    parser.add_argument("--inspect-year", type=int, default=DEFAULT_SAMPLE_YEAR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_batch_rows < 1:
        raise ValueError("--write-batch-rows must be >= 1")
    compression = None if args.compression == "none" else args.compression
    manifest_dir = ensure_manifest_dir(args.output, args.manifest_dir)

    if args.inspect_sample:
        inspect_sample(
            args.inspect_station,
            args.inspect_year,
            args.raw_dir,
            args.timeout,
            args.retries,
        )
        return 0

    station_years = load_station_years(args.docs_dir, args.years)
    if args.sample is not None:
        station_years = station_years[: args.sample]
        log.info("Sample mode: first %s station-years", len(station_years))

    write_station_year_manifest(station_years, manifest_dir / "station_year_manifest.csv")
    if args.dry_run_manifest:
        return 0

    if not args.process_only:
        download_counts = download_all(
            station_years,
            args.raw_dir,
            manifest_dir,
            args.max_workers,
            args.timeout,
            args.retries,
        )
        log.info("Download complete: %s", dict(download_counts))

    if not args.download_only:
        rows, reject_counts = process_all(
            station_years,
            args.raw_dir,
            args.output,
            manifest_dir,
            args.omit_time_utc,
            args.sort_output_by_report_time,
            args.write_batch_rows,
            compression,
        )
        log.info("Processing complete: rows=%s output=%s", rows, args.output)
        log.info("Reject counts written: %s", manifest_dir / "reject_counts.csv")
        if args.validate_output:
            validate_output(args.output, args.omit_time_utc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
