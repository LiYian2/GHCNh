#!/usr/bin/env python3
"""Build a compact, quality-controlled hourly GHCNh weather dataset."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipelinev1 import (
    DEFAULT_YEARS,
    MEASUREMENT_GOOD_BY_SOURCE,
    QC_GOOD_BY_SOURCE,
    TRACE_BY_SOURCE,
    ProcessResult,
    StationYear,
    download_all,
    ensure_manifest_dir,
    load_station_years,
    normalize_code,
    raw_path,
    write_station_year_manifest,
)


log = logging.getLogger("ghcnh_weather")

BASE_COLUMN_ALIASES = {
    "station_id": ("Station_ID", "STATION"),
    "date_raw": ("DATE",),
}


@dataclass(frozen=True)
class VariableSpec:
    source: str
    output: str
    kind: str = "numeric"
    minimum: float | None = None
    maximum: float | None = None
    provenance: bool = False
    precipitation: bool = False


VARIABLES = (
    VariableSpec("temperature", "temperature_c", minimum=-90, maximum=60, provenance=True),
    VariableSpec(
        "dew_point_temperature",
        "dew_point_c",
        minimum=-100,
        maximum=60,
        provenance=True,
    ),
    VariableSpec(
        "relative_humidity",
        "relative_humidity_pct",
        minimum=0,
        maximum=100,
    ),
    VariableSpec(
        "wet_bulb_temperature",
        "wet_bulb_c",
        minimum=-100,
        maximum=60,
    ),
    VariableSpec(
        "station_level_pressure",
        "station_pressure_hpa",
        minimum=400,
        maximum=1100,
        provenance=True,
    ),
    VariableSpec(
        "sea_level_pressure",
        "sea_level_pressure_hpa",
        minimum=800,
        maximum=1100,
        provenance=True,
    ),
    VariableSpec(
        "altimeter",
        "altimeter_hpa",
        minimum=800,
        maximum=1100,
        provenance=True,
    ),
    VariableSpec(
        "pressure_3hr_change",
        "pressure_change_3h_hpa",
        minimum=-100,
        maximum=100,
    ),
    VariableSpec(
        "wind_direction",
        "wind_direction_deg",
        minimum=0,
        maximum=360,
        provenance=True,
    ),
    VariableSpec(
        "wind_speed",
        "wind_speed_mps",
        minimum=0,
        maximum=150,
        provenance=True,
    ),
    VariableSpec(
        "wind_gust",
        "wind_gust_mps",
        minimum=0,
        maximum=150,
        provenance=True,
    ),
    VariableSpec(
        "visibility",
        "visibility_km",
        minimum=0,
        maximum=1000,
        provenance=True,
    ),
    VariableSpec(
        "precipitation",
        "precip_1h_mm",
        minimum=0,
        maximum=2000,
        provenance=True,
        precipitation=True,
    ),
    VariableSpec(
        "precipitation_3_hour",
        "precip_3h_mm",
        minimum=0,
        maximum=3000,
        precipitation=True,
    ),
    VariableSpec(
        "precipitation_6_hour",
        "precip_6h_mm",
        minimum=0,
        maximum=4000,
        precipitation=True,
    ),
    VariableSpec(
        "precipitation_24_hour",
        "precip_24h_mm",
        minimum=0,
        maximum=5000,
        precipitation=True,
    ),
    VariableSpec("snow_depth", "snow_depth_mm", minimum=0, maximum=20000),
    VariableSpec(
        "ceiling_height",
        "ceiling_height_m",
        minimum=0,
        maximum=50000,
        provenance=True,
    ),
    VariableSpec("sky_condition", "sky_condition", kind="code"),
    VariableSpec(
        "sky_condition_baseht",
        "sky_condition_base_m",
        minimum=0,
        maximum=50000,
    ),
    VariableSpec("sky_cover_summation_1", "cloud_cover_1", kind="code"),
    VariableSpec(
        "sky_cover_summation_baseht_1",
        "cloud_base_1_m",
        minimum=0,
        maximum=50000,
    ),
    VariableSpec("sky_cover_summation_2", "cloud_cover_2", kind="code"),
    VariableSpec(
        "sky_cover_summation_baseht_2",
        "cloud_base_2_m",
        minimum=0,
        maximum=50000,
    ),
    VariableSpec("sky_cover_summation_3", "cloud_cover_3", kind="code"),
    VariableSpec(
        "sky_cover_summation_baseht_3",
        "cloud_base_3_m",
        minimum=0,
        maximum=50000,
    ),
    VariableSpec("sky_cover_summation_4", "cloud_cover_4", kind="code"),
    VariableSpec(
        "sky_cover_summation_baseht_4",
        "cloud_base_4_m",
        minimum=0,
        maximum=50000,
    ),
    VariableSpec("pres_wx_MW1", "weather_manual_1", kind="code"),
    VariableSpec("pres_wx_MW2", "weather_manual_2", kind="code"),
    VariableSpec("pres_wx_MW3", "weather_manual_3", kind="code"),
    VariableSpec("pres_wx_AU1", "weather_auto_text_1", kind="code"),
    VariableSpec("pres_wx_AU2", "weather_auto_text_2", kind="code"),
    VariableSpec("pres_wx_AU3", "weather_auto_text_3", kind="code"),
    VariableSpec("pres_wx_AW1", "weather_auto_code_1", kind="code"),
    VariableSpec("pres_wx_AW2", "weather_auto_code_2", kind="code"),
    VariableSpec("pres_wx_AW3", "weather_auto_code_3", kind="code"),
)

NUMERIC_OUTPUTS = [spec.output for spec in VARIABLES if spec.kind == "numeric"]
CODE_OUTPUTS = [spec.output for spec in VARIABLES if spec.kind == "code"]
PROVENANCE_OUTPUTS = [
    f"{spec.output}_source_qc" for spec in VARIABLES if spec.provenance
]
FLAG_OUTPUTS = ["wind_calm", "wind_variable", "precip_trace"]

WEATHER_OUTPUT_COLUMNS = [
    "station_id",
    "report_time_utc",
    "lat",
    "lon",
    "elevation_m",
    "report_type",
    *NUMERIC_OUTPUTS,
    "pressure_tendency_code",
    *FLAG_OUTPUTS,
    *CODE_OUTPUTS,
    *PROVENANCE_OUTPUTS,
]

# GHCNh source groups documented in Table 3.
LEGACY_GROUP_A = {"220", "221", "222", "223", "347", "348"}
LEGACY_GROUP_B = {"313", "314", "315", "322", "335", "343", "344", "346"}
GENERIC_BAD_QC = {"2", "3", "6", "7"}
GROUP_A_BAD_QC = GENERIC_BAD_QC | {"0", "5"}
QC_SPLIT = re.compile(r"[;,:|]+")


def source_columns_for(spec: VariableSpec) -> tuple[str, ...]:
    base = spec.source
    return (
        base,
        f"{base}_Measurement_Code",
        f"{base}_Quality_Code",
        f"{base}_Report_Type",
        f"{base}_Source_Code",
    )


def resolve_column(columns: set[str], aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in columns), None)


def normalize_quality_tokens(value: object) -> tuple[str, ...]:
    code = normalize_code(value).strip(" ;:")
    if not code:
        return ()
    return tuple(token for token in QC_SPLIT.split(code) if token)


def acceptable_quality(source: object, quality: object) -> bool:
    src = normalize_code(source)
    tokens = normalize_quality_tokens(quality)
    if not tokens:
        return True
    for token in tokens:
        if token.islower():
            return False
        if src in LEGACY_GROUP_A and token in GROUP_A_BAD_QC:
            return False
        if src in LEGACY_GROUP_B and token in GENERIC_BAD_QC:
            return False
        if src == "345" and token != "0":
            return False
        if src not in LEGACY_GROUP_A | LEGACY_GROUP_B | {"345", "382"}:
            if token in GENERIC_BAD_QC | {"5"}:
                return False
    return True


def general_quality_mask(source: pd.Series, quality: pd.Series) -> pd.Series:
    src = normalize_code_series(source)
    qc = normalize_code_series(quality)
    lower_failure = qc.str.contains(r"[a-z]", regex=True, na=False)

    separators = r"(?:^|[;,:|])"
    ending = r"(?:$|[;,:|])"
    group_a_bad = qc.str.contains(
        separators + r"(?:0|2|3|5|6|7)" + ending,
        regex=True,
        na=False,
    )
    group_b_bad = qc.str.contains(
        separators + r"(?:2|3|6|7)" + ending,
        regex=True,
        na=False,
    )
    unknown_bad = qc.str.contains(
        separators + r"(?:2|3|5|6|7)" + ending,
        regex=True,
        na=False,
    )

    group_a = src.isin(LEGACY_GROUP_A)
    group_b = src.isin(LEGACY_GROUP_B)
    source_345 = src.eq("345")
    known = group_a | group_b | source_345 | src.eq("382")
    bad = lower_failure.copy()
    bad |= group_a & group_a_bad
    bad |= group_b & group_b_bad
    bad |= source_345 & qc.ne("") & qc.ne("0")
    bad |= ~known & unknown_bad
    return (~bad).astype(bool)


def precipitation_masks(
    source: pd.Series,
    quality: pd.Series,
    measurement: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    src = normalize_code_series(source)
    qc = normalize_code_series(quality)
    meas = normalize_code_series(measurement)
    keep = pd.Series(False, index=source.index, dtype=bool)
    trace = pd.Series(False, index=source.index, dtype=bool)
    for source_code, good_quality in QC_GOOD_BY_SOURCE.items():
        selected = src.eq(source_code)
        keep_for_source = (
            selected
            & qc.isin(good_quality)
            & meas.isin(MEASUREMENT_GOOD_BY_SOURCE[source_code])
        )
        keep |= keep_for_source
        trace |= keep_for_source & meas.isin(TRACE_BY_SOURCE[source_code])
    return keep, trace


def read_weather_frame(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
    except Exception as exc:  # noqa: BLE001
        return None, f"read_error:{exc!r}"

    station_col = resolve_column(available, BASE_COLUMN_ALIASES["station_id"])
    date_col = resolve_column(available, BASE_COLUMN_ALIASES["date_raw"])
    if station_col is None or date_col is None:
        return None, "missing_columns:station_id_or_date"

    try:
        measurement_columns = [
            spec.source for spec in VARIABLES if spec.source in available
        ]
        frame = pd.read_parquet(
            path,
            columns=[station_col, date_col, *measurement_columns],
        )
        active_specs = [
            spec
            for spec in VARIABLES
            if spec.source in frame and frame[spec.source].notna().any()
        ]
        active_sources = {spec.source for spec in active_specs}
        inactive_sources = [
            name for name in measurement_columns if name not in active_sources
        ]
        if inactive_sources:
            frame = frame.drop(columns=inactive_sources)
        companion_columns = list(
            dict.fromkeys(
                name
                for spec in active_specs
                for name in source_columns_for(spec)[1:]
                if name in available
            )
        )
        if companion_columns:
            companions = pd.read_parquet(path, columns=companion_columns)
            frame = pd.concat(
                [frame.reset_index(drop=True), companions.reset_index(drop=True)],
                axis=1,
            )
    except Exception as exc:  # noqa: BLE001
        return None, f"read_error:{exc!r}"

    frame = frame.rename(columns={station_col: "station_id", date_col: "date_raw"})
    return frame, ""


def clean_code_series(series: pd.Series) -> pd.Series:
    values = normalize_code_series(series).replace("", pd.NA)
    return values.astype("string")


def normalize_code_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    values = values.replace({"nan": "", "none": "", "<NA>": ""})
    numeric_float = values.str.fullmatch(r"\d+\.0", na=False)
    values = values.where(~numeric_float, values.str[:-2])
    return values


def paired_source_qc(source: pd.Series, quality: pd.Series) -> pd.Series:
    src = normalize_code_series(source)
    qc = normalize_code_series(quality)
    combined = src + ":" + qc
    combined = combined.where(src.ne(""), pd.NA)
    return combined.astype("string")


def prepare_variable(
    frame: pd.DataFrame,
    spec: VariableSpec,
    prepared: pd.DataFrame,
    reject_counts: Counter,
) -> None:
    if spec.source not in frame:
        return
    if not frame[spec.source].notna().any():
        return

    index = frame.index
    measurement_col = f"{spec.source}_Measurement_Code"
    quality_col = f"{spec.source}_Quality_Code"
    report_col = f"{spec.source}_Report_Type"
    source_col = f"{spec.source}_Source_Code"
    measurement = frame.get(measurement_col, pd.Series("", index=index))
    quality = frame.get(quality_col, pd.Series("", index=index))
    report_type = frame.get(report_col, pd.Series("", index=index))
    source = frame.get(source_col, pd.Series("", index=index))

    if spec.kind == "numeric":
        values = pd.to_numeric(frame[spec.source], errors="coerce")
        invalid = frame[spec.source].notna() & values.isna()
        if spec.minimum is not None:
            invalid |= values.lt(spec.minimum)
        if spec.maximum is not None:
            invalid |= values.gt(spec.maximum)
    else:
        values = clean_code_series(frame[spec.source])
        invalid = pd.Series(False, index=index)

    if spec.precipitation:
        quality_keep, trace = precipitation_masks(source, quality, measurement)
        if spec.kind == "numeric":
            values = values.mask(trace, 0.0)
    else:
        quality_keep = general_quality_mask(source, quality)

    rejected_qc = frame[spec.source].notna() & ~quality_keep
    reject_counts[f"{spec.output}:qc"] += int(rejected_qc.sum())
    reject_counts[f"{spec.output}:invalid"] += int(invalid.sum())
    keep = quality_keep & ~invalid
    values = values.where(keep)

    if spec.output == "precip_1h_mm":
        prepared["precip_trace"] = trace.where(keep & values.notna()).astype("boolean")

    if spec.kind == "numeric":
        prepared[spec.output] = values.astype("float32")
    else:
        prepared[spec.output] = values.astype("string")

    if spec.provenance:
        provenance = paired_source_qc(source, quality).where(values.notna())
        prepared[f"{spec.output}_source_qc"] = provenance.astype("string")

    if "report_type" not in prepared:
        prepared["report_type"] = clean_code_series(report_type)
    else:
        prepared["report_type"] = prepared["report_type"].fillna(
            clean_code_series(report_type)
        )

    if spec.output == "pressure_change_3h_hpa":
        tendency = pd.to_numeric(measurement, errors="coerce")
        tendency = tendency.where(tendency.between(0, 8) & values.notna())
        prepared["pressure_tendency_code"] = tendency.astype("Int8")

    if spec.output == "wind_direction_deg":
        meas = normalize_code_series(measurement)
        observed = frame[spec.source].notna() & quality_keep
        prepared["wind_calm"] = meas.eq("C").where(observed).astype("boolean")
        prepared["wind_variable"] = meas.eq("V").where(observed).astype("boolean")


def ensure_output_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_OUTPUTS:
        if column not in frame:
            frame[column] = pd.Series(np.nan, index=frame.index, dtype="float32")
        else:
            frame[column] = frame[column].astype("float32")
    for column in CODE_OUTPUTS + PROVENANCE_OUTPUTS + ["report_type"]:
        if column not in frame:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
        else:
            frame[column] = frame[column].astype("string")
    for column in FLAG_OUTPUTS:
        if column not in frame:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        else:
            frame[column] = frame[column].astype("boolean")
    if "pressure_tendency_code" not in frame:
        frame["pressure_tendency_code"] = pd.Series(
            pd.NA, index=frame.index, dtype="Int8"
        )
    else:
        frame["pressure_tendency_code"] = frame["pressure_tendency_code"].astype("Int8")
    return frame


def process_station_year(
    item: StationYear,
    path: Path,
) -> tuple[pd.DataFrame | None, ProcessResult, Counter]:
    frame, error = read_weather_frame(path)
    if frame is None:
        return None, ProcessResult(item.station_id, item.year, "skip", error=error), Counter()

    input_rows = len(frame)
    reject_counts: Counter = Counter()
    prepared = pd.DataFrame(index=frame.index)
    prepared["report_time_utc"] = pd.to_datetime(
        frame["date_raw"], utc=True, errors="coerce"
    ).dt.floor("min")
    invalid_time = prepared["report_time_utc"].isna()
    reject_counts["invalid_time"] += int(invalid_time.sum())
    prepared = prepared.loc[~invalid_time].copy()
    frame = frame.loc[prepared.index]

    for spec in VARIABLES:
        prepare_variable(frame, spec, prepared, reject_counts)

    prepared = ensure_output_dtypes(prepared)
    prepared["time_utc"] = prepared["report_time_utc"].dt.floor("h")
    prepared = prepared.sort_values("report_time_utc", kind="mergesort")

    value_columns = NUMERIC_OUTPUTS + CODE_OUTPUTS
    grouped = prepared.groupby("time_utc", sort=True).last().reset_index()
    grouped = grouped.loc[grouped[value_columns].notna().any(axis=1)].copy()
    if grouped.empty:
        return (
            None,
            ProcessResult(item.station_id, item.year, "empty", input_rows=input_rows),
            reject_counts,
        )

    grouped["station_id"] = item.station_id
    grouped["lat"] = np.float32(item.lat)
    grouped["lon"] = np.float32(item.lon)
    grouped["elevation_m"] = np.float32(item.elevation_m)
    grouped = ensure_output_dtypes(grouped)
    grouped = grouped[WEATHER_OUTPUT_COLUMNS].reset_index(drop=True)

    reject_counts["duplicate_reports_within_hour"] += input_rows - len(grouped)
    return (
        grouped,
        ProcessResult(
            item.station_id,
            item.year,
            "ok",
            input_rows=input_rows,
            output_rows=len(grouped),
        ),
        reject_counts,
    )


def write_processing_manifest(results: list[ProcessResult], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
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


def write_reject_counts(counter: Counter, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in sorted(counter.items()):
            writer.writerow({"reason": reason, "count": count})


def write_output_batch(
    frames: list[pd.DataFrame],
    writer: pq.ParquetWriter | None,
    output: Path,
    compression: str | None,
) -> tuple[pq.ParquetWriter | None, int]:
    batch = pd.concat(frames, ignore_index=True)
    batch = batch.sort_values(["report_time_utc", "station_id"], kind="mergesort")
    table = pa.Table.from_pandas(batch[WEATHER_OUTPUT_COLUMNS], preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(
            output,
            table.schema,
            compression=compression,
            use_dictionary=True,
        )
    writer.write_table(table)
    return writer, len(batch)


def process_all(
    station_years: list[StationYear],
    raw_dir: Path,
    output: Path,
    manifest_dir: Path,
    write_batch_rows: int,
    compression: str | None,
    process_workers: int,
) -> tuple[int, Counter]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    writer: pq.ParquetWriter | None = None
    output_buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    total_rows = 0
    reject_counts: Counter = Counter()
    results: list[ProcessResult] = []

    def process_item(item: StationYear):
        path = raw_path(raw_dir, item.station_id, item.year)
        if not path.exists():
            return (
                None,
                ProcessResult(item.station_id, item.year, "missing_raw"),
                Counter({"missing_raw_file": 1}),
            )
        return process_station_year(item, path)

    try:
        with ThreadPoolExecutor(max_workers=process_workers) as pool:
            futures = [pool.submit(process_item, item) for item in station_years]
            for index, future in enumerate(as_completed(futures), start=1):
                chunk, result, counts = future.result()
                results.append(result)
                reject_counts.update(counts)

                if chunk is not None and not chunk.empty:
                    output_buffer.append(chunk)
                    buffered_rows += len(chunk)
                    if buffered_rows >= write_batch_rows:
                        writer, written = write_output_batch(
                            output_buffer, writer, output, compression
                        )
                        total_rows += written
                        output_buffer = []
                        buffered_rows = 0

                if index == len(station_years) or index % 100 == 0:
                    log.info(
                        "Process progress %s/%s output_rows=%s",
                        index,
                        len(station_years),
                        total_rows + buffered_rows,
                    )
    finally:
        if output_buffer:
            writer, written = write_output_batch(
                output_buffer, writer, output, compression
            )
            total_rows += written
        if writer is not None:
            writer.close()

    results.sort(key=lambda result: (result.year, result.station_id))
    write_processing_manifest(results, manifest_dir / "processing_manifest.csv")
    write_reject_counts(reject_counts, manifest_dir / "reject_counts.csv")
    return total_rows, reject_counts


def validate_output(output: Path) -> None:
    parquet_file = pq.ParquetFile(output)
    columns = parquet_file.schema_arrow.names
    missing = [column for column in WEATHER_OUTPUT_COLUMNS if column not in columns]
    if missing:
        raise AssertionError(f"output missing columns: {missing}")
    if parquet_file.metadata.num_rows == 0:
        raise AssertionError("output contains no rows")
    log.info(
        "Output validation passed: rows=%s columns=%s row_groups=%s",
        parquet_file.metadata.num_rows,
        len(columns),
        parquet_file.metadata.num_row_groups,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_weather"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/final/ghcnh_conus_weather_2020_2022_compact.parquet"),
    )
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--station-ids",
        nargs="+",
        default=None,
        help="optionally keep only these GHCNh station IDs",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--dry-run-manifest", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument("--write-batch-rows", type=int, default=500_000)
    parser.add_argument(
        "--process-workers",
        type=int,
        default=4,
        help="number of station-year files to process concurrently",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["snappy", "zstd", "gzip", "brotli", "none"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_batch_rows < 1:
        raise ValueError("--write-batch-rows must be >= 1")
    if args.process_workers < 1:
        raise ValueError("--process-workers must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num-shards")
    if args.download_only and args.process_only:
        raise ValueError("--download-only and --process-only are mutually exclusive")

    compression = None if args.compression == "none" else args.compression
    manifest_dir = ensure_manifest_dir(args.output, args.manifest_dir)
    station_years = load_station_years(args.docs_dir, args.years)
    if args.station_ids:
        requested = set(args.station_ids)
        station_years = [
            item for item in station_years if item.station_id in requested
        ]
        missing = requested - {item.station_id for item in station_years}
        if missing:
            raise ValueError(
                f"requested stations unavailable for selected years: {sorted(missing)}"
            )
        log.info("Station filter: %s station-years", len(station_years))
    if args.num_shards > 1:
        station_years = [
            item
            for position, item in enumerate(station_years)
            if position % args.num_shards == args.shard_index
        ]
        log.info(
            "Shard %s/%s: %s station-years",
            args.shard_index,
            args.num_shards,
            len(station_years),
        )
    if args.sample is not None:
        station_years = station_years[: args.sample]
        log.info("Sample mode: first %s station-years", len(station_years))
    write_station_year_manifest(station_years, manifest_dir / "station_year_manifest.csv")
    if args.dry_run_manifest:
        return 0

    if not args.process_only:
        counts = download_all(
            station_years,
            args.raw_dir,
            manifest_dir,
            args.max_workers,
            args.timeout,
            args.retries,
        )
        log.info("Download complete: %s", dict(counts))

    if not args.download_only:
        rows, _ = process_all(
            station_years,
            args.raw_dir,
            args.output,
            manifest_dir,
            args.write_batch_rows,
            compression,
            args.process_workers,
        )
        log.info("Processing complete: rows=%s output=%s", rows, args.output)
        if args.validate_output:
            validate_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
