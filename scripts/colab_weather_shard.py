#!/usr/bin/env python3
"""Remote Colab worker for one deterministic GHCNh year shard."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq


CONTENT = Path("/content")
JOB_PATH = CONTENT / "weather_job.json"
BUNDLE_PATH = CONTENT / "ghcnh_weather_bundle.tgz"


def valid_output(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:  # noqa: BLE001
        return False
    return metadata.num_rows > 0 and metadata.num_columns == 58


def main() -> int:
    job = json.loads(JOB_PATH.read_text())
    year = int(job["year"])
    shard = int(job["shard_index"])
    num_shards = int(job["num_shards"])
    drive_root = job.get("drive_root")

    root = CONTENT / "GHCNh"
    deliverable = CONTENT / "deliverable"
    raw_root = CONTENT / "ghcnh_raw"
    for path in (root, deliverable, raw_root):
        if path.exists():
            shutil.rmtree(path)

    root.mkdir(parents=True)
    with tarfile.open(BUNDLE_PATH, "r:gz") as archive:
        archive.extractall(root, filter="data")

    output_root = (
        Path(drive_root) / "weather_by_year"
        if drive_root
        else deliverable / "weather_by_year"
    )
    output = output_root / f"year={year}" / f"part-{shard:03d}.parquet"
    manifests = output_root / "manifests" / str(year) / f"shard-{shard:03d}"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    if valid_output(output):
        print(f"SKIP_REMOTE existing {output}")
        return 0

    sys.path.insert(0, str(root / "src"))
    sys.argv = [
        str(root / "src/weather_pipeline.py"),
        "--docs-dir",
        str(root / "docs"),
        "--raw-dir",
        str(raw_root),
        "--output",
        str(output),
        "--manifest-dir",
        str(manifests),
        "--years",
        str(year),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard),
        "--max-workers",
        str(job.get("download_workers", 16)),
        "--process-workers",
        str(job.get("process_workers", 4)),
        "--compression",
        "zstd",
        "--write-batch-rows",
        str(job.get("write_batch_rows", 500_000)),
        "--validate-output",
    ]

    try:
        runpy.run_path(str(root / "src/weather_pipeline.py"), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    shutil.rmtree(raw_root)

    if drive_root:
        print(f"DRIVE_OUTPUT={output}")
    else:
        archive_path = CONTENT / f"ghcnh_weather_{year}_shard_{shard:03d}.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(output_root, arcname="weather_by_year")
        print(f"RESULT_ARCHIVE={archive_path}")
    print(f"OUTPUT_BYTES={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    main()
