# GHCNh CONUS Hourly Precipitation Pipeline

This repo builds a quality-controlled hourly precipitation dataset for the
contiguous United States from NOAA's Global Historical Climatology Network
hourly product (GHCNh).

The main implementation is [`src/pipelinev1.py`](src/pipelinev1.py). The
earlier prototype is kept as [`src/pipelinev0.py`](src/pipelinev0.py) for
comparison.

## What It Produces

Default output:

```text
data/final/ghcnh_conus_precip_2020_2022.parquet
```

Final Parquet schema:

| Column | Meaning |
| --- | --- |
| `station_id` | GHCNh station ID |
| `time_utc` | UTC hour start |
| `report_time_utc` | original report timestamp retained for that hour |
| `lat`, `lon` | station coordinates |
| `elevation_m` | station elevation in meters |
| `precip_mm` | hourly precipitation in millimeters |
| `measurement_code` | original precipitation measurement code |
| `quality_code` | original precipitation quality code |
| `source_code` | original GHCNh source code |
| `report_type` | original report type, such as FM15/FM16 |

The script also writes audit manifests under `data/final/manifests/`.

## Data Sources

The pipeline expects the official GHCNh metadata files in `docs/`:

- `ghcnh-station-list.csv`
- `ghcnh-inventory.txt`
- `ghcnh_DOCUMENTATION.pdf`
- `ghcnh-columns.pdf`

Raw station-year Parquet files are downloaded from NOAA using this URL pattern:

```text
https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/{year}/parquet/GHCNh_{station_id}_{year}.parquet
```

## Install

Use the local environment requested for this project:

```bash
source ~/.venvs/py313/bin/activate
pip install -r requirements.txt
```

## Usage

Inspect one known sample file and print precipitation-related columns:

```bash
python src/pipelinev1.py --inspect-sample
```

Build only the station-year manifest without downloading raw data:

```bash
python src/pipelinev1.py --dry-run-manifest
```

Run a small end-to-end smoke test:

```bash
python src/pipelinev1.py \
  --sample 5 \
  --max-workers 4 \
  --validate-output \
  --output data/final/sample_5.parquet \
  --manifest-dir data/final/sample_5_manifests
```

Run the full local pipeline:

```bash
python src/pipelinev1.py --max-workers 16
```

Run only the downloader:

```bash
python src/pipelinev1.py --download-only --max-workers 16
```

Process already-downloaded raw files:

```bash
python src/pipelinev1.py --process-only
```

Use custom paths:

```bash
python src/pipelinev1.py \
  --docs-dir docs \
  --raw-dir data/raw \
  --output data/final/ghcnh_conus_precip_2020_2022.parquet \
  --manifest-dir data/final/manifests \
  --years 2020 2021 2022 \
  --max-workers 16
```

## Colab Usage

In Colab, keep raw files on the session disk and write only the final Parquet to
Google Drive:

```bash
python /content/GHCNh/src/pipelinev1.py \
  --docs-dir /content/GHCNh/docs \
  --raw-dir /content/ghcnh_raw \
  --output /content/drive/MyDrive/GHCNh/ghcnh_conus_precip_2020_2022.parquet \
  --max-workers 16
```

Mount Google Drive manually in the Colab UI or a notebook cell before running
the full job.

## Pipeline Logic

1. Load station metadata from `docs/ghcnh-station-list.csv`.
2. Keep contiguous-US stations only:
   - `ISO_CODE == "US"`
   - latitude between `24` and `50`
   - longitude between `-125` and `-66`
3. Load `docs/ghcnh-inventory.txt` and keep station-years present for the
   requested years.
4. Download each station-year Parquet file into `data/raw/{year}/`.
5. Read only the hourly precipitation columns:
   - `precipitation`
   - `precipitation_Measurement_Code`
   - `precipitation_Quality_Code`
   - `precipitation_Report_Type`
   - `precipitation_Source_Code`
6. Normalize source, quality, and measurement codes.
7. Drop missing precipitation.
8. Apply strict source-aware QC:
   - sources `220,221,222,223,347,348`: keep quality code `1`
   - sources `313,314,315,322,335,343,344,346`: keep quality codes `1` and `5`
   - source `382`: keep blank quality code only
9. Apply strict measurement-code filtering:
   - trace codes become `precip_mm = 0`
   - accumulated, missing, deleted, estimated, incomplete, suspect, or unknown
     measurement flags are rejected for the hourly table
10. Convert report times to UTC and compute `time_utc = floor(report_time, hour)`.
11. For each `station_id + time_utc`, keep the last report in that hour. This
    follows the GHCNh documentation for METAR/SPECI precipitation running totals.
12. Stream accepted rows into one Snappy-compressed Parquet file.
13. Write audit manifests:
    - `station_year_manifest.csv`
    - `download_manifest.csv`
    - `processing_manifest.csv`
    - `reject_counts.csv`

## Improvements Over `pipelinev0.py`

`pipelinev0.py` was useful as an exploratory prototype, but it had several
issues that would make a full 2020-2022 CONUS run fragile.

| Area | v0 behavior | v1 improvement | Why it matters |
| --- | --- | --- | --- |
| Station selection | Bounding box only, with local ad hoc station filtering | Uses `ISO_CODE == "US"` plus CONUS bounding box and inventory station-years | Avoids non-US stations inside the box and avoids station-years with no inventory record |
| Downloads | `urlretrieve` with minimal status tracking | `requests` streaming, timeout, retries, 404/error manifests, atomic temp files | Makes large runs restartable and auditable |
| QC | Keeps `""`, `0`, `1`, `NaN` globally | Source-aware strict QC based on official GHCNh docs | `0` means "not checked" for some sources; strict QC avoids treating unchecked data as passed |
| Measurement codes | Converts only `T` to zero and keeps other flags | Converts documented trace codes and rejects accumulation/missing/deleted/estimated/incomplete flags | Prevents multi-hour or problematic observations from entering an hourly precipitation table |
| Time handling | Keeps the last row per minute and also creates an hour column | Keeps the last report per station-hour | NOAA documents METAR/SPECI precipitation as running hourly totals; the last report gives the hourly total |
| Deduplication | Groups only by minute, risking cross-station collisions | Deduplicates by `station_id + time_utc` | Preserves all stations while resolving within-hour reports correctly |
| Memory use | Buffers and concatenates chunks manually | Streams accepted chunks to a `pyarrow.ParquetWriter` | Avoids holding the full dataset in memory |
| Column compatibility | Assumes one column naming style | Supports both `Station_ID/Latitude/...` and `STATION/LATITUDE/...` | NOAA files vary across versions |
| Observability | Basic progress CSV | Download, processing, station-year, and reject-count manifests | Makes failures and filtering decisions inspectable |
| CLI | Hard-coded globals and commented control flow | Reusable CLI modes for inspect, dry-run, sample, download-only, process-only | Easier to test locally and run at full scale on Colab |

## Validation Already Performed

Local checks run with `source ~/.venvs/py313/bin/activate`:

- Python compile check passed.
- Manifest dry-run found `5,922` CONUS stations and `13,014` station-years for
  2020-2022.
- `--sample 5 --validate-output` produced `38,586` rows with no duplicate
  `(station_id, time_utc)` pairs.
- A METAR/SPECI sample station (`USW00094728`, 2020) converted `491` trace
  precipitation rows to zero and rejected bad QC codes.

## Notes

- `data/` is ignored by git because raw NOAA files and final Parquet outputs can
  become large.
- `src/ref_repo/` is ignored because it contains local reference repositories
  used during development, not the production pipeline.
- The default years are `2020 2021 2022`, but the CLI supports other years when
  present in the inventory.

