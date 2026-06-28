import pandas as pd
import numpy as np
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# =========================
# CONFIG
# =========================

YEARS = range(2020, 2023)

# spatial filter
LAT_MIN = 24.00
LAT_MAX = 50.00
LON_MIN = -125
LON_MAX = -65

STATION_FILE = "stations.csv"

OUT_DIR = Path("station_dataset")
OUT_DIR.mkdir(exist_ok=True)

CHUNK_DIR = OUT_DIR / "chunks"
CHUNK_DIR.mkdir(exist_ok=True)

RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

# Save one chunk every N successful station-year results.
SAVE_EVERY = 500

# If True, download raw parquet to local disk first, then process locally.
DOWNLOAD_RAW_FIRST = True

PROGRESS_FILE = OUT_DIR / "processed_station_years.csv"

BASE_URL = (
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
    "hourly/access/by-year/{year}/parquet/GHCNh_{station}_{year}.parquet"
)

GOOD_QC = ["", "0", "1", np.nan]

# =========================
# LOAD STATION LIST
# =========================

stations = pd.read_csv(STATION_FILE)

stations = stations[
    (stations.LATITUDE >= LAT_MIN)
    & (stations.LATITUDE <= LAT_MAX)
    & (stations.LONGITUDE >= LON_MIN)
    & (stations.LONGITUDE <= LON_MAX)
    & (stations.GHCN_ID.notna())
]

station_ids = stations.GHCN_ID.tolist()

print("Stations in bbox:", len(station_ids))

# 新添加
import os
files = os.listdir('./station_dataset/raw')
stat = set()
for name in files:
    name_parts = name.split('_')
    if len(name_parts) >= 3:
        stat.add(name_parts[1])
print(len(stat))

station_ids = list(stat)
print("Stations with raw data:", len(station_ids))
# 到此为止
# =========================
# PROCESS STATION
# =========================


def raw_path(station_id, year):
    return RAW_DIR / f"GHCNh_{station_id}_{year}.parquet"


def get_parquet_source(station_id, year):
    url = BASE_URL.format(station=station_id, year=year)

    if not DOWNLOAD_RAW_FIRST:
        return url

    local_file = raw_path(station_id, year)
    #print(f"Checking local file: {local_file}")
    if local_file.exists():
        return local_file
    print(f"Downloading: {url} to {local_file}")
    urlretrieve(url, local_file)
    return local_file


def process_station_year(station_id, year):

    try:
        source = get_parquet_source(station_id, year)
        df = pd.read_parquet(source)
    except (HTTPError, URLError, OSError, ValueError):
        print(f"Failed to load data for {station_id} in {year}")
        return None

    if "precipitation" not in df.columns:
        print(f"No precipitation data for {station_id} in {year}")
        return None

    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    if "Station_ID" in df.columns:
        df = df[
            [
                "Station_ID",
                "DATE",
                "Latitude",
                "Longitude",
                "Elevation",
                "precipitation",
                "precipitation_Measurement_Code",
                "precipitation_Quality_Code",
            ]
        ]
        df["Station_ID"] = station_id
        df["Latitude"] = stations.set_index("GHCN_ID").loc[station_id, "LATITUDE"]
        df["Longitude"] = stations.set_index("GHCN_ID").loc[station_id, "LONGITUDE"]
        df["Elevation"] = stations.set_index("GHCN_ID").loc[station_id, "ELEVATION"]
    else:
        df = df[
            [
                "STATION",
                "DATE",
                "LATITUDE",
                "LONGITUDE",
                "ELEVATION",
                "precipitation",
                "precipitation_Measurement_Code",
                "precipitation_Quality_Code",
            ]
        ]
        df["Station_ID"] = station_id
        df["Latitude"] = stations.set_index("GHCN_ID").loc[station_id, "LATITUDE"]
        df["Longitude"] = stations.set_index("GHCN_ID").loc[station_id, "LONGITUDE"]
        df["Elevation"] = stations.set_index("GHCN_ID").loc[station_id, "ELEVATION"]

    


    #print("hello")
    #print(stations.set_index("GHCN_ID").loc[station_id])
    # remove missing
    df = df[df.precipitation.notna()]

    # trace → 0
    trace = df.precipitation_Measurement_Code == "T"
    df.loc[trace, "precipitation"] = 0.0

    # QC filter
    df = df[df.precipitation_Quality_Code.isin(GOOD_QC)]

    # hour bin
    df["hour"] = df.DATE.dt.floor("h")
    df["time(min)"] = df.DATE.dt.floor("min")

    df = df.sort_values("DATE")

    df = df.groupby("time(min)", as_index=False).tail(1)

    df = df.rename(
        columns={
            "Station_ID": "station_id",
            "hour": "time_utc",
            "Latitude": "lat",
            "Longitude": "lon",
            "Elevation": "elevation_m",
            "precipitation": "precip_mm",
        }
    )
    #print(df[["station_id", "time_utc", "lat", "lon", "elevation_m", "precip_mm"]].head())
    return df[["station_id", "time_utc","time(min)", "lat", "lon", "elevation_m", "precip_mm"]]


def load_processed_keys():
    if not PROGRESS_FILE.exists():
        return set()

    done = pd.read_csv(PROGRESS_FILE)
    if done.empty:
        return set()

    return set(zip(done["station_id"], done["year"]))


def append_progress(rows):
    progress_df = pd.DataFrame(rows, columns=["station_id", "year"])
    write_header = not PROGRESS_FILE.exists()
    progress_df.to_csv(PROGRESS_FILE, mode="a", header=write_header, index=False)


def flush_buffer(buffer, chunk_idx):
    if not buffer:
        return chunk_idx

    chunk_file = CHUNK_DIR / f"part_{chunk_idx:05d}.parquet"
    pd.concat(buffer, ignore_index=True).to_parquet(chunk_file)
    print(f"Saved chunk: {chunk_file}")
    return chunk_idx + 1


# =========================
# BUILD DATASET
# =========================

# done_keys = load_processed_keys()
# if done_keys:
#     print("Already processed station-years:", len(done_keys))

existing_parts = sorted(CHUNK_DIR.glob("part_*.parquet"))
next_chunk_idx = len(existing_parts)

buffer = []
new_progress_rows = []
success_count = 0
#skip_station = [("USC00128967", 2021), ("USC00410779", 2021), ("USC00416136", 2021), ("USC00416104", 2021), ("USC00153194", 2021), ("USC00304102", 2021), ("USC00465002", 2021), ("USC00250622",2021), ("USC00409493", 2021),("USC00294862",2021),("USC00157215",2021), ("USC00305426", 2021), ("USC00422256", 2021)]
for station in station_ids:
    for year in YEARS:

        key = (station, year)
        # if key in done_keys:
        #     continue

        #print(station, year)
        # if key in skip_station:
        #     print(f"Skipping known bad station-year: {key}")
        #     # rename the header
        #     continue
        df = process_station_year(station, year)

        if df is None:
            continue

        buffer.append(df)
        new_progress_rows.append(key)
        success_count += 1

        if success_count % SAVE_EVERY == 0:
            print(f"Remaining: {len(station_ids) - station_ids.index(station)} stations, year: {year}, success count: {success_count}")
            next_chunk_idx = flush_buffer(buffer, next_chunk_idx)
            buffer = []
            append_progress(new_progress_rows)
            new_progress_rows = []

if new_progress_rows:
    append_progress(new_progress_rows)

next_chunk_idx = flush_buffer(buffer, next_chunk_idx)

all_parts = sorted(CHUNK_DIR.glob("part_*.parquet"))
if not all_parts:
    print("No data collected. Exiting.")
    exit(1)

dataset = pd.concat([pd.read_parquet(p) for p in all_parts], ignore_index=True)
dataset = dataset.sort_values("time_utc")

out_file = OUT_DIR / "station_precip_ml_dataset.parquet"

dataset.to_parquet(out_file)

print("Saved dataset:", out_file)
print("Rows:", len(dataset))
