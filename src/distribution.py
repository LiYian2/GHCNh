import pandas as pd
import pyarrow.parquet as pq

path = "/Users/leo/Downloads/GHCNh/data/final/ghcnh_conus_precip_2020_2022_compact.parquet"
df = pq.ParquetFile(path)

stations = set()
minute_count = pd.Series()

for rg in range(df.num_row_groups):
    row_group = df.read_row_group(rg)
    stations.update(row_group["station_id"].unique())
    minute_count = minute_count.add(row_group["precipitation"].value_counts(), fill_value=0)
