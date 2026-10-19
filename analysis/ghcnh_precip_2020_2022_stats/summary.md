# GHCNh CONUS Precipitation 2020-2022 Compact Dataset Stats

- Input: `data/final/ghcnh_conus_precip_2020_2022_compact.parquet`
- Rows: 60,008,899
- Stations: 3,687
- Time span UTC: 2020-01-01 00:00:00+00:00 to 2022-12-31 23:59:00+00:00
- Zero precipitation rows: 55,159,839 (91.92%)
- Positive precipitation rows: 4,849,060 (8.08%)
- Precipitation max: 990.60 mm; p99: 3.00 mm; p999: 13.50 mm
- Station unique-hour median: 24,078; p95: 26,299; max: 26,304
- Most common report minute: 0 (58.42%)
- Most common report UTC hour: 21 (4.17%)

## Top 10 Stations by Rows

| station_id   |   rows |   unique_hours |     lat |       lon |   positive_precip_share |   max_precip_mm |
|:-------------|-------:|---------------:|--------:|----------:|------------------------:|----------------:|
| USC00214721  |  26304 |          26304 | 44.4698 |  -93.9025 |               0.0510189 |            43.7 |
| USC00325993  |  26304 |          26304 | 48.1802 | -101.296  |               0.0516651 |            18.5 |
| USC00337383  |  26304 |          26304 | 40.5449 |  -84.4374 |               0.0761861 |            50   |
| USC00134963  |  26304 |          26304 | 41.8604 |  -90.9222 |               0.0765663 |            35.3 |
| USC00110082  |  26304 |          26304 | 41.0579 |  -90.5654 |               0.0751977 |            44.5 |
| USC00130364  |  26304 |          26304 | 41.4175 |  -95.0041 |               0.0524635 |            36.6 |
| USC00207407  |  26304 |          26304 | 43.9331 |  -86.2453 |               0.0878954 |            58.7 |
| USC00242827  |  26304 |          26304 | 48.8978 | -115.064  |               0.0606752 |            15   |
| USC00427026  |  26304 |          26304 | 39.5991 | -110.819  |               0.029007  |            14.2 |
| USC00042899  |  26304 |          26304 | 41.4574 | -122.899  |               0.0636405 |            32   |

## Output Files

- `overall_summary.csv`, `precip_summary.csv`, `station_summary.csv`
- `precip_bin_distribution.csv`, `report_hour_distribution.csv`, `report_minute_distribution.csv`
- `station_stats.csv`, `station_geo_1deg_grid.csv`, `code_report_type_distribution.csv`
- PNG charts in the same directory.