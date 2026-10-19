from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelinev1 import StationYear  # noqa: E402
from weather_pipeline import (  # noqa: E402
    acceptable_quality,
    read_weather_frame,
    process_station_year,
)


class WeatherPipelineTest(unittest.TestCase):
    def test_source_aware_quality_policy(self):
        self.assertTrue(acceptable_quality("223", "1"))
        self.assertFalse(acceptable_quality("223", "0"))
        self.assertFalse(acceptable_quality("223", "5"))
        self.assertTrue(acceptable_quality("343", "5"))
        self.assertTrue(acceptable_quality("343", "A"))
        self.assertFalse(acceptable_quality("343", "2"))
        self.assertFalse(acceptable_quality("343", "s"))

    def test_hourly_last_valid_and_weather_flags(self):
        frame = pd.DataFrame(
            {
                "STATION": ["USWTEST", "USWTEST", "USWTEST"],
                "DATE": [
                    "2020-01-01T00:10:00Z",
                    "2020-01-01T00:50:00Z",
                    "2020-01-01T01:05:00Z",
                ],
                "temperature": [10.0, None, 11.0],
                "temperature_Quality_Code": ["5", None, "2"],
                "temperature_Report_Type": ["FM15", None, "FM15"],
                "temperature_Source_Code": ["343", None, "343"],
                "wind_direction": ["999", None, "180"],
                "wind_direction_Measurement_Code": ["V", None, "N"],
                "wind_direction_Quality_Code": ["5", None, "5"],
                "wind_direction_Report_Type": ["FM15", None, "FM15"],
                "wind_direction_Source_Code": ["343", None, "343"],
                "precipitation": [0.2, None, 1.0],
                "precipitation_Measurement_Code": ["T", None, "3"],
                "precipitation_Quality_Code": ["5", None, "5"],
                "precipitation_Report_Type": ["FM15", None, "FM15"],
                "precipitation_Source_Code": ["343", None, "343"],
            }
        )
        item = StationYear("USWTEST", 2020, 40.0, -75.0, 10.0, "Test", "NY")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            frame.to_parquet(path, index=False)
            result, manifest, _ = process_station_year(item, path)

        self.assertEqual(manifest.status, "ok")
        self.assertEqual(len(result), 2)
        row = result.iloc[0]
        self.assertEqual(row["temperature_c"], 10.0)
        self.assertTrue(pd.isna(row["wind_direction_deg"]))
        self.assertTrue(row["wind_variable"])
        self.assertEqual(row["precip_1h_mm"], 0.0)
        self.assertTrue(row["precip_trace"])
        second = result.iloc[1]
        self.assertTrue(pd.isna(second["temperature_c"]))
        self.assertEqual(second["wind_direction_deg"], 180.0)
        self.assertTrue(pd.isna(second["precip_1h_mm"]))

    def test_read_skips_columns_proven_all_null(self):
        frame = pd.DataFrame(
            {
                "STATION": ["USCTEST"],
                "DATE": ["2020-01-01T00:00:00Z"],
                "temperature": [None],
                "temperature_Quality_Code": [None],
                "precipitation": [1.5],
                "precipitation_Quality_Code": ["5"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            frame.to_parquet(path, index=False)
            result, error = read_weather_frame(path)

        self.assertEqual(error, "")
        self.assertNotIn("temperature", result.columns)
        self.assertNotIn("temperature_Quality_Code", result.columns)
        self.assertIn("precipitation", result.columns)
        self.assertIn("precipitation_Quality_Code", result.columns)


if __name__ == "__main__":
    unittest.main()
