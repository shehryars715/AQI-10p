"""Unit tests for feature engineering transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline.transformer import (
    AQI_CATEGORIES,
    aqi_category,
    compute_aqi_category_col,
    compute_change_rate,
    compute_lag_features,
    compute_rolling_features,
    compute_time_features,
    transform,
)


class TestAQICategory:
    @pytest.mark.parametrize(
        "aqi,expected",
        [
            (0, "Good"),
            (50, "Good"),
            (51, "Moderate"),
            (100, "Moderate"),
            (101, "Unhealthy for Sensitive Groups"),
            (150, "Unhealthy for Sensitive Groups"),
            (151, "Unhealthy"),
            (200, "Unhealthy"),
            (201, "Very Unhealthy"),
            (300, "Very Unhealthy"),
            (301, "Hazardous"),
            (500, "Hazardous"),
        ],
    )
    def test_boundaries(self, aqi, expected):
        assert aqi_category(aqi) == expected

    def test_six_categories(self):
        labels = {label for _, _, label in AQI_CATEGORIES}
        assert len(labels) == 6


class TestTimeFeatures:
    def test_basic_columns(self):
        df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2026-06-03 14:30:00", "2026-01-01 08:00:00"])}
        )
        result = compute_time_features(df)
        assert result.loc[0, "hour"] == 14
        assert result.loc[1, "hour"] == 8
        assert result.loc[0, "month"] == 6
        assert result.loc[0, "day_of_week"] == 2  # Wednesday

    def test_seasons(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-15", "2026-04-15", "2026-07-15", "2026-10-15"]
                )
            }
        )
        result = compute_time_features(df)
        assert result.loc[0, "season"] == "Winter"
        assert result.loc[1, "season"] == "Summer"
        assert result.loc[2, "season"] == "Monsoon"
        assert result.loc[3, "season"] == "Autumn"

    def test_weekend_flag(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-05-30", "2026-06-01"]  # Saturday, Monday
                )
            }
        )
        result = compute_time_features(df)
        assert result.loc[0, "is_weekend"] == 1
        assert result.loc[1, "is_weekend"] == 0


class TestLagFeatures:
    def test_single_city(self, sample_historical_df):
        df = sample_historical_df.copy()
        result = compute_lag_features(df)

        # First row has no lags
        assert pd.isna(result.loc[0, "aqi_lag_1h"])
        # 25th row (t+24h) should have lag_24h defined
        assert not pd.isna(result.loc[24, "aqi_lag_24h"])

    def test_multi_city_isolation(self):
        df1 = pd.DataFrame(
            {
                "city": "A",
                "timestamp": pd.date_range("2026-06-01", periods=12, freq="h"),
                "aqi": range(100, 112),
            }
        )
        df2 = pd.DataFrame(
            {
                "city": "B",
                "timestamp": pd.date_range("2026-06-01", periods=12, freq="h"),
                "aqi": range(200, 212),
            }
        )
        combined = pd.concat([df1, df2], ignore_index=True)
        result = compute_lag_features(combined)

        # City A's first row lag should reference City A's prior value
        # (since there's only 1 row per city per hour, lag is NaN for first)
        row_a1 = result[result.city == "A"].iloc[0]
        assert pd.isna(row_a1["aqi_lag_1h"])
        # City B's first row should also be NaN (not picking up City A's value)
        row_b1 = result[result.city == "B"].iloc[0]
        assert pd.isna(row_b1["aqi_lag_1h"])


class TestRollingFeatures:
    def test_rolling_mean(self):
        df = pd.DataFrame(
            {
                "city": ["X"] * 10,
                "timestamp": pd.date_range("2026-06-01", periods=10, freq="h"),
                "aqi": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            }
        )
        result = compute_rolling_features(df, windows=(3,))
        # Rolling mean over 3h at index 2
        expected = (100 + 110 + 120) / 3
        assert result.loc[2, "aqi_rolling_mean_3h"] == pytest.approx(expected)


class TestChangeRate:
    def test_positive_change(self):
        df = pd.DataFrame(
            {"aqi": [150], "aqi_lag_1h": [140]}
        )
        result = compute_change_rate(df)
        assert result.loc[0, "aqi_change_rate_1h"] == pytest.approx(
            (150 - 140) / 140
        )

    def test_zero_lag_avoids_division(self):
        df = pd.DataFrame({"aqi": [100], "aqi_lag_1h": [0]})
        result = compute_change_rate(df)
        assert pd.isna(result.loc[0, "aqi_change_rate_1h"])


class TestTransformPipeline:
    def test_returns_expected_columns(self, sample_raw_aqi_df, sample_raw_weather_df):
        result = transform(sample_raw_aqi_df, sample_raw_weather_df)

        expected_cols = {
            "city", "timestamp", "aqi", "hour", "day_of_week", "month",
            "season", "is_weekend", "aqi_lag_1h", "aqi_lag_6h",
            "aqi_lag_24h", "aqi_lag_48h", "aqi_rolling_mean_6h",
            "aqi_rolling_mean_24h", "aqi_change_rate_1h", "aqi_category",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_empty_input(self, sample_raw_weather_df):
        result = transform(pd.DataFrame(), sample_raw_weather_df)
        assert result.empty

    def test_with_historical_context(self, sample_historical_df, sample_raw_aqi_df):
        result = transform(
            sample_raw_aqi_df,
            pd.DataFrame(),  # empty weather — only AQI features matter here
            hist_df=sample_historical_df,
        )
        # With 72 h of history, lags for the new row should be populated
        row = result.iloc[0]
        assert not pd.isna(row["aqi_lag_1h"])
        assert not pd.isna(row["aqi_lag_24h"])
