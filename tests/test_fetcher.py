"""Unit tests for the AQICN / OpenWeather API fetcher."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from src.feature_pipeline.fetcher import (
    fetch_aqicn,
    fetch_all_cities,
    fetch_openweather,
)


class TestFetchAQICN:
    def test_returns_dataframe(self, mock_requests_get, sample_aqicn_response):
        mock_requests_get.return_value.json.return_value = sample_aqicn_response
        mock_requests_get.return_value.raise_for_status = lambda: None
        mock_requests_get.return_value.status_code = 200

        # Patch the settings to have a dummy token
        import src.feature_pipeline.fetcher as m
        m._validate_api_key = lambda key, name: "dummy-token"

        df = fetch_aqicn("Delhi")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.loc[0, "city"] == "Delhi"
        assert df.loc[0, "aqi"] == 155
        assert df.loc[0, "pm25"] == 68.2

    def test_raises_on_error_status(self, mock_requests_get):
        mock_requests_get.return_value.json.return_value = {
            "status": "error",
            "data": "Invalid key",
        }
        mock_requests_get.return_value.raise_for_status = lambda: None

        import src.feature_pipeline.fetcher as m
        m._validate_api_key = lambda key, name: "dummy-token"

        with pytest.raises(RuntimeError, match="Invalid key"):
            fetch_aqicn("Delhi")

    def test_missing_pollutants_become_none(self, mock_requests_get):
        """If a pollutant is missing from iaqi, it should be NaN."""
        resp = {
            "status": "ok",
            "data": {
                "aqi": 50,
                "city": {"name": "Clean"},
                "iaqi": {"pm25": {"v": 12.0}},  # only PM2.5
                "time": {"s": "2026-06-03 10:00:00"},
            },
        }
        mock_requests_get.return_value.json.return_value = resp
        mock_requests_get.return_value.raise_for_status = lambda: None

        import src.feature_pipeline.fetcher as m
        m._validate_api_key = lambda key, name: "dummy-token"

        df = fetch_aqicn("Clean")
        assert pd.isna(df.loc[0, "pm10"])
        assert pd.isna(df.loc[0, "o3"])
        assert df.loc[0, "pm25"] == 12.0


class TestFetchOpenWeather:
    def test_returns_dataframe(self, mock_requests_get, sample_openweather_response):
        mock_requests_get.return_value.json.return_value = sample_openweather_response
        mock_requests_get.return_value.raise_for_status = lambda: None

        import src.feature_pipeline.fetcher as m
        m._validate_api_key = lambda key, name: "dummy-key"

        df = fetch_openweather("Delhi")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.loc[0, "temperature"] == 32.5
        assert df.loc[0, "weather_main"] == "Haze"


class TestFetchAllCities:
    def test_partial_failure_does_not_crash(self, mock_requests_get):
        """If one city fails, the others should still be returned."""
        import src.feature_pipeline.fetcher as m

        m._validate_api_key = lambda key, name: "dummy"

        call_count = 0

        def side_effect(url, params, timeout):
            nonlocal call_count
            call_count += 1
            resp = mock_requests_get.return_value
            if "Mumbai" in url:
                raise requests.exceptions.ConnectionError("timeout")
            resp.json.return_value = {
                "status": "ok",
                "data": {
                    "aqi": 100,
                    "city": {"name": "Delhi"},
                    "iaqi": {},
                    "time": {"s": "2026-06-03 10:00:00"},
                },
            }
            return resp

        mock_requests_get.side_effect = side_effect
        mock_requests_get.return_value.raise_for_status = lambda: None
        mock_requests_get.return_value.status_code = 200

        aqi_df, _ = fetch_all_cities(["Delhi", "Mumbai"])
        # Delhi should succeed, Mumbai failure should be logged but not crash
        assert len(aqi_df) >= 1
