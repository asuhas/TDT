# File: tests\test_obb_data.py

from datetime import datetime

import polars as pl
import pytest
from src.tdt.data.obb_data import get_treasury_yields


def test_get_treasury_yields_default():
    """Test get_treasury_yields with default parameters."""
    df = get_treasury_yields()
    unclean = get_treasury_yields(clean=False)
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] >= 0  # DataFrame should have zero or more rows
    assert "dt" in df.columns  # Ensure column 'dt' exists in DataFrame
    assert df.shape[0]< unclean.shape[0]


def test_get_treasury_yields_custom_date_range():
    """Test get_treasury_yields with a custom date range."""
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    df = get_treasury_yields(start_date=start_date, end_date=end_date)
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] >= 0
    if df.shape[0] > 0:
        assert df["dt"].min() >= start_date
        assert df["dt"].max() <= end_date

