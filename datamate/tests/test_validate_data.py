import pytest
import pandas as pd
import numpy as np
from datamate import dataset
from datamate.validate_data import validate_data

def test_ols_no_violations():
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 4],
        "x1": [1, 2, 3, 4, 5],
        "x2": [2, 4, 6, 8, 10]  # Perfectly collinear
    })
    ds = dataset(df, dependent_var="y", independent_vars=["x1"], analysis_type="ols")
    results = validate_data(ds)

    assert "Multicollinearity (VIF)" in results
    assert results["Durbin-Watson"] is not None
    assert results["Shapiro-Wilk p-val"] is not None
    assert results["Breusch Pagan p-val"] is not None
    assert results["Skewness"] is not None
    assert results["Kurtosis"] is not None

def test_multicollinearity():
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 4.9],
        "x1": [1, 2, 3, 4, 5],
        "x2": [2, 4, 6, 8, 10.1]  # Perfectly collinear
    })
    ds = dataset(df, dependent_var="y", independent_vars=["x1", "x2"], analysis_type="ols")
    results = validate_data(ds)

    # Expecting VIF > 10 for collinear variables
    assert results["Multicollinearity (VIF)"]["x1"] > 10
    assert results["Multicollinearity (VIF)"]["x2"] > 10

def test_non_normal_residuals():
    df = pd.DataFrame({
        "y": [1, 1, 1, 1000, 1],  # Extreme outlier
        "x1": [1, 200, 3, 4, 5]
    })
    ds = dataset(df, dependent_var="y", independent_vars=["x1"], analysis_type="ols")
    results = validate_data(ds)

    assert results["Shapiro-Wilk p-val"] < 0.05  

def test_high_leverage_points():
    df = pd.DataFrame({
        "y": [10, 20, 30, 40, 500],  # Outlier in last row
        "x1": [1, 2, 3, 4, 100]  # High leverage in last row
    })
    ds = dataset(df, dependent_var="y", independent_vars=["x1"], analysis_type="ols")
    results = validate_data(ds)

    assert results["Influential Points (Cook's Distance > 4/n)"] > 0

