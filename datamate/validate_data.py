import pytest
import numpy as np
import pandas as pd
from datamate import dataset

import statsmodels.api as sm
from scipy.stats import shapiro, skew, kurtosis
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

def validate_data_ols(ds):
    '''
    Validates whether dataset meets assumptions for OLS regression

    Parameters:
        ds (dataset): Instance of dataset class from dataset.py with "ols" analysis type
    Returns:
        dict: Dictionary containing results
    '''
    results = {}
    if ds.x is None or ds.y is None:
        raise ValueError('Check if both independent and dependent variables are specified')
    x_plus_const = sm.add_constant(ds.x)

    model = sm.OLS(ds.y,x_plus_const).fit()
    residuals = model.resid
    
    #Summary
    results['Summary'] = model.summary()

    #Linearity 
    vif_data = {ds.x.columns[i]: variance_inflation_factor(x_plus_const.values, i+1) for i in range(ds.x.shape[1])}
    results['Multicollinearity (VIF)'] = vif_data

    #Homoscedasticity
    _,results['Breusch Pagan p-val'],_,_ = het_breuschpagan(residuals, x_plus_const)

    #Normality of Residuals
    _,results['Shapiro-Wilk p-val'] = shapiro(residuals)

    #Autocorrelation
    results['Durbin-Watson'] = sm.stats.stattools.durbin_watson(residuals)

    #Skew
    results["Skewness"] = skew(residuals)

    #Kurtosis
    results["Kurtosis"] = kurtosis(residuals, fisher=True)

    # Standardized Residuals (Outlier Detection)
    influence = OLSInfluence(model)
    standardized_residuals = influence.resid_studentized_internal
    num_outliers = sum(abs(standardized_residuals) > 3)

    # Cook's Distance (Leverage Detection)
    cooks_d = influence.cooks_distance[0]
    high_leverage_points = sum(cooks_d > 4 / len(ds.x)) 

    results["Outliers (Std Resids over +/- 3)"] = num_outliers
    results["Influential Points (Cook's Distance > 4/n)"] = high_leverage_points

    return results

def check_stationarity(series, max_diffs=2):
    """
    Tests if a series is stationary using ADF test. If not, checks if differencing or log transformation helps

    Parameters:
        series (pd.Series): Time series to check
        max_diffs (int): Maximum number of differencing steps to try

    Returns:
        dict: Results for stationarity tests
    """
    results = {}
    
    #Step 1: Raw Data Stationarity Test
    adf_stat, adf_pval, _, _, critical_values, _ = adfuller(series.dropna())
    results["Original ADF p-value"] = adf_pval
    results["Stationary"] = adf_pval < 0.05

    #Step 2: Differencing
    if not results["Stationary"]:
        for d in range(1, max_diffs + 1):
            diff_series = series.diff(periods=d).dropna()
            adf_stat, adf_pval, _, _, _, _ = adfuller(diff_series)
            if adf_pval < 0.05:
                results["Differencing Required"] = d
                results["Stationary After Differencing"] = True
                break
        else:
            results["Stationary After Differencing"] = False

    #Step 3: Log Transformation
    if not results["Stationary"] and (series > 0).all():
        log_series = np.log(series).dropna()
        adf_stat, adf_pval, _, _, _, _ = adfuller(log_series)
        results["Log Transform Stationary"] = adf_pval < 0.05

    return results

def select_ar_ma(series, nlags=30):
    """
    Selects optimal AR and MA using PACF and ACF

    Parameters:
        series (pd.Series): Stationary time series.
        nlags (int): Number of lags to check.

    Returns:
        dict: {'ar': [list of significant AR lags], 'ma': [list of significant MA lags]}.
    """
    acf_vals = acf(series, nlags=nlags, fft=False)
    pacf_vals = pacf(series, nlags=nlags)

    #Significance threshold (2/sqrt(N))
    significance_threshold = 2 / np.sqrt(len(series))

    #Select all significant lags for AR (PACF) and MA (ACF)
    ar_lags = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > significance_threshold]
    ma_lags = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > significance_threshold]

    return {"ar": ar_lags, "ma": ma_lags}

def validate_data_arima(ds, nlags = 30):
    """
    Validates whether a dataset meets assumptions for ARIMA time series modeling

    Parameters:
        ds (dataset): dataset with "arima" analysis type

    Returns:
        dict: Dictionary containing validation results
    """
    results = {}
    stationary = {}

    if ds.y is None:
        raise ValueError("Time series data must be provided.")

    y_squeeze = ds.y.squeeze()  # Convert DataFrame to Series if necessary

    #Stationary
    for col in [ds.y.name] + list(ds.x.columns):
        stationary[col] = check_stationarity(ds.df[col])
    results["Stationary"] = stationary

    #Optional differences for y variable
    stationarity_results = check_stationarity(y_squeeze)
    d = stationarity_results["Differencing Required"]

    y_diff = y_squeeze.diff(periods=d).dropna() if d > 0 else y_squeeze

    #Find All Significant AR and MA terms
    ar_ma_lags = select_ar_ma(y_diff, nlags=nlags)

    results["Optimal ARIMA (dependent)"] = (ar_ma_lags['ar'], d, ar_ma_lags['ma'])
    results["Stationarity"] = stationarity_results

    #Fit ARIMA Model with Multiple AR & MA Terms
    residuals_init = y_diff - y_diff.mean()

    arma_terms = pd.DataFrame({'y': y_diff})
    for lag in ar_ma_lags['ar']:
        arma_terms[f'lag_y_{lag}'] = y_diff.shift(lag)
    for arma_terms in ar_ma_lags['ma']:
        arma_terms[f'lag_resid_{lag}'] = residuals_init.shift(lag)
    arma_terms = arma_terms.dropna()

    X = sm.add_constant(arma_terms.drop(columns='y'))
    arima_model = sm.OLS(arma_terms['y'], X).fit()
    #arima_model = sm.tsa.ARIMA(y_squeeze, order=(max(ar_ma_lags['ar'], default=0), d, max(ar_ma_lags['ma'], default=0))).fit()
    residuals = arima_model.resid

    #Ljung-Box Test for Residual Autocorrelation
    lb_stat, lb_pval = acorr_ljungbox(residuals, lags=[10], return_df=False)
    results["Ljung-Box p-value (dependent)"] = lb_pval[0]

    #Shapiro-Wilk Test for Residual Normality
    _, shapiro_pval = shapiro(residuals)
    results["Shapiro-Wilk p-value (dependent)"] = shapiro_pval

    #Autocorrelation Function (ACF) on Residuals
    results["Auto-correlation (ACF) (dependent)"] = acf(residuals, nlags=10, fft=False).tolist()

    #Summary
    results['Summary'] = arima_model.summary()

    #AIC/BIC
    results["AIC"] = arima_model.aic
    results["BIC"] = arima_model.bic



    return results



def validate_data(ds):
    '''
    Validates whether dataset meets assumptions for its analysis type

    Parameters:
        ds (dataset): Instance of dataset class from dataset.py
    Returns:
        dict: Dictionary containing results
    '''

    analysis_type_to_validate = ds.analysis_type
    if analysis_type_to_validate == 'ols':
        return validate_data_ols(ds)
    elif analysis_type_to_validate == 'arima':
        return validate_data_arima(ds)
    else:
        raise NotImplementedError('Validation for '+analysis_type_to_validate+' is not implemented yet.')
