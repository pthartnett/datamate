import pytest
import numpy as np
import pandas as pd
from datamate import dataset

import statsmodels.api as sm
from scipy.stats import shapiro, skew, kurtosis
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.stats.outliers_influence import OLSInfluence

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
    else:
        raise NotImplementedError('Validation for '+analysis_type_to_validate+' is not implemented yet.')
