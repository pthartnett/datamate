#dataset class stores relevant metadata for statistical tests

import numpy as np
import pandas as pd

class dataset:
    def __init__(self, data, dependent_var=None, independent_vars = None, analysis_type = "ols"):
        '''
        Initialized a dataset object for use within the datamate package
        
        Parameters:
            data (pandas DataFrame, numpy array, list of lists for now): The data for analysis
            dependent_var (str, int, float, tuple, list, optional): The dependent variable
            independent_vars (str, int, float, tuple, list, optional): The independent variable name(s), defaults to non-dependendent variable columns
            analysis_type (str, optional): Type of analysis (e.g., "ols").
        '''

        self.analysis_type = analysis_type
        self.original_format = type(data)
        self.dependent_var = [dependent_var] if isinstance(dependent_var, str) else dependent_var
        self.independent_vars = [independent_vars] if isinstance(independent_vars, str) else independent_vars

        #for now work with pandas df, can change to make faster later
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        elif isinstance(data, np.ndarray):
            self.df = pd.DataFrame(data)
        elif isinstance(data, list):
            self.df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported data type. Must be pandas dataframe, numpy array, or list.")
        
        variables_to_keep = set(self.target or []) | set(self.independent_vars or [])
        if all(isinstance(var, int) for var in variables_to_keep):
            df = df.iloc[:, list(variables_to_keep)]
        else:
            df = df[list(variables_to_keep)]