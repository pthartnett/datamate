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

        

        self.df = df
        self.y = df[self.dependent_var] if self.dependent_var else None
        self.x = df[self.independent_vars]

        def return_original_format(self):
            if self.original_format == pd.DataFrame:
                return self.df
            elif self.original_format == np.ndarray:
                return self.df.to_numpy()
            elif self.original_format == list:
                return self.df.values.tolist()
            else:
                raise ValueError('Unexpected original format: '+str(self.return_original_format))
            
        def summary(self):
            return {
                'analysis_type':self.analysis_type,
                'original_format':self.original_format,
                'dependent_var':self.dependent_var,
                'independent_vars':self.independent_vars,
                'num_rows':self.df.shape[0],
                'num_columns':self.df.shape[1],
                'num_missing':self.df.isnull().sum().sum(),
                'num_inf': (self.df == np.inf).sum().sum() + (self.df == -np.inf).sum().sum(),
                'columns_with_missing': self.df.loc[:,self.df.isnull().sum()>0].isnull().sum().to_dict(),
            }