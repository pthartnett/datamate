import pytest
import numpy as np
import pandas as pd
from datamate import dataset

def test_dataframe_named_cols_input():
    df = pd.DataFrame([[1,2,3,4],[2,3,5,7],[3,3,None,5]],columns=['a','b','c','d'])
    ds = dataset(df, dependent_var = 'a', independent_vars=['b','c'])

    assert ds.df.shape==(3,3)
    assert set(ds.df.columns)=={'a','b','c'}
    assert ds.y is not None
    assert ds.x is not None

def test_dataframe_no_named_cols_input():
    df = pd.DataFrame([[1,2,3,4],[2,3,5,7],[3,3,None,5]])
    ds = dataset(df, dependent_var = 0, independent_vars=[1,2])

    assert ds.df.shape==(3,3)
    assert set(ds.df.columns)=={0,1,2}
    assert ds.y is not None
    assert ds.x is not None
