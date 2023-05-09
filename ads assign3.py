# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:17:44 2023

@author: BIJI PC
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn import cluster
import cluster_tools as ct
import errors as err
import numpy as np
import math

def curve(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1960))
    return f
def read_data(fn):
    """ 
    This function read data 
    return transposed value and original value
    """
    df = pd.read_csv(fn)
    dd = df.drop(columns=["Country Code","Indicator Name","Indicator Code"])
    print(dd)
    """
    replacing the nul vaules with 0
    """
    dd = dd.replace(np.nan,0)
    u = ["Colombia","India"]
    """
    selecting the countries from the list
    """
    dd = dd["Country Name"].isin(u)
    dd = df[dd]
    """
    dropping some columns
    """
    dd = dd.drop(columns={"Country Code","Indicator Name","Indicator Code","Country Name"})
    print(dd)
    """
    transposing 
    """
    dt = np.transpose(dd)
    dt = dt.reset_index()
    
    