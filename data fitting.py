# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:38:08 2023

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
    data1 = pd.read_csv(fn)
    data2 = data1.drop(columns=["Country Code","Indicator Name","Indicator Code"])
    print(data2)
    """
    replacing the nul vaules with 0
    """
    data2 = data2.replace(np.nan,0)
    u = ["Colombia","India"]
    """
    selecting the countries from the list 
    and check if dataframe contains specified value
    """
    data2 = data2["Country Name"].isin(u)
    data2 = data1[data2]
    """
    dropping some columns
    """
    data2 = data2.drop(columns={"Country Code","Indicator Name","Indicator Code","Country Name"})
    print(data2)
    """
    transposing 
    """
    dt = np.transpose(data2)
    dt = dt.reset_index()
    
    """
    renaming the data with index as year
    """
    dt = dt.rename(columns={"index":"year"})
    dt = dt.rename(columns={109:"INDIA",45:"COLOMBIA"})
    dt = dt.dropna()
    dt["COLOMBIA"] = pd.to_numeric(dt["COLOMBIA"])
    dt["INDIA"] = pd.to_numeric(dt["INDIA"])
    dt["year"] = pd.to_numeric(dt["year"])
    return data1,dt;
data,dt = read_data(r"C:\Users\BIJI PC\Desktop\data sci\ADS\assignment3\Forest sq.csv")
sdata,sdt = read_data(r"C:\Users\BIJI PC\Desktop\data sci\ADS\assignment3\CO2.csv")

"""
optimising the curve
"""
param,cp = opt.curve_fit(curve,dt["year"],dt["INDIA"],p0=[4e8, 0.1])
print(*param)
""" 
taking the error value
"""
sigma = np.sqrt(np.diag(cp))
"""
low and up values for error ranges
"""
low,up = err.err_ranges(dt["year"],curve,param,sigma)
"""
data fitting
"""
dt["fit"] = curve(dt["year"],*param)
plt.plot(dt["year"],dt["INDIA"],label="data")
plt.plot(dt["year"],dt["fit"],c="red",label="fit")
"""
plot the error ranges in the graph
"""
plt.fill_between(dt["year"],low,up,alpha=0.6)
plt.title("INDIA(FOREST AREA)")
plt.legend()
plt.show()