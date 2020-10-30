#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 00:05:32 2020

@author: alikemalcelenk
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multipleLinearRegression_dataset.csv",sep = ";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

# %% fitting data
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)

# predict
print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))
#iki tane örneği tahmin ettik. yaşlar 35, 10 yıl deneyim vs 5 yıl deneyim
