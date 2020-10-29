#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:44:12 2020

@author: alikemalcelenk
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("linearRegression_dataset.csv",sep = ";")

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
 
from sklearn.linear_model import LinearRegression
LinearRegression = LinearRegression()
LinearRegression.fit(x,y)

y_head = LinearRegression.predict(x)  

#%% evaluation
from sklearn.metrics import r2_score
print("r2_score: ", r2_score(y,y_head))
# r^2(r2_score) değeri 1 e ne kadar yakın olursa o kadar iyi predict etmişiz demektir.
