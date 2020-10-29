#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:38:01 2020

@author: alikemalcelenk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("randomForestRegression_dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor(n_estimators = 100, random_state = 42)
randomForest.fit(x,y)

y_head = randomForest.predict(x)

# %% evaluation
from sklearn.metrics import r2_score
print("r2_score: ", r2_score(y,y_head))
# r^2(r2_score) değeri 1 e ne kadar yakın olursa o kadar iyi predict etmişiz demektir.
