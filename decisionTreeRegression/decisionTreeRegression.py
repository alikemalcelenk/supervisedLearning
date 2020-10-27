#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:06:25 2020

@author: alikemalcelenk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decisionTreeRegression_dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%  decision tree regression
from sklearn.tree import DecisionTreeRegressor
treeRegression = DecisionTreeRegressor()
treeRegression.fit(x,y)

treeRegression.predict([[5.5]])
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
# bunu ekleyerek x değerlerini 0.01 aralıklarla sıkılaştırıyoruz ve grafikte zikzak yapısını oluşturmasını sağlıyoruz
y_head = treeRegression.predict(x_)

# %% visualize
plt.scatter(x,y,color="red") #dots
plt.plot(x_,y_head,color = "green")  #line
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
