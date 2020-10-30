#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:17:35 2020

@author: alikemalcelenk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("randomForestRegression_dataset.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

# %%
from sklearn.ensemble import RandomForestRegressor
# random forest, ensemble learning üyesidir. Ensemble learning ML yöntemdir. Aynı anda pek çok ML algoritmasını kullanmaktır.
randomForest = RandomForestRegressor(n_estimators = 100, random_state = 42)
# n_estimators = tree number
# random_state = yapcağımız işlemin id si gibi bir şey demek.
# Şu an nasıl bölüyorsa başka zaman kullandığımızda random_state = 42 verirsek yine böyle bölecek
randomForest.fit(x,y)

print("7.8 seviyesinde fiyatın ne kadar olduğu: ",randomForest.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = randomForest.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()
