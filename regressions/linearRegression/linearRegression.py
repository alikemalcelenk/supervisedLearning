#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:57:23 2020

@author: alikemalcelenk
"""

# import library
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("linearRegression_dataset.csv",sep = ";") 

# plot data
plt.scatter(df.deneyim,df.maas)
# Grafikte X eksenim = df.deneyim,  Y eksenim = df.maas
plt.xlabel("deneyim") #label adı
plt.ylabel("maas") #label adı
plt.show() #grafiği çizer

#%% linear regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1) 
# type(x) = pandas.core.series.Series  if x = df.deneyim
# df.deneyim in veri tipi pandas. .values ile numpy arr ye çevirdim. 
# çevirme nedenimiz numpy da daha iyi oluyor başka şeylerde kullanılacağı için
# shape i (14,) geliyo. Bu kütüphaneyi yazan kişiler sayı görmeyi zorunlu kılmışlar bu yüzden reshape ediyoruz ve artık (14,1) olarak gözüküyor
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% prediction
import numpy as np

b0 = linear_reg.predict([[0]])
#iki tane arr nin içine almamızın nedeni = kütüphanenin öyle çalışması
print("b0: ",b0)

b0_ = linear_reg.intercept_ #bias
#yukardaki işlemin aynısını sklearn bize sağlıyor
print("b0_: ",b0_)   

b1 = linear_reg.coef_ #eğim
print("b1: ",b1)   

print(linear_reg.predict([[11]]))
# maas11 = 1663 + 1138*11
# print(maas11)

# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim
plt.scatter(x,y)
plt.show()
y_head = linear_reg.predict(array)  # maas
plt.plot(array, y_head, color = "red")
# x = array
# y = y_head
