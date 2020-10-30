#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:17:38 2020

@author: alikemalcelenk
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomialLinearRegression_dataset.csv",sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()


# %% polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polynomialRegression = PolynomialFeatures(degree = 2)
# degree 2 yani x^2 ye kadar yap. y = b0 + b1.x + b2.x^2
#degree arttırkça karöaşıklık artıyor. fazla karmaşıklaştırmak iyi değil. orta yolu deneyip bulmak lazım. degree 4 güzel sonuç veriyor, 10 kötü

x_polynomial = polynomialRegression.fit_transform(x)
#fit_transform = x'i 2. dereceden polynomial a çevir

# %% fit
linearRegression = LinearRegression()
linearRegression.fit(x_polynomial,y)

# %% visualize line
y_head2 = linearRegression.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()
