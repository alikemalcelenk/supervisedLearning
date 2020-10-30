#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:51:59 2020

@author: alikemalcelenk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True) 
# 'id' ve 'Unnamed: 32' olan sütunların bütün satırlarını(axis=1)ını kaldırdım çünkü gereksiz datalardı

# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
#M.info() 
#B.info()

# scatter plot
# x ekseni tümörlerin yarıçapı, y ekseni dokusu. 
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean") 
plt.ylabel("texture_mean")
plt.legend() # sağ üsste dataların ne olduğunu gösteriyor
plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1) 

# %%
# normalization 
# dataları normalize etmemiz gerekiyor çünkü;
# mesela x1 = 100, x2= 103 olsun. y1 = 0.01, y2 = 0.02 olsun .
# Mesafeleri hesaplarken kök[(103-100)^2 + (0.02-0.01)^2](euclidian Distance) işleminden 3,0001 gibi bir değer gelecek.
# Sonuç x değerinin neredeyse aynısı yani x değeri y değerini domine etti.
# Bundan kurtulmak için x değerierlini 0 ile 1 arasındaki değerlere çeviriyoruz ki domine etmiş olmasın.
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# %%
# KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k value
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

# %%
# find k value
# k değeri hyperparameter dır bu yüzden en iyi sonucu verdiği değeri arıyoruz.
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
#k değeerim 6 ile 10 arasındayken ve 12yken en iyi sonucu veriyor
