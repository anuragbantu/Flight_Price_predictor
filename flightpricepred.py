# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:48:30 2020

@author: GENIUS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dtrain = pd.read_excel('Data_Train.xlsx')

X = dtrain.iloc[:,0:9]
y = dtrain.iloc[:,10:11]

columns_to_remove = ['Route']
X.drop(labels=columns_to_remove, axis=1, inplace=True)

X["Journey_day"] = pd.to_datetime(X.Date_of_Journey, format="%d/%m/%Y").dt.day
X["Journey_month"] = pd.to_datetime(X.Date_of_Journey, format = "%d/%m/%Y").dt.month

X = pd.get_dummies(data=X, columns=['Airline', 'Source','Destination'])
X.drop(['Date_of_Journey'], axis=1 , inplace=True)

X["Dep_hour"] = pd.to_datetime(X["Dep_Time"]).dt.hour
X["Dep_min"] = pd.to_datetime(X["Dep_Time"]).dt.minute

X["Arrival_hour"] = pd.to_datetime(X["Arrival_Time"]).dt.hour
X["Arrival_min"] = pd.to_datetime(X["Arrival_Time"]).dt.minute
X.drop(["Arrival_Time"], axis = 1, inplace = True)


duration = list(X["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h" + duration[i]
            
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    
    duration_mins.append(int(duration[i].split(sep = "m")[0].split(sep = "h")[-1]))

X["Duration_hours"] = duration_hours
X["Duration_mins"] = duration_mins
X.drop(["Duration"], axis = 1,inplace = True)

X.drop(["Dep_Time"], axis = 1,inplace = True)

X.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from lightgbm.sklearn import LGBMRegressor
reg = LGBMRegressor()
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

filename = 'flightfare.pkl'
pickle.dump(reg, open(filename, 'wb'))

