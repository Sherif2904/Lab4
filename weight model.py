# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:08:27 2021

@author: sherif
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score


file_path='F:/DC Material/My Run/Sem 2/2004 AI in enterprise systems/Lab 4'
df=pd.read_csv(file_path+'/fish.csv')

y = df.iloc[:,1:2]
x = df.iloc[:,2:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 16)


from sklearn import ensemble
params = {}
params['n_estimators'] = 100
clf = ensemble.GradientBoostingRegressor(n_estimators = params['n_estimators'], max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)
clf.score(x_test,y_test)
test_score = np.zeros((params['n_estimators']),dtype=np.float64)

y_pred = clf.predict(x_test)

print("Training accuracy :", clf.score(x_train, y_train))
print("Testing accuarcy :", clf.score(x_test, y_test))

import pickle
pickle_out=open(file_path+'/clf.pkl',"wb")
pickle.dump(clf,pickle_out)
pickle_out.close()
