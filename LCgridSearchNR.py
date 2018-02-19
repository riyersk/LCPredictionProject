#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:13:19 2018

@author: subramanianiyer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
with open("LCDC3.pkl", 'rb') as picklefile: 
    df1 = pickle.load(picklefile)
from copy import deepcopy
varss = []
for i in df1.columns.values:
    if i != 'loan_status':
        varss.append(i)
dfs = []
Xs = []
ys = []
subgrades = sorted(list(df1.sub_grade.unique()))
a = 0
for i in subgrades:
    dfs.append(deepcopy(df1[df1.sub_grade==i]))
    Xs.append(dfs[a][varss])
    del dfs[a]['sub_grade']
    del Xs[a]['sub_grade']
    ys.append(dfs[a]['loan_status'])
    a+=1
curX = Xs[len(Xs)-2]
cury = ys[len(ys)-2]
curX = pd.get_dummies(curX)
def ret(y_pred, y_test, X_test):
    portfolio = []
    rates = []
    for i in range(len(y_pred)):
        if y_pred[i]==0:
            portfolio.append(y_test[i])
            rates.append(X_test[i])
    if len(portfolio)==0:
        return float(-1)
    return ((float(len(portfolio)-sum(portfolio))/float(len(portfolio)))*(1.+ (np.mean(rates)/100.))) - 1
def myScore(estimator, X, y):
    y_pred = estimator.predict(X)
    return ret(list(y_pred), list(y), list(X['int_rate']))
for i in range(1,12):
    curX = Xs[len(Xs)-i]
    cury = ys[len(ys)-i]
    curX = pd.get_dummies(curX)
    X_train, X_test, y_train, y_test = train_test_split(curX, cury, test_size=0.15, random_state = i)
    parameters = {'C':[10**x for x in [1.86,1.87,1.88,1.89,1.90]], 'n_jobs':[-1]}
    mod = GridSearchCV(LogisticRegression(), parameters, scoring = myScore, n_jobs = -1, cv = 5)
    mod.fit(X_train,y_train)
    f = open('lr'+subgrades[len(subgrades)-i]+'.txt','w')
    f.write('Logistic Regression for '+subgrades[len(subgrades)-i]+'\n')
    f.write(str(myScore(mod, X_test, y_test))+'\n')
    f.write(str(confusion_matrix(mod.predict(X_test), y_test))+'\n')
    f.close()
    parameters = {'n_estimators':range(8,23),'max_features':range(7,34),'min_samples_split':range(4,13), 'n_jobs':[-1]}
    mod = GridSearchCV(RandomForestClassifier(), parameters, scoring = myScore, n_jobs = -1, cv = 5)
    mod.fit(X_train,y_train)
    f = open('rf'+subgrades[len(subgrades)-i]+'.txt','w')
    f.write('Random Forest for '+subgrades[len(subgrades)-i]+'\n')
    f.write(str(myScore(mod, X_test, y_test))+'\n')
    f.write(str(confusion_matrix(mod.predict(X_test), y_test))+'\n')
    f.close()