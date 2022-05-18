#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:15:46 2021

@author: Mar
"""

import numpy as np

#Data
data_xtest = np.load('Xtest_Regression_Part1.npy')
data_xtrain = np.load('Xtrain_Regression_Part1.npy')
data_ytrain = np.load('Ytrain_Regression_Part1.npy')



#Functions
def Get_score(model, X_train, X_validation, y_train, y_validation):
    model.fit(X_train, y_train)
    return model.score(X_validation, y_validation)

def Predict_val(model, X_train, X_validation, y_train):
    model.fit(X_train, y_train)
    return model.predict(X_validation)

def Average(lst):
    return sum(lst) / len(lst)



#K-Fold Cross Validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE

kfold = KFold(10)

scores_LR = [];
mse_LR = [];
scores_R = [];
mse_R = [];
scores_L = [];
mse_L = [];

for train_index, validate_index in kfold.split(data_xtrain, data_ytrain):
    X_train, X_validation = data_xtrain[train_index], data_xtrain[validate_index]
    y_train, y_validation = data_ytrain[train_index], data_ytrain[validate_index]
    scores_LR.append(Get_score(LinearRegression(), X_train, X_validation, y_train, y_validation))
    scores_R.append(Get_score(Ridge(alpha = 1), X_train, X_validation, y_train, y_validation)) #alpha values obtained further on
    scores_L.append(Get_score(Lasso(alpha = 0.0015771543086172347), X_train, X_validation, y_train, y_validation))
    mse_LR.append(MSE(y_validation, Predict_val(LinearRegression(), X_train, X_validation, y_train)))
    mse_R.append(MSE(y_validation, Predict_val(Ridge(alpha = 1), X_train, X_validation, y_train)))
    mse_L.append(MSE(y_validation, Predict_val(Lasso(alpha = 0.0015771543086172347), X_train, X_validation, y_train))) 

print("Average of scores_LR:", Average(scores_LR))
print("Average of scores_R:", Average(scores_R))
print("Average of scores_L:", Average(scores_L))
print("Average of mse_LR:", Average(mse_LR))
print("Average of mse_R:", Average(mse_R))
print("Average of mse_L:", Average(mse_L))



#Another K-Fold Cross Validation (compare alpha values of Ridge and Lasso and plots)
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

alphas = np.linspace(0.0001, 0.004, 50)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), data_xtrain, data_ytrain, cv=10, scoring='neg_mean_squared_error').mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

#second plot - higher alpha values
alphas = np.linspace(0.0001, 0.01, 50)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), data_xtrain, data_ytrain, cv=10, scoring='neg_mean_squared_error').mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

#third plot - even higher alpha values
alphas = np.linspace(0.0001, 3, 50)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), data_xtrain, data_ytrain, cv=10, scoring='neg_mean_squared_error').mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()

#fourth plot - just Ridge
alphas = np.linspace(0.0001, 2, 50)

plt.figure(figsize=(5, 3))

scores = [cross_val_score(Ridge(alpha), data_xtrain, data_ytrain, cv=10, scoring='neg_mean_squared_error').mean() 
          for alpha in alphas]
plt.plot(alphas, scores, label=Ridge.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()



#Another K-Fold Cross Validation (using RidgeCV e LassoCV and getting the best alpha value)
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

alphas = np.linspace(0.0001, 0.01, 500)
regressorR = RidgeCV(alphas=alphas, cv=10)
regressorR.fit(data_xtrain, data_ytrain)
print(regressorR.alpha_) 

alphas = np.linspace(0.0001, 0.004, 500)
regressorL = LassoCV(alphas=alphas, cv=10)
regressorL.fit(data_xtrain, data_ytrain.ravel())
print(regressorL.alpha_) 
#alpha is 0.0015771543086172347



#Final prediction
final_regressor = Lasso(alpha = 0.0015771543086172347)
final_regressor.fit(data_xtrain, data_ytrain)
final_regressor.coef_
final_results = final_regressor.predict(data_xtest)
np.save('data1.npy', final_results)







