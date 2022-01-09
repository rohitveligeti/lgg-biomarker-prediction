# Rohit Veligeti
# December 29th 2021

# This python file uses the 251 features found before and utilizes a xgboost model and hyperparameter tuning to accurately determine what are the feature importances of the 351 selected features
# Furthermore, this result will be compared to a result taken from an optimized Random Forest model

from dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import xgboost as xgb
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import export_graphviz

from sklearn import tree

import matplotlib.pyplot as plt

import time


selected = pandas.read_csv('selected_features/351.csv')

use_list = True
to_use_list = list(selected.columns)[1::]

df = aggregrate()
df = df.sample(frac=1)

df = df[df['Outcome'] != 'Unknown']
df = df.replace("Progressive Disease", 0)
df = df.replace("Stable Disease", 0.5)
df = df.replace("Partial Remission/Response", 0.7)
df = df.replace("Complete Remission/Response", 1)

X = df.drop(['Outcome'], axis=1)

if use_list:
    X = X[to_use_list]

y = df['Outcome']

X = X.apply(pandas.to_numeric)
y = y.apply(pandas.to_numeric)

param_grid = {'bootstrap': [True],
 'criterion': ['squared_error'],
 'max_depth': [5, 6, 7, 8, None],
 'max_features': ['sqrt', 'auto'],
 'max_leaf_nodes': [15, 20, 30],
 'min_impurity_decrease': [0],
 'min_samples_leaf': [1],
 'min_samples_split': [2, 3],
 'n_estimators': [200],
 'n_jobs': [-1],
 'random_state': [0],
 'verbose': [0]}

rf = RandomForestRegressor()

start_time = time.time()

grid_search = RandomizedSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2, n_iter=1000)
grid_search.fit(X, y)

print(grid_search.best_params_)
clf = grid_search.best_estimator_.estimators_[0]
text_rep = tree.export_text(clf)
print(text_rep)
print(grid_search.best_score_)

fi = list(grid_search.best_estimator_.feature_importances_)
new_dict = {}
for i in range(len(fi)):
    new_dict[to_use_list[i]] = fi[i]

    if fi[i] > 0.05:
        print(i, to_use_list[i], fi[i])

average_d = sorted(new_dict.items(), key=lambda x: x[1])

print(average_d)

print("--- %s seconds ---" % (time.time() - start_time))