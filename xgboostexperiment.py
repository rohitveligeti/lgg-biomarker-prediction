# Rohit Veligeti
# December 29th 2021

# This python file uses the 251 features found before and utilizes a xgboost model and hyperparameter tuning to accurately determine what are the feature importances of the 351 selected features
# Furthermore, this result will be compared to a result taken from an optimized Random Forest model

from dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from sklearn.metrics import median_absolute_error

selected = pandas.read_csv('selected_features/351.csv')

use_list = True
to_use_list = list(selected.columns)[1::]

df = aggregrate()
df = df.sample(frac=1)

df = df[df['Outcome'] != 'Unknown']
df = df.replace("Progressive Disease", -1)
df = df.replace("Stable Disease", 0)
df = df.replace("Partial Remission/Response", 0.3)
df = df.replace("Complete Remission/Response", 1)

X = df.drop(['Outcome'], axis=1)

if use_list:
    X = X[to_use_list]



y = df['Outcome']

kf = KFold(n_splits=10)
kf.get_n_splits(X)

d = {}


for ne in range(30, 40, 1):
    round_loss = []
    print(ne)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = X_train.apply(pandas.to_numeric)
        X_test = X_test.apply(pandas.to_numeric)
        y_train = y_train.apply(pandas.to_numeric)
        y_test = y_test.apply(pandas.to_numeric)

        xgbr = xgb.XGBRFRegressor(n_estimators=130, colsample_bynode=float(ne)/100.0)
        xgbr.fit(X_train, y_train)

        x = mean_squared_error(y_test, xgbr.predict(X_test))
        # print(x)

        fi = xgbr.feature_importances_
        cn = X.columns.tolist()

        for i in range(len(fi)):
            if cn[i] in d:
                d[cn[i]].append(float(fi[i]))
            else:
                d[cn[i]] = [fi[i]]

        round_loss.append(x)

    average_loss = sum(round_loss) / len(round_loss)
    print(average_loss)

    average_d = {}

    for key, value in d.items():
        average_d[key] = sum(value) / len(value)

    average_d = sorted(average_d.items(), key=lambda x: x[1])

    new_average_d = {}

    for i in average_d:
        new_average_d[i[0]] = i[1]

    # print(new_average_d)

    print('#----------------------------------------------------')





