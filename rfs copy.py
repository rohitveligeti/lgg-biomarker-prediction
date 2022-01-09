# Rohit Veligeti
# December 29th 2021

# This python file uses the 251 features found before and utilizes a xgboost model and hyperparameter tuning to accurately determine what are the feature importances of the 351 selected features
# Furthermore, this result will be compared to a result taken from an optimized Random Forest model

from dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

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

round_loss = []

r2loss = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train = X_train.apply(pandas.to_numeric)
    X_test = X_test.apply(pandas.to_numeric)
    y_train = y_train.apply(pandas.to_numeric)
    y_test = y_test.apply(pandas.to_numeric)

    # rf = RandomForestRegressor(bootstrap=True, criterion='squared_error', max_depth=None, max_features='auto', max_leaf_nodes=30, min_impurity_decrease=0, min_samples_leaf=1, min_samples_split=2, n_estimators=200, n_jobs=-1, random_state=0, verbose=0)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    x = mean_squared_error(y_test, rf.predict(X_test))
    x2 = r2_score(y_test, rf.predict(X_test))
    print(x2, "X2")
    # print(x)

    fi = rf.feature_importances_
    cn = X.columns.tolist()

    for i in range(len(fi)):
        if cn[i] in d:
            d[cn[i]].append(float(fi[i]))
        else:
            d[cn[i]] = [fi[i]]

    round_loss.append(x)
    r2loss.append(x2)

average_loss = sum(round_loss) / len(round_loss)
print(average_loss)

al2 = sum(r2loss) / len(r2loss)
print(al2)

average_d = {}

for key, value in d.items():
    average_d[key] = sum(value) / len(value)

average_d = sorted(average_d.items(), key=lambda x: x[1])

new_average_d = {}

for i in average_d:
    new_average_d[i[0]] = i[1]

print(new_average_d)

print('#----------------------------------------------------')





