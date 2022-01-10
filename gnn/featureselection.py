# Rohit Veligeti
# December 13th 2021

# This python file uses random forests to quickly figure out the most important features to include in a model of low-grade-glioma progression

from dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

use_list = False
to_use_list = None

while True:
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

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)

        x = mean_squared_error(y_test, rf.predict(X_test))
        print(x)

        fi = rf.feature_importances_
        cn = X.columns.tolist()

        for i in range(len(fi)):
            if cn[i] in d:
                d[cn[i]].append(float(fi[i]))
            else:
                d[cn[i]] = [fi[i]]

    average_d = {}

    for key, value in d.items():
        average_d[key] = sum(value) / len(value)

    average_d = sorted(average_d.items(), key=lambda x: x[1])

    new_average_d = {}

    for i in average_d:
        new_average_d[i[0]] = i[1]

    x = int(0.7 * len(new_average_d))
    new_ones = list(new_average_d.keys())[len(list(new_average_d.keys())) - x:len(list(new_average_d.keys()))]

    # print(new_ones)

    use_list = True
    to_use_list = new_ones

    pand = pandas.DataFrame.from_records([new_average_d])
    pand.to_csv(f'{len(new_average_d)}.csv')

    print('#----------------------------------------------------')





