used_list = ['ALT_ARM20q', 'headache_history', 'preoperative_corticosteroids', 'ALT_ARM20p', 'ALT_MUT_NF1', 'ALT_MUT_IDH1', 
'RAGNUM_HYPOXIA_SCORE', 'karnofsky_performance_score', 'ALT_CNA_LINC00864', 'WINTER_HYPOXIA_SCORE', 'BUFFA_HYPOXIA_SCORE', 
'eastern_cancer_oncology_group', 'age_at_initial_pathologic_diagnosis', 'year_of_initial_pathologic_diagnosis']

ulist = {'ALT_CNA_RELN': 0.003159496814493476, 'ALT_CNA_SEMA3E': 0.003191729316823619, 'ALT_MUT_USH2A': 0.0032307634970862955, 'ALT_CNA_NSUN3': 0.0032697715479317506, 'ALT_MUT_HERC2P3': 0.003335780941866773, 'ALT_CNA_AGBL4': 0.003339939182548409, 'ALT_CNA_MTAP': 0.0033927282731480045, 'ALT_CNA_DHFRL1': 0.003416157228372533, 'ALT_MUT_BAGE2': 0.003456725723078666, 'ALT_CNA_AGBL4-IT1': 0.00350174733772676, 'ALT_CNA_LRRN3': 0.00354432555970218, 'ALT_MUT_FRMPD4': 0.0035834649241704977, 'ALT_CNA_SCG5': 0.0036260196127072936, 'ALT_CNA_CYP2E1': 0.003629974557430707, 'ALT_CNA_AKAP13': 0.00364615967170579, 'ALT_ARM4p': 0.004145333757959921, 'ALT_MUT_DOCK8': 0.004582990969964384, 'ALT_MUT_ATRX': 0.004706282010839112, 'ALT_MUT_NPAP1': 0.004754006943694945, 'ALT_MUT_BCOR': 0.005373596104124539, 'ALT_CNA_LUZP6': 0.0055781104604352375, 'ALT_MUT_SCN10A': 0.0059942332343675235, 'ALT_ARM20q': 0.007725589984808929, 'preoperative_corticosteroids': 0.008440142477180834, 'headache_history': 0.009297889160807444, 'ALT_ARM20p': 0.011204270576991953, 'ALT_MUT_IDH1': 0.012380964273045844, 'ALT_MUT_NF1': 0.012389625911031012, 'RAGNUM_HYPOXIA_SCORE': 0.014516024798971683, 'karnofsky_performance_score': 0.017049062535148967, 'WINTER_HYPOXIA_SCORE': 0.024209357364230764, 'ALT_CNA_LINC00864': 0.02570955059314763, 'BUFFA_HYPOXIA_SCORE': 0.026880094551603083, 'eastern_cancer_oncology_group': 0.030507076841313076, 'age_at_initial_pathologic_diagnosis': 0.038671628537486666, 'year_of_initial_pathologic_diagnosis': 0.1316133805696521}
ul2 = list(ulist.keys())

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
import pandas

import statistics

use_list = True
to_use_list = used_list.copy()

df = aggregrate()
df = df.sample(frac=1)

df = df[df['Outcome'] != 'Unknown']
df = df.replace("Progressive Disease", -1)
df = df.replace("Stable Disease", 0)
df = df.replace("Partial Remission/Response", 0.3)
df = df.replace("Complete Remission/Response", 1)

X = df.drop(['Outcome'], axis=1)

if use_list:
    X = X[used_list]



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

df = pandas.DataFrame()

for key, value in d.items():
    average_d[key] = sum(value) / len(value)
    df[key] = value

average_d = sorted(average_d.items(), key=lambda x: x[1])

new_average_d = {}

for i in average_d:
    new_average_d[i[0]] = i[1]

print(new_average_d)

print('#----------------------------------------------------')
print(df)
df.to_csv('alterationstuff.csv')




