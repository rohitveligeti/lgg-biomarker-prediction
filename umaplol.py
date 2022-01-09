# Rohit Veligeti
# December 29th 2021

# This python file uses the 251 features found before and utilizes a xgboost model and hyperparameter tuning to accurately determine what are the feature importances of the 351 selected features
# Furthermore, this result will be compared to a result taken from an optimized Random Forest model

from sklearn import neighbors
from dataset import *

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import umap



selected = pandas.read_csv('selected_features/351.csv')

use_list = True
to_use_list = list(selected.columns)[1::]

df = aggregrate()
df = df.sample(frac=1)

df = df[df['Outcome'] != 'Unknown']


X = df.drop(['Outcome'], axis=1)

if use_list:
    X = X[to_use_list]

y = df['Outcome']



X = X.apply(pandas.to_numeric)

for s in [2, 5, 10, 15, 20, 25]:
    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_neighbors=s, min_dist=0.1, metric='euclidean')
    embedding = reducer.fit_transform(X_scaled)

    print(embedding)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = embedding[:,0]
    df["comp-2"] = embedding[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette={'Progressive Disease': '#E5346E', 'Stable Disease': '#CBA175', 'Partial Remission/Response': '#C0D278', 'Complete Remission/Response': '#B6FC7B'},
                    data=df).set(title="UMAP projection") 

    plt.show()




