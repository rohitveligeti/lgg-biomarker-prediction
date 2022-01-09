# Rohit Veligeti
# December 12th 2021

# This python file generates a tSNE-based visualization of the data without any of the alteration-based input

from dataset import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
from sklearn.decomposition import PCA

df = aggregrate()

columns = df.columns.tolist()
new_columns = []

for i in columns:
    if i[0:4] == 'ALT_':
        continue
    else:
        new_columns.append(i)

df = df[new_columns]


filter = pandas.read_csv('351.csv', header=0, index_col=0)
filter = filter.columns.tolist()

X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# X = X[filter]

print(X)

for i in range(3, 50):

    tsne = manifold.TSNE(n_components=2, random_state=711, perplexity=i, learning_rate=0.01)
    z = tsne.fit_transform(X)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette={'Unknown': 'Silver', 'Progressive Disease': 'Red', 'Stable Disease': 'Orchid', 'Partial Remission/Response': 'Aqua', 'Complete Remission/Response': 'Blue'},
                    data=df).set(title="No Alterations: T-SNE projection") 

    plt.show()