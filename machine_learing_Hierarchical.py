import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/BaskinRobbins.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)

from sklearn import  preprocessing
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch #draw dendrogram
cols = [
    'Calories', 'Total Fat (g)', 'Trans Fat (g)',
    'Carbohydrates (g)', 'Sugars (g)', 'Protein (g)']
#yeo-johnson tansformation(preferable) that support only positive value
pt=preprocessing.PowerTransformer(method='yeo-johnson',standardize=True)
mat=pt.fit_transform((df[cols]))
x=pd.DataFrame(mat.round(3),columns=cols)
# print(x.head(5))
df[cols].hist(layout=(1,len(cols)),figsize=(3*len(cols),3.5))
x[cols].hist(layout=(1,len(cols)),figsize=(3*len(cols),3.5),color='orange')

fig,ax=plt.subplots(figsize=(20,7))
dg = sch.dendrogram(sch.linkage(x, method='ward'), ax=ax, labels=df['Flavour'].values)
sns.clustermap(x,col_cluster=False,cmap="Blues")
# plt.show()
hc = AgglomerativeClustering(n_clusters=2,linkage='ward')
hc.fit(x)
df['cluster'] = hc.labels_
stat2=df.groupby('cluster').agg(['count','mean','median']).T.round(3)
print(stat2)

cols = [
    'Calories', 'Total Fat (g)', 'Trans Fat (g)', 'Carbohydrates (g)',
    'Sugars (g)', 'Protein (g)'
]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 9))
ax = ax.ravel()
for i, col in enumerate(cols):
    sns.violinplot(x='cluster', y=col, data=df, ax=ax[i])

dx = x
dx['cluster'] = hc.labels_
# print(dx.head(5))
# print(dx.groupby('cluster').median())
fig, ax = plt.subplots( figsize=(18, 6))
cols = [
    'Calories', 'Total Fat (g)', 'Trans Fat (g)', 'Carbohydrates (g)',
    'Sugars (g)', 'Protein (g)', 'cluster'
]
sns.heatmap(df[cols].groupby('cluster').median(),
            cmap="Oranges",
            linewidths=1,
            square=True,
            annot=True,
            fmt='.2f',
            ax=ax)

plt.show()
print(df.groupby('cluster').head(3).sort_values('cluster'))