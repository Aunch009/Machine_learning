import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/iris.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)
print(df.species.value_counts())
# sns.pairplot(df,
#              vars=[
#                  'sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
#             hue='species',
#             markers=['o','D','+'],
#             plot_kws={'alpha':.4} )
# plt.show()

#Scikit-learn:KMeans Clustering
# sns.scatterplot(data=df, x='petal_length',y='petal_width')
rx = np.random.uniform(1, 7, 3)
ry = np.random.uniform(0, 2.5, 3)
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
plt.scatter(rx, ry, color='.1', marker='D')
plt.show()
print(rx)
print(ry)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
# print(model)
x=df[[ 'petal_length', 'petal_width']]
model.fit(x)
print(model.cluster_centers_)
# plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,0])
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species',alpha=.5,palette=['green','blue','orange'])
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            color='.1',
            marker='D')
plt.show()
model.labels_
df['cluster'] = model.labels_
print(df.head(5))
hh=pd.crosstab(df['species'],df['cluster'])
print(hh)
sns.scatterplot(data=df,
                x='petal_length',
                y='petal_width',
                hue='cluster',
                alpha=.5,
                palette=['green', 'blue', 'orange'])
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            color='.1',
            marker='D')
plt.show()
fig,ax=plt.subplots(1,2,figsize=(12,5),sharey=True,sharex=True)
g1= sns.scatterplot(data=df,
                x='petal_length',
                y='petal_width',
                hue='species',
                alpha=.5,
                palette=['green', 'blue', 'orange'],ax=ax[0])
g1.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            color='.1',
            marker='D')
g2=sns.scatterplot(data=df,
                x='petal_length',
                y='petal_width',
                hue='cluster',
                alpha=.5,
                palette=['green', 'blue', 'orange'],ax=ax[1])
g2.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            color='.1',
            marker='D')
plt.show()

print(model.predict([[1.5,.3]]))
