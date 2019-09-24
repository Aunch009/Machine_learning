import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/BaskinRobbins.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)

#Scikit-learn:KMeans Clustering
# 1.Scale data
# 2.Optimal no. of cluster
#     A.Silhouette analysis
#         Visualize Silhouette
#     B.Elbow method
# 3.Compute and name clusters
from sklearn import preprocessing
from sklearn.cluster import KMeans
#1.Scale data
cols = [
    'Calories', 'Total Fat (g)', 'Trans Fat (g)', 'Carbohydrates (g)',
    'Sugars (g)', 'Protein (g)'
]
# df[cols].hist(layout=(1,len(cols)),figsize=(3*len(cols),3.5))
# plt.show()

#power transform
scaler = preprocessing.PowerTransformer(standardize=True)
S = scaler.fit_transform(df[cols])
print(S[:5].round(4))
x = pd.DataFrame(S, columns=cols)
# print(x.head())
# x[cols].hist(layout=(1,len(cols)),figsize=(3*len(cols),3.5),color='orange')
# plt.show()
# A.Silhouette analysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


def sil_score(x, from_k=2, to_k=6):
    '''Calculate silhouette score from k clusters'''
    sils = []
    for k in range(from_k, to_k + 1):
        m = KMeans(n_clusters=k)
        m.fit(x)
        #The silhouette_score give the average value for all sample from center
        silhouette_avg = silhouette_score(x, m.labels_).round(3)
        sils.append([silhouette_avg, k])
    #Compute the silhouette score for each sample
    # sample_silhouette_value = silhouette_samples(x,m.labels_)
    # print(sample_silhouette_value)
    return sils


ss = sil_score(x, 2, 5)
print(f'score={ss}')
print(f'optinum number of clusters ={max(ss)[1]}')
#
#Visualize Silhouette
#Instantiate the clustering model and visualizer
model = KMeans(n_clusters=3)
visualizer = SilhouetteVisualizer(model)
#fit the training data to visualizer
visualizer.fit(x)
#Draw/show/poof the data
visualizer.poof()
print(visualizer.silhouette_score_)#near 1 is good

def Silhouette_plot(x,from_k,to_k):
    sil_score=[]
    for k in range(from_k,to_k+1):
        #Instatiate the clustering model and visualizer
        m = KMeans(n_clusters=k)
        visualizer = SilhouetteVisualizer(m)
        visualizer.fit(x)
        #Draw/show/poof the data
        visualizer.poof()
        sil_score.append([visualizer.silhouette_score_.round(3), k])
    return sil_score

score= Silhouette_plot(x,2,5)
print(score)
print(max(score)[1])

#elbow method


def elbow_plot(x, from_k=2, to_k=5):
    '''plot elbow chart to help optimal number of clusters'''
    ssd = []
    for k in range(from_k, to_k + 1):
        m = KMeans(n_clusters=k)
        m.fit(x)
        ssd.append([k, m.inertia_])
    dd = pd.DataFrame(ssd, columns=['k', 'ssd'])
    dd['pct_chg'] = dd['ssd'].pct_change() * 100
    plt.plot(dd['k'], dd['ssd'], linestyle='--', marker='o')
    for index, row in dd.iterrows():
        plt.text(row['k'] + .02,
                 row['ssd'] + .02,
                 f'{row["pct_chg"]:.2f}',
                 fontsize=12)


elbow_plot(x, 2, 10)
plt.show()

#3.Compute and name clusters
model = KMeans(n_clusters=3)
model.fit(x)
model.cluster_centers_.round(4)
model.labels_
df['cluster'] = model.labels_
print(df.head())
sns.countplot(x='cluster',data=df)
# plt.show()
cols=[ 'Calories', 'Total Fat (g)', 'Trans Fat (g)',
       'Carbohydrates (g)', 'Sugars (g)', 'Protein (g)']
fig,ax =plt.subplots(nrows=2,ncols=3,figsize=(20,9))
ax=ax.ravel()
for i,col in enumerate(cols):
    sns.violinplot(x='cluster', y=col, data=df,ax=ax[i])
plt.show()

#scaled data(either z-score,power transform)
dx=x
dx['cluster']=model.labels_
print(dx.head(5))
print(dx.groupby('cluster').median())
fig, ax = plt.subplots(ncols=2, figsize=(18, 6))
ax=ax.ravel()
cols = [ 'Calories', 'Total Fat (g)', 'Trans Fat (g)', 'Carbohydrates (g)',
    'Sugars (g)', 'Protein (g)', 'cluster']
sns.heatmap(df[cols].groupby('cluster').median(),cmap="Oranges",linewidths=1,square=True,annot=True,fmt='.2f',ax=ax[0])
sns.heatmap(dx[cols].groupby('cluster').median(),cmap="Blues",linewidths=1,square=True,annot=True,fmt='.2f',ax=ax[1])
plt.show()