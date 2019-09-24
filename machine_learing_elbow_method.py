import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
x, y = make_blobs(n_samples=100, centers=4, n_features=2, center_box=(0, 30))
# x,y = make_classification(n_samples=100,n_classes=4,n_features=2,center_box(0,30))
print(x[:5])
print(y)
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], s=20, c=y, alpha=.4)


from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(x)
print(model.labels_)
print(model.cluster_centers_)
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], s=20, c=y, alpha=.4)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],color='red',marker='D',s=50)


fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
k=1
ssd=[]#(model.inertai_)Sum of squared distances of samples to their closet cluster center
for r in range(ax.shape[0]):
    for c in range(ax.shape[1]):
        ax[r, c].scatter(x[:, 0], x[:, 1], s=20,  alpha=.4)
        m=KMeans(n_clusters=k)
        m.fit(x)
        ssd.append([k,m.inertia_])
        ax[r, c].scatter(m.cluster_centers_[:, 0],
                         m.cluster_centers_[:, 1],
                         color='red',
                         marker='D',
                         s=50)
        ax[r,c].set_title(f'k={k},inertia={m.inertia_ :,.2f}')

        k+=1
plt.show()
print(ssd)

xy=np.array(ssd)
plt.plot(xy[:, 0], xy[:, 1],linestyle='--',marker='o')
plt.show()

