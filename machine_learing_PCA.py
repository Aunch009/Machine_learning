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

cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[cols].hist(layout=(1,len(cols)),figsize=(3*len(cols),3.5))


#Correlation matrix
dcorr=df[cols].corr()
mask=[]
mask=np.zeros_like(dcorr)
#mask.shape
mask[np.triu_indices_from(mask)]=True
fig,ax=plt.subplots(figsize=(7,5))
print((fig,ax))
# sns.heatmap(dcorr,cmap=sns.diverging_palette(10,145,n=100),linewidths=1,vmin=-1,vmax=1,center=0)
sns.heatmap(dcorr,cmap=sns.diverging_palette(10,145,n=100),vmin=-1,vmax=1,center=0,linewidths=1,annot=True,mask=mask,ax=ax)
# sns.pairplot(df,
#              kind='reg',
#              hue='species',
#              plot_kws={
#                  'scatter_kws': {
#                      'alpha': 0.4
#                  },
#                  'line_kws': {
#                      'color': 'orange'
#                  }
#              },
#              diag_kws={
#                  'color': 'green',
#                      'alpha': .2
#                  })
sns.pairplot(
    df,
    vars=cols,
    hue='species',
    markers=['o', 'D', '+'],
    plot_kws={'alpha':.4})
plt.show()

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#creat StandardScaler instance
scaler=StandardScaler()
#Transform
x=scaler.fit_transform((df[cols]))#cal z-score
y=df.species
# x.shape[1]=4 เนื่องจากมี variable 4 ตัว
pca=PCA(n_components=x.shape[1])
pca.fit_transform(x)
print(f'explained_variance(n_components={pca.n_components})={pca.explained_variance_.round(3)}')
print(f'explained_variance_ratio(n_components={pca.n_components})={pca.explained_variance_ratio_.round(3)}')
print(f'sum explained_variance_ratio={np.sum(pca.explained_variance_ratio_).round(3)}')

def scree_plot(x,max_componemts,with_cumulative=False):
    vr=[]
    for n in range(1,max_componemts+1):
        pca=PCA(n_components=n)
        pca.fit_transform(x)
        vr.append([
            n,
            pca.explained_variance_ratio_.round(3),
            np.sum(pca.explained_variance_ratio_).round(3)
        ])
        # print(f'explained_variance_ratio(n_components={pca.explained_variance_ratio_.round(3)},total={np.sum(pca.explained_variance_ratio_).round(3)}')
    xy=np.array(vr)
    if with_cumulative:
        plt.plot(xy[:,0],xy[:,2],linestyle='--',marker='o',label='cumulative')
    plt.plot(xy[:,0],xy[:,1][-1],linestyle='--',marker='o',label='individual')
    plt.title('explained_variance_ratio_')
    plt.xlabel('# of component')
    plt.ylabel('proportion of variance explained')
    plt.legend()

    for n,v,cv in zip(np.nditer(xy[:,0],flags=['refs_ok']),np.nditer(xy[:,1][-1],flags=['refs_ok']),np.nditer(xy[:,2],flags=['refs_ok'])):
        plt.text(n+.02,v+.02,f'{v*100:.2f}%',fontsize=10)
        if with_cumulative:
            plt.text(n+.02,cv+.02,f'{cv*100:.2f}%',fontsize=10)

scree_plot(x,4,True)
plt.show()

dpc = pd.DataFrame(pca.components_.T,
                   index=cols,
                   columns=[f'PC{n+1}' for n in range(pca.n_components)]).round(4)
print(dpc)
print(dpc.style.applymap(lambda e:'background-color:yellow' if np.abs(e)>.5 else'background-color:yellow'))
sns.heatmap(dpc,
            cmap=sns.diverging_palette(10, 145, n=100),
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=1,
            annot=True)
plt.show()
dd= pd.concat([pd.DataFrame(pca.transform(x),columns=[f'PC{n}' for n in range(1,pca.n_components_+1)]), df[['species']]],axis='columns' )
sns.scatterplot(data=dd,x='PC1',y='PC2',hue='species',alpha=.4)
plt.show()