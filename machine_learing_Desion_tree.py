import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/BaskinRobbinsCluster.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)
cols = [
    'Calories', 'Total Fat (g)', 'Trans Fat (g)', 'Carbohydrates (g)',
    'Sugars (g)', 'Protein (g)'
]
# fix,ax=plt.subplots(nrows=3,ncols=3,figsize=(20,9))
# ax=ax.ravel()
# for i,col in enumerate(cols):
#     sns.violinplot(x='cluster', y=col, data=df, ax=ax[i])
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x=df[cols]
y = df['cluster']
x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.3,  random_state=7)
model = DecisionTreeClassifier(criterion='gini')#gini is defualt, can be entropy
model.fit(x_train, y_train)
print(model.feature_importances_)
fs = pd.Series(model.feature_importances_,index=x_train.columns.sort_values(ascending=True))
print(fs)
print(y_train.value_counts())

# from sklearn.externals.six import StringIO
# from sklearn.tree import export_graphviz
# import pydotplus
# from IPython.display import Image
# dot_data =StringIO()
# export_graphviz(model, out_file=dot_data,feature_names=cols,class_names=['2','1','0'],filled=True,rounded=True, special_characters=True)
# graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())

print(model.tree_.impurity)
print(model.tree_.value)

from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
colors=y.map({'0':'green','1':'blue','2':'purple'})
ax.scatter(x['Calories'],
           x['Total Fat (g)'],
           x['Sugars (g)'],
           alpha=.5,
           c=colors)
ax.set_xlabel('Calories')
ax.set_ylabel('Total Fat (g)')
ax.set_zlabel('Sugars (g)')
plt.show()
