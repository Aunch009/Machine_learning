import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/iris.csv'
)
df = pd.read_csv(url)
# print(df.sample(10))
print(df.columns)
print(df.species.value_counts())
# sns.pairplot(
#     df,
#     vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
#     hue='species',
#     markers=['o', 'D', '+'],
#     plot_kws={'alpha':.4});
# plt.show()
print(df.info())#'species' need to be object,str

#Scikit-learn:KneighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test =train_test_split(df[['sepal_length','sepal_width','petal_length','petal_width']], df.species,test_size=0.3,random_state=7)
print(x_train.head(5))

model = KNeighborsClassifier()
print(model)
model.fit(x_train, y_train)
print(model.score(x_train,  y_train))
print(model.predict([[3,4,5,6],[3,4,5,2],[5,3.5,1.5,2]]))

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
predicted = model.predict(x_train)
# print(predicted)
dx = pd.DataFrame({'y_true': y_train ,'y_pre': predicted})
print(dx[dx.y_true!=dx.y_pre])
print(confusion_matrix(y_train, predicted ))
print(accuracy_score(y_train, predicted ))
print(classification_report(y_train, predicted ))

model_b=LogisticRegression(solver='lbfgs')
model_b.fit(x_train,  y_train)
predicted_b=model_b.predict(x_train)
print(confusion_matrix(y_train, predicted_b))
print(accuracy_score(y_train, predicted_b))
print(classification_report(y_train, predicted_b))
