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

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

x_train, x_test, y_train, y_test =train_test_split(df[['sepal_length','sepal_width','petal_length','petal_width']], df.species,test_size=0.3,random_state=7)
# #Step 1:choose model
# model=KNeighborsClassifier()
# #Step2 : fit model
# model.fit(x_train,  y_train)
# #step3:predict
# y_pre=model.predict(x_test)
# #step4 :Score
# print(model.score(x_test, y_test ))

# #Step 1:choose model
# model=LogisticRegression()
# #Step2 : fit model
# model.fit(x_train,  y_train)
# #step3:predict
# y_pre=model.predict(x_test)
# #step4 :Score
# print(model.score(x_test, y_test ))

# #Step 1:choose model
# model = GaussianNB()
# #Step2 : fit model
# model.fit(x_train,  y_train)
# #step3:predict
# y_pre=model.predict(x_test)
# #step4 :Score
# print(model.score(x_test, y_test ))

algo = [[KNeighborsClassifier(), 'KNeighborsClassifier'],
        [LogisticRegression(solver='lbfgs'), 'LogisticRegression'],
        [GaussianNB(), 'GaussianNB'],
        [GradientBoostingClassifier(), 'GradientBoostingClassifier'],
        [RandomForestClassifier(), 'RandomForestClassifier'],
        [AdaBoostClassifier(), 'AdaBoostClassifier']]
model_score=[]
for a in algo:
    model = a[0]
    #Step2 : fit model
    model.fit(x_train, y_train)
    #step3:predict
    y_pre = model.predict(x_test)
    #step4 :Score
    score = model.score(x_test, y_test)
    model_score.append([score,a[1]])
    print(f'{a[1]}score={score}')
    print(metrics.confusion_matrix(y_test, y_pre))
    print(metrics.classification_report(y_test, y_pre))
    print('----------------------------'*3)
print(model_score)
dscore = pd.DataFrame(model_score, columns=['score','Model Classifier'])
# print(dscore)
print(dscore.sort_values('score',ascending=False))
