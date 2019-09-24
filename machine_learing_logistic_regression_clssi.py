import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/study_hours.csv'
)
df = pd.read_csv(url)

from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LogisticRegression
x,y=df[['Hours']],df.Pass
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=7)
model = LogisticRegression(solver='lbfgs')
print(model)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
predicted=model.predict(x_test)
# print(model.score(x_test, y_test))
# print(model.predict_proba(x_test))

from sklearn import  metrics
print(metrics.confusion_matrix(y_test,predicted))
print(metrics.accuracy_score(y_test, predicted))
print(metrics.precision_score(y_test, predicted))
print(metrics.recall_score(y_test, predicted))
print(metrics.f1_score(y_test, predicted))
print(metrics.classification_report(y_test, predicted))
