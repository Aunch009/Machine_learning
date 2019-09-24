import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/diabetes.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)
# sns.pairplot(df,kind='reg',plot_kws={'scatter_kws':{'alpha':0.4},'line_kws':{'color':'orange'}},diag_kws={'color':'green','alpha':.2})
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics

x = df[[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]]
y = df['Outcome']
model = RandomForestClassifier(n_estimators=250, random_state=7)
# model=DecisionTreeClassifier(random_state=7)
# model=ExtraTreesClassifier(n_estimators=250,random_state=7)
model.fit(x, y)
fs = pd.Series(model.feature_importances_,
               index=x.columns).sort_values(ascending=True)
# print(fs)
# fs.plot(kind='barh')
# plt.show()
# print(fs[fs>.1])
# print(fs.nlargest(4))#.index)
x = df[fs[fs > .1].index]
print(x.head())
testsize = 0.3
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=testsize,
                                                   stratify=y,random_state=7)
algo = [[KNeighborsClassifier(n_neighbors=5), 'KNeighborsClassifier'],
        [LogisticRegression(solver='lbfgs'), 'LogisticRegression'],
        [Perceptron(), 'Perceptron'], [GaussianNB(), 'GaussianNB'],
        [
            DecisionTreeClassifier(min_samples_split=10),
            'DecisionTreeClassifier'
        ], [GradientBoostingClassifier(), 'GradientBoostingClassifier'],
        [RandomForestClassifier(), 'RandomForestClassifier'],
        [AdaBoostClassifier(),'AdaBoostClassifier'],[BaggingClassifier(), 'BaggingClassifier'],
         [MLPClassifier(), 'MLPClassifier'],
        [SVC(kernel='linear'), 'SVC_linear'],
        [GaussianProcessClassifier(), 'GaussianProcessClassifier']]

model_score=[]
for a in algo:
    model=a[0]
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_score.append([score,a[1]])
    y_pred=model.predict(x_test)
    print(f'{a[1]:20} score:{score:.4f}')
    print(metrics.confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred))
    print('-'*100)
print(model_score)
print(f'best score ={max(model_score)}')
dscore=pd.DataFrame(model_score,columns=['Score','Classifier'])
print(dscore.sort_values('Score',ascending=False))