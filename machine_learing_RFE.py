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

#Sklearn:Classifier
from sklearn.model_selection import train_test_split

#these classifiers contain coef_ or feature_importances_attribute.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#KNeighborsClassifier has coef_ or feature_importances_attribute.
from sklearn.neighbors import KNeighborsClassifier
#recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn import metrics

cols = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
x = df[cols]
y = df['Outcome']
testsize = .3
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=testsize,
                                                    stratify=y,
                                                    random_state=7)

estimator = GradientBoostingClassifier(n_estimators=100)
#estimator =  RandomForestClassifier(n_estimators=100)
#estimator =  KNeighborsClassifier(n_estimators=100)

selector = RFE(estimator, 4, step=1)  #select 4 features
selector = selector.fit(x_train, y_train)
print(selector)
print(selector.support_)
print(selector.ranking_)
print(selector.n_features_)
sel=np.array(cols)[selector.support_]
selector.transform(x_test)
x_train_sel = pd.DataFrame(selector.transform(x_train),columns=np.array(cols)[selector.support_])
x_test_sel = pd.DataFrame(selector.transform(x_test),columns=np.array(cols)[selector.support_])
print(x_test_sel.head())

#Run model with selected features
model = RandomForestClassifier(n_estimators=100,random_state=777)
model.fit(x_train_sel,y_train)
score=model.score(x_test_sel,y_test)
y_pred = model.predict(x_test_sel)
print(f'feature importances',model.feature_importances_)
print(f'x[selected features] score :{score:.4f}')
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
fs=pd.Series(model.feature_importances_,index=x_train_sel.columns).sort_values(ascending=True)
print(fs)
fs.plot(kind='barh')
plt.show()

#Run model with all  features
model = RandomForestClassifier(n_estimators=100,random_state=777)
model.fit(x_train,y_train)
score=model.score(x_test,y_test)
y_pred = model.predict(x_test)
print(f'feature importances',model.feature_importances_)
print(f'x[selected features] score :{score:.4f}')
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
fs=pd.Series(model.feature_importances_,index=x_train.columns).sort_values(ascending=True)
print(fs)
fs.plot(kind='barh')
plt.show()