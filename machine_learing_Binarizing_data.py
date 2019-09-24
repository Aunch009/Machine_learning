import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/boston_house_price.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)
# df.hist(figsize=(15,10))
# plt.show()

from sklearn import preprocessing
#pandas normal method
print(df.AGE.mean())
print((df['AGE'] > df.AGE.mean()).head(5))
print((df['AGE'] > df.AGE.mean()).astype(int).head(5))

#Using binarize() function
mat = preprocessing.binarize(
    df[['AGE']],
    threshold=df.AGE.mean())  #return 2D numpy array by df.AGE.mean() criteria
df['age_cat'] = mat[:, 0]
# print(df.head(10))
cols = ['DIS', 'RM', 'MEDV']
mat = preprocessing.binarize(df[cols], threshold=[5, 7, df.MEDV.mean()])
# print(mat[:5])

#Using Binarizer() class(good when using with pipeline)
pbin = preprocessing.Binarizer(threshold=[5, 7, df.MEDV.mean()])
cols = ['DIS', 'RM', 'MEDV']
mat = pbin.fit_transform(df[cols])
print(mat[:5])
dn = pd.concat(
    [df, pd.DataFrame(mat, columns=['DIS_cat', 'RM_cat', 'MEDV_cat'])], axis=1)
print(dn.head())

#LogisticRegression for classifying house price
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x = df[[
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT',
]]
y = preprocessing.binarize(df[['MEDV']],threshold=df.MEDV.mean())
testsize=.3
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=testsize,random_state=7)
model=LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
dd=pd.Series(model.coef_.ravel(),index=x.columns)
print(dd)
#confusion matrix
from sklearn import metrics
y_pred =model.predict(x_test)
print(metrics.confusion_matrix(y_test,y_pred))
iii=pd.crosstab(y_test.ravel(), y_pred,rownames=['Actual'],colnames=['Predicted'],margins=True,margins_name='Total')
print(iii)
print(metrics.classification_report(y_test, y_pred))
