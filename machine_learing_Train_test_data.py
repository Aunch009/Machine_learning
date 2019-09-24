import pandas as pd
import seaborn as sns
import numpy as np

url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/ISLR/Advertising.csv'
)
df = pd.read_csv(url, usecols=[1, 2, 3, 4])

#Sklearn:train_test_split
from sklearn.model_selection import train_test_split
print(len(df))
print(df.columns)

# #Method1: Split into(train,test)
train,test=train_test_split(df,train_size=0.7,random_state=7)#train size 70% And randomใส่ก็ได้ไม่ใส่ก้ได้
# print(len(train))
# print(len(test))
# print(train.head(5))

#Method 2:split into (x_train,x_test,y_train,y_test)
# print(df.columns)
# x = df[['TV', 'Radio', 'Newspaper']]
# y= df['Sales']
# x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=7)
# # print(x_train.head())

# #LinearRegression
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# print(model.fit(x_train,  y_train))
# # print(model.score(x_train, y_train))
# y_hat = model.predict(x_train)
# # print(y_hat)
# train = pd.concat([x_train, y_train],axis='columns')
# dc = pd.concat([train.reset_index(), pd.Series(y_hat,name='predicted')],axis='columns')
# print(dc.head())
# y_hat_test = model.predict(x_test)
# test = pd.concat([x_test, y_test],axis='columns')
# print(test.head(5))
# dt=pd.concat([test.reset_index(),pd.Series(y_hat_test,name='predicted')],axis='columns')
# print(dt.head())
# print(dt.corr())

import statsmodels.api as sm
import statsmodels.formula.api as smf
#formular
# mode_a = smf.ols(formula='Sales~ TV+ Radio+ Newspaper', data=df[140:]).fit()
model_a = smf.ols(formula='Sales~ TV+ Radio+ Newspaper', data=train).fit()
print(model_a.summary())