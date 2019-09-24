import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Advertising.csv'
)
df = pd.read_csv(url)
print(df.head())
print(df.info())
print(df.columns)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=7)
model = LinearRegression()
model.fit(x_train, y_train)
print(model.intercept_)
print(model.coef_)
pp=pd.Series(model.coef_,index=x.columns)
print(pp)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
print(model.predict([[200,40,70]]))

#dump(save)and load model with joblib
from  sklearn.externals import joblib
joblib.dump(model,'Advertising.joblib')#Save  model
lr=joblib.load('Advertising.joblib')#load 'Advertising.joblib' for using
print(lr)
print(pd.Series(lr.coef_,index=x.columns))
print(lr.predict([[200, 40, 70]]))
