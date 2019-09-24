import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/ISLR/Advertising.csv'
)
df = pd.read_csv(url, usecols=[1, 2, 3, 4])
# print(df.head(5))
# print(df.info())
# sns.lmplot(x='TV',y='Sales',data=df,ci=None,scatter_kws={'alpha':0.4},line_kws={'color':'orange'});
# # plt.show()
# sns.pairplot(df,kind='reg');
# sns.pairplot(df,
#              kind='reg',
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
#                  });
# plt.show()

#sklearn:Linear Regression
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# # print(len(df))#size of Dataframe
# print(df.columns)
# #train model
# x = df.drop(columns=['Sales'])[:140]
# y = df['Sales'][:140]
# # print(len(x))
# # print(x.head(5))
# #fit model
# print(model.fit(x, y))
# #R square
# print(model.score(x, y))
# #xintercept
# print(model.intercept_)
# print(model.coef_)
# # print(model.predict([[200,40,70]]))
# # print(model.predict([[200, 40, 70],[100,80,50],[40,20,10]]))
x_test = df.drop(columns=['Sales'])[140:]
# print(x_test.head())
# y_hat = model.predict(x_test)
# print(y_hat)
# dc = pd.concat(
#     [df[140:]. reset_index(),
#      pd.Series(y_hat, name='predicted')],
#     axis='columns')
# print(dc)

#Statsmodel:Mutiple Linear regression
import statsmodels.api as sm
import statsmodels.formula.api as smf
#formular
# mode_a = smf.ols(formula='Sales~ TV+ Radio+ Newspaper', data=df[140:]).fit()
model_a = smf.ols(formula='Sales~ TV+ Radio', data=df[140:]).fit()
print(model_a.summary())
y_hat = model_a.predict(x_test)
print(y_hat)