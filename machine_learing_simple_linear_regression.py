import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
urltr = (
    'https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-adjusted-index.csv'
)
big_mac = pd.read_csv(urltr, parse_dates=['date'])
# print(big_mac.head(5) )
# print(big_mac.info())
big_mac = big_mac[(big_mac['date'].dt.year == 2019)
                  & (big_mac['date'].dt.month == 1)]
# print(big_mac.head(5) )
# sns.lmplot(x='GDP_dollar', y='dollar_price',data=big_mac,ci=None)
# plt.show()
# sns.jointplot(x='GDP_dollar', y='dollar_price', data=big_mac, kind='reg',ci=None)
# plt.show()

#sklearn:LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)
#x need 2D array
x = big_mac[['GDP_dollar']]
#y is 1D array
y = big_mac.dollar_price
model.fit(x, y)
# print(model.score(x,y))#Rsqure
# print(model.intercept_)
# print(model.coef_)
# # y=a+Bx
# print(model.intercept_+model.coef_*9000)
# #same as
# print(model.predict([[9000]]))#2D array
# print(model.predict([[9000],[40000]]))
# print(np.arange(5000, 50001, 2500))  #start 5000-50001 each 2500
# print(np.arange(5000, 50001, 2500).reshape(-1, 1))  #2D array
# pred = model.predict(np.arange(5000, 50001, 2500).reshape(-1, 1))
# print(pred)
# pred_1 = model.predict(np.linspace(5000, 50000,
#                                    20).reshape(-1,
#                                                1))  #start 5000-50001 20time
# print(pred_1)

# s=pd.Series(np.linspace(5000, 50000,20))
# print(model.predict(s.to_frame()))
# s1 = pd.DataFrame(np.linspace(5000, 50000, 20),columns=['gdp'])
# print(model.predict(s1))

#Statsmodel
import statsmodels.api as sm
import statsmodels.formula.api as smf
#formular
model_a = smf.ols(formula='dollar_price ~ GDP_dollar', data=big_mac).fit()
print(model_a.summary())
print(model_a.pvalues)