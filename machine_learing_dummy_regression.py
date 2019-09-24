import pandas as pd
url = (
    'https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv')
df = pd.read_csv(url)
# print(df.columns)
# print(df.info())

#check NA in "vore" coloumns
df[df.vore.isna()]
#delete row na then update in dataframe
df.dropna(subset=['vore'], axis='index', inplace=True)
# print(df.info())
#create dummy variable
dummies = pd.get_dummies(df['vore'])
# print(dummies)
dm = pd.concat([df, dummies], axis='columns')
print(dm.columns)

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
dt = dm[['vore', 'bodywt', 'carni', 'herbi', 'insecti', 'omni',
         'sleep_total']]
x = dt[['bodywt', 'herbi', 'insecti', 'omni']]
# model.fit(x, dt.sleep_total)
# print(model.score(x, dt.sleep_total))

import statsmodels.api as sm
import statsmodels.formula.api as smf
model_a=smf.ols(formula='sleep_total~bodywt+herbi+insecti+omni',data=dt).fit()
print(model_a.pvalues)
print(model_a.summary())

#c() in statsmodel to create dmmy variable internally
model_b = smf.ols(formula='sleep_total~bodywt+C(vore)',
                  data=dt).fit()
print(model_b.pvalues)
print(model_b.summary())