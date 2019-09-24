import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/study_hours.csv'
)
df = pd.read_csv(url)
print(df.columns)
# sns.scatterplot(data=df, x='Hours',y='Pass')
# sns.lmplot(x='Hours', y='Pass',data=df,logistic=False,ci=False,height=4,aspect=1.5,line_kws={'color':'orange'})
# plt.axhline(.5,color='red',linestyle='--')
# sns.lmplot(x='Hours',
#            y='Pass',
#            data=df,
#            logistic=True,
#            ci=False,
#            height=4,
#            aspect=1.5,
#            line_kws={'color': 'orange'})
# plt.axvline(2.7, color='green', linestyle='--')
# plt.axhline(.5, color='red', linestyle='--')
# plt.show()

#statsmodel
from patsy import dmatrices
import statsmodels.api as sm
y, x = dmatrices('Pass ~ Hours', data=df, return_type='dataframe')
# print(x.head())
# print(y.head())
model_a = sm.Logit(y, x).fit()
print(model_a.summary())
# print(model_a.summary2())
# print(model_a.predict([[1, 2], [1, 4]]))
# print(model_a.predict(x))
#predict less than .5 fail
# print(model_a.predict(x).apply(lambda p:0 if p<.5 else 1))
df['predicted'] = model_a.predict(x).apply(lambda p: 0 if p < .5 else 1)
df['log_odds'] = model_a.params[
    'Intercept'] + model_a.params['Hours'] * df['Hours']
df['odds'] = np.exp(model_a.params['Intercept'] +
                    model_a.params['Hours'] * df['Hours'])
df['prob'] = model_a.predict(x)
# print(df)

#statsmodels:confusion matrix
#[tn fp]  tp=ทายว่าสอบตกแล้วตกจริง  tp,fn ทายผิด
# fn  tp
print(model_a.pred_table())
tn, fp, fn, tp = model_a.pred_table().ravel()
print((tp + tn) / (tp + tn + fp + fn))

#visualize
import math


#ver1 sigmoius formula
def sg(intercept, coef, x):
    ex = math.exp(-(intercept + coef * x))
    return (1 / (1 + ex))


#ver2 sigmoius formula
def sp(intercept, coef, x):
    ex = np.exp(-(intercept + coef * x))
    return (1 / (1 + ex))


xp = np.linspace(0, 5, 20)
yp = sp(model_a.params[0], model_a.params[1], xp)
plt.plot(xp, yp)
plt.axhline(.5, color='red', linestyle='--')
plt.axvline(np.abs(model_a.params['Intercept'] / model_a.params['Hours']),
            color='green',
            linestyle='--')
plt.show()
