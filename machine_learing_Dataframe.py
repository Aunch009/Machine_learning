from sklearn import datasets
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dat=datasets.load_boston()
# print(dat)
# type(dat)
# print(dat.data)
# print(dat.target)
# print(dat.DESCR)
# print(dat.feature_names)
# print(dat.data.shape)
# print(dat.target.shape)
# print(dat.target.reshape(-1,1))
c=np.hstack((dat.data,dat.target.reshape(-1,1)))
# print(c[:5])
df = pd.DataFrame(c, columns=list(dat.feature_names)+['MEDV'])
print(df.head(5))
print(df.columns)
sns.pairplot(data=df[[
    'CRIM',
    'RM',
    'AGE','MEDV'
]])
plt.show()

def sk_data_to_df(dat,target_column_name='target'):
    c = np.hstack((dat.data, dat.target.reshape(-1, 1)))
    return pd.DataFrame(c, columns=list(dat.feature_names)+['target_column_name'])

df2 = sk_data_to_df(datasets.load_iris())
print(df2)