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
# df.hist(figsize=(20,10))
# plt.show()
from sklearn import preprocessing
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
cols = [
    'CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
    'LSTAT', 'MEDV'
]
mat = pt.fit_transform(df[cols])
box_cols=[f'bc_{c}' for c in cols]
print(box_cols)
# ds = pd.concat([df, pd.DataFrame(mat, columns=box_cols)],axis='columns')#Not use for negative and zero
ds = pd.concat([df, pd.DataFrame(mat, columns='yeo-johnson')],axis='columns')#defual can use for negative&positive and zero
print(ds.head())
df[cols].hist(layout=(2,6),figsize=(15,4))
ds[box_cols].hist(layout=(2,6),figsize=(15, 4))
plt.show()