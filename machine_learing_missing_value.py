import pandas as pd
import seaborn as sns
import numpy as np
url = (
    'https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv')
df = pd.read_csv(url)
df=df.sample(20,random_state=123)
# print(df)
# print(df.info())

#check NA in 'vore' column
print(df[df.vore.isna()])

#Scikit-learn:Simplelmputer : ยืดหยุ่นกว่า pandas ในกรณีที่ไม่ได้อยู่ในรูปdata frame แต่อยู่ในรูปอื่นๆ ex np.array
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")# แทนที่ตัวที่มีมากที่สุด
df['vore2']=imp.fit_transform(df[['vore']])
# print(df[['vore']])
print(df[df.vore.isna()][['name','vore','vore2']])
imp2 = SimpleImputer(strategy="constant",fill_value='omni')  # แทนที่ตัวที่มีมากที่สุด
df['vore3']=imp2.fit_transform(df[['vore']])
print(df[df.vore.isna()][['name','vore','vore2','vore3']])
#replace by number
imp3 = SimpleImputer(strategy='mean')
df['sleep_rem2'] = imp3.fit_transform(df[['sleep_rem']])
print(df[df.sleep_rem.isna()][['name', 'sleep_rem', 'sleep_rem2']])

imp = SimpleImputer(strategy="constant", fill_value=-99)
df['sleep_rem4'] = imp.fit_transform(df[['sleep_rem']])
#ถ้าค่านำเข้า ไม่ make sense เช่น มากกกกกเกินไป ให้เปลี่ยนเป็นnan ก่อน
imp_x = SimpleImputer(missing_values=-99,strategy='constant',fill_value=np.nan)
df['sleep_rem5'] = imp_x.fit_transform(df[['sleep_rem4']])
print(df[df.sleep_rem.isna()][[
    'name', 'sleep_rem', 'sleep_rem2',  'sleep_rem4', 'sleep_rem5'
]])
