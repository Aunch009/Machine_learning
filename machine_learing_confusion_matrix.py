import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
url = (
    'https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/study_hours.csv'
)
df = pd.read_csv(url)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(df[['Hours']], df.Pass,
                                                    test_size=0,
                                                    random_state=3)

from sklearn import metrics
