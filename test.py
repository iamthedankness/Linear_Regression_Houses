import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr

HouseDf = pd.read_csv('USAHD.csv')
# print(HouseDf.head())
# print(HouseDf.info())
# print(HouseDf.columns)
X = HouseDf[['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']]
y =HouseDf['price']
X_train , X_test , y_train, y_test =train_test_split(X , y , test_size=0.3, random_state=121   )

lr.fit(X_train, y_train)
