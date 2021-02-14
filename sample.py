import numpy as np
import matplotlib as mtp 
import pandas as pd 

data_set= pd.read_csv('50_Complist.csv')
data_set.head(2)

x= data_set.iloc[:, :-1].values
y= data_set.iloc[:, 4].values

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encode State Column 
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
from sklearn.compose import ColumnTransfer
ct = ColumnTransfer([("state", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)

#Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred= regressor.predict(x_test)

print('Train Score: ', regressor.score(x_train, y_train))
print('Test Score ', regressor.score(x_test, y_test))