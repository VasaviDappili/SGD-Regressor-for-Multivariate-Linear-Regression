# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California housing dataset and preprocess the data.
2. Split the dataset into training and testing sets.
3. Scale the features using StandardScaler and train the SGDRegressor using MultiOutputRegressor.
4. Evaluate the model using metrics like mean squared error and R-squared score.
## Program:
```import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.datasets import fetch_usa_housing

from sklearn.linear_model import SGDRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputRegressor

dataset=fetch_usa_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df["HousingPrice"]=dataset.target
df.head()
```
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DAPPILI VASAVI
RegisterNumber: 212223040030 
*/
```

## Output:
![image](https://github.com/user-attachments/assets/407dec9b-7c82-4179-ac06-82f76a44c98f)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
