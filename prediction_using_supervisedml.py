# -*- coding: utf-8 -*-
"""Prediction_using_SupervisedML.ipynb

Original file is located at
    https://colab.research.google.com/drive/1iFSdR5yFZvPuqNBtlZpP4GO34TjsrYYq

# Task 1 - Prediction using Supervised ML
## (Level - Beginner)

Author : Aditya K. Kataria
Email : adityakataria36@gmail.com
Data Science & Business Analytics Internship
GRIP December2020

Aim : Predict the percentage of a student based on the no. of study hours.
Dataset : Data can be found at http://bit.ly/w-data

What will be predicted score if a student studies for 9.25 hrs/day?
"""

# Importing all the important Libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# %matplotlib inline

# Loading Dataset
df = pd.read_csv('http://bit.ly/w-data')
print('Shape:', df.shape)
print(df.head())

# Vizualizing Data
df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.savefig('datasetVizualization.png')
plt.show()

"""## Preparing the data
Here, we store the values of attribute 'Hours' in X and label 'Scores' in y.
"""
# Dividing the dataset columns. 
# Hours in X and Scores in y
X = df.iloc[:, :-1].values    
y = df.iloc[:, 1].values

"""## Splitting Dataset
The split of data into the training and test sets is very important. We use Scikit Learn's built-in method of train_test_split() from model_selection.

The test size is kept 0.3 which indicates that 70% of data is used for training and remaining 30% of data is used for testing purpose.
"""
# Split Data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""## Linear Regression Model
LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

With Scikit-Learn it is extremely straight forward to implement linear regression models, as all you really need to do is import the LinearRegression class, instantiate it, and call the fit() method along with our training data. 
"""

# Linear Regression Model
lr = LinearRegression()    
lr.fit(X_train.reshape(-1, 1).astype(np.float32), y_train)
print('Linear Regression Model Trained Successfully!')

# Getting B0 and B1
print('coef_ :', lr.coef_)
print('intercept_ :', lr.intercept_)

# Plotting a Regression Line
regression_line = lr.coef_ * X + lr.intercept_
plt.scatter(X, y)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.plot(X, regression_line, color='orange')
plt.savefig('regressionLine.png')
plt.show()

"""## Prediction
Using the trained Linear Regression Model to predict scores for test data and compare the results.
"""
# Prediction
y_pred = lr.predict(X_test)

# Actual Score vs Predicted Score
new_df = pd.DataFrame({'Actual Score': y_test, 
                       'Predicted Score': y_pred})
print(new_df)

fig, ax = plt.subplots()
index = np.arange(len(X_test))
bar_width = 0.35
actual = plt.bar(index, new_df['Actual Score'], bar_width, label='Actual Score')
predicted = plt.bar(index + bar_width, new_df['Predicted Score'], bar_width, label='Predicted Score')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Actual vs Predicted Scores')
plt.xticks(index + bar_width, X_test)
plt.legend()
plt.savefig('actualvspredicted.png')
plt.show()

# Model Evaluation
print('Mean Absolute Error MAE:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error MSE:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error RMSE:', math.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Predicted score if a student studies for 9.25 hrs/day
input_hours = input('\nEnter number of hours for predicting the score: ')
input_hours = float(input_hours)
hours = np.array([input_hours])
hours = hours.reshape(-1, 1)
pred_score = lr.predict(hours)
print("Number of Hours =", input_hours, 'hours/day')
print("Predicted Score = {:.4}".format(pred_score[0]))
