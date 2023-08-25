"""
1. LINEAR REGRESSION
    Linear regression is one of the most commonly used machine learning algorithms for solving regression
    problems. It is a statistical method that is used to model the relationship between a dependent variable
    and one or more independent variables. The goal of linear regression is to find the best-fitting line
    that represents the relationship between the variables.

2. USE-CASES:
    2.1. House-price estimations using various variables like the area of the property, location, number of
    bedrooms, etc.
    2.2. Stock price prediction models.

3. Example
    Here's the code snippet to implement the linear regression algorithm using the sci-kit learn library:
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load the data into a Pandas dataframe
data = pd.read_parquet("data_lr.parquet")

# Split the data into training and testing sets
x = data.drop("y", axis=1)
y = data["y"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model using the training data
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the dependent variable using the test data
y_pred = regressor.predict(x_test)

# Plot the comparison results
fig = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y_test, y_pred, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()