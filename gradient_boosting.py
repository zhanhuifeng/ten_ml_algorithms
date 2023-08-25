"""
1. GRADIENT BOOSTING
    Gradient boosting trees (GBT) is a popular machine learning algorithm that is used for classification
    and regression tasks. It is an ensemble method that combines multiple decision trees to create a more
    accurate and stable model. GBT works by sequentially adding decision trees, where each new tree is trained
    to correct the errors of the previous trees. The model combines the predictions of all trees to make a
    final prediction.

    The gradient boosting algorithm is better compared to other models for regression tasks. It can handle
    multi-collinearity and non-linear relationships between variables. It is sensitive to an outlier, therefore
    can cause overfitting. Now let's look at some of its use cases.

2. USE-CASES:
    2.1. Fraud detection
    2.2. Customer Churn Prediction

3. Example:
    Let's look at the code implementation of the Gradient Boosting algorithm
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from matplotlib import pyplot as plt


# Sample data
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_classification = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# Load the data into a Pandas dataframe
data = pd.read_parquet("data_lr.parquet")
# Split the data into training and testing sets
x_reg = data.drop("y", axis=1)
y_reg = data["y"]

x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg, y_reg, test_size=0.2, random_state=0)

# Create and fit the classification model
clf_model = GradientBoostingClassifier()
clf_model.fit(x, y_classification)

# Create and fit the regression model
reg_model = GradientBoostingRegressor()
reg_model.fit(x_reg_train, y_reg_train)

# Predict using the models
y_pred_clf = clf_model.predict(x)
y_reg_pred = reg_model.predict(x_reg)


# Plot the comparison results
fig1 = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y_classification, y_pred_clf, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# Plot the comparison results
fig2 = plt.figure(figsize=(8, 5), dpi=90)
ax2 = plt.scatter(y_reg, y_reg_pred, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
