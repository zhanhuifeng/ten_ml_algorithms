"""
1. K-NEAREST NEIGHBORS
    K-Nearest Neighbors (KNN) is a supervised learning algorithm that is used for classification and
    regression tasks. It works by finding the k-closest data points to a given data point and then
    using the labels of those data points to classify the given data point.

    KNN is commonly used for image classification, text classification, and predicting the value of
    a given data point. Some use cases are as below:

2. USE-CASES:
    1. Product recommendation system
    2. Fraud prevention

3. Example:
    Let's look at the code implementation of the K-Nearest Neighbors algorithm using the sklearn library.

"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Sample data
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])
y_classification = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_regression = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144])

# Create and fit the classification model
clf_model = KNeighborsClassifier(n_neighbors=3)
clf_model.fit(x, y_classification)

# Create and fit the regression model
reg_model = KNeighborsRegressor(n_neighbors=3)
reg_model.fit(x, y_regression)

# Predict using the models
y_pred_clf = clf_model.predict(x)
y_pred_reg = reg_model.predict(x)


# Plot the comparison results
fig1 = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y_classification, y_pred_clf, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

# Plot the comparison results
fig2 = plt.figure(figsize=(8, 5), dpi=90)
ax2 = plt.scatter(y_regression, y_pred_reg, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()