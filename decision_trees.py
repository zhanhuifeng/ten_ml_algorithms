"""
1. DECISION TREES
    Decision Trees are one of the most popular machine learning algorithms. They are used for classification,
    regression, and anomaly detection. Decision trees set up a hierarchy of decisions based on the outcome
    of the test data. Each decision is made by choosing a split at some point in the tree.

    The decision tree algorithm is useful because it can be easily visualized as a series of splits and leaf
    nodes, which helps understand how to make a decision in an ambiguous situation.

    Decision trees are widely used because they are interpretable as opposed to black box algorithms like
    Neural Networks, gradient boosting trees, etc.

2. USE-CASES:
    2.1. Loan approval classification
    2.2. Student graduation rate classification
    2.3. Medical expenses prediction
    2.4. Customer churn prediction

3. Example:
    Let's look at the code implementation of the Decision Trees algorithm using the sklearn library.

"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Sample data
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_classification = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y_regression = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

# Create and fit the classification model
clf_model = DecisionTreeClassifier()
clf_model.fit(x, y_classification)

# Create and fit the regression model
reg_model = DecisionTreeRegressor()
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
