"""
1. SUPPORT VECTOR MACHINES
    Support Vector Machine (SVM) is a machine learning algorithm that represents data as points in a
    high-dimensional space, called a hyperplane. The hyperplane is found that maximizes the margin
    between the training data and the margin of mis-classification on it. The algorithm compares this
    margin with a threshold called the support vector. This threshold determines how accurately each
    point will be classified as belonging to one of two classes.

    SVM has been widely used in many different applications, especially in computer vision and text
    classification. Some of them are as below:

2. USE-CASES:
    2.1. Image understanding
    2.2. Speech recognition
    2.3. Natural language processing

3. Example:
    Let's look at the code implementation of the Support Vector Machines algorithm using the sklearn
    library.

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

# Sample data
x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Create and fit the model
model = SVC(kernel="linear")
model.fit(x, y)

# Predict using the model
y_pred = model.predict(x)

# Plot the comparison results
fig1 = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y, y_pred, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
