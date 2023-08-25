"""
1. LOGISTICS REGRESSION
    Logistic regression is a type of regression analysis that is used for solving classification
    problems. It is a statistical method that is used to model the relationship between a dependent
    variable and one or more independent variables. It used the 'logit' function to classify the
    outcome of input into two categories. Unlike linear regression, logistic regression is used to
    predict a binary outcome, such as yes/no or true/false.
2. USE-CASES:
    2.1. Credit risk classification
    2.2. Fraud detection
    2.3. Medical diagnosis classification
3. Example
    Let's look at the code implementation of the logistics regression algorithm using the sklearn library.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load the data into a Pandas dataframe
data = pd.read_parquet("data_lr.parquet")
data["y"] = data["y"].mask(data["y"] < 5, 0)
data["y"] = data["y"].mask(data["y"] >= 5, 1)
# Split the data into training and testing sets
x = data.drop("y", axis=1)
y = data["y"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model using the training data
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predict the dependent variable using the test data
y_pred = classifier.predict(x_test)

# Plot the comparison results
fig = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y_test, y_pred, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
