"""
1. Navie Bayes is a probabilistic inference algorithm for continuous (rather than discrete) data.
    It's also known as Bayes' theorem, Bayesian inference, and Bayes'rule. In its simplest from,
    Navie Bayes assumes that the conditional probability of an event given evidence A is proportional
    to the product of two terms:

    P(A|B) = (P(A)*P(B|A))/P(B)

    The first term represents the probability of A given B, while the second term represents the probability
    of B given A, multiplied by the probability of A whole divided by the probability of B.

    The Naive Bayes algorithm is used widely in text data classification given the amount of data available
    in a text corpus. The algorithm assumes all the input variables are independent of each which is the
    reason it is called a Naive Bayes algorithm. Let's look at some of its use cases.

2. USE-CASES:
    2.1. Document classification (e.g. newspaper article category classification)
    2.2. Email spam classification
    2.3. Fraud detection

3. Example:
    Let's look at the code implementation of the Naive Bayes algorithm.

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

# # Sample data
# x = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]])
# y = np.array([0, 0, 1, 1, 1])

# Load the data into a Pandas dataframe
data = pd.read_parquet("data_lr.parquet")
data["y"] = data["y"].mask(data["y"] < 5, 0)
data["y"] = data["y"].mask(data["y"] >= 5, 1)
x = data.drop("y", axis=1)
y = data["y"]
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and fit the model
model = GaussianNB()
model.fit(x_train, y_train)

# Predict using the model
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

# Plot the comparison results
fig = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y_test, y_pred_test, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

# Plot the comparison results
fig2 = plt.figure(figsize=(8, 5), dpi=90)
ax2 = plt.scatter(y_train, y_pred_train, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
