"""
1. ARTIFICIAL NEURAL NETWORKS
    Artificial Neural Networks (ANNs) are a type of supervised learning algorithm that inspired by the
    biological neurons in the human brain. They are used for complex tasks such as image recognition,
    natural language processing, and speech recognition.

    ANNs are composed of multiple interconnected neurons which are organized into layers, with each neuron
    in a layer having a weight and a bias associated with it. When given an input, the neurons process the
    information and output a prediction.

    There are types of neural networks used in a variety of applications. Convolutional Neural Networks are
    used in image classification, object detection, and segmentation tasks while Recurrent Neural Networks
    are used in language modeling tasks. Let's look at some use cases of ANNs.

2. USE-CASES:
    1. Image classification tasks
    2. Text classification
    3. Language Translation
    4. Language detection

3. Example:
    Let's look at the code implementation of the Artificial Neural Networks algorithm.

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

# Sample data
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Create the model
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and fit the model
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.fit(x, y, epochs=100, batch_size=1)

# Predict using the model
y_pred = model.predict(x)

# Plot the comparison results
fig1 = plt.figure(figsize=(8, 5), dpi=90)
ax1 = plt.scatter(y, y_pred, s=20)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()
