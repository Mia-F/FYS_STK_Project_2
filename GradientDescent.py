"""
Still need some improvements, but it is a start. 
"""

import numpy as np

class GradientDescent:
    def __init__(self, X, y, learning_rate,  n_iter = 1000, threshold = 0.0001, momentum = 0.9):
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.momentum = momentum

    def gradient(self, X, theta):
        return (-2/X.shape[0]) * X.T.dot(self.y - X.dot(theta))

    def gradient_descent(self):
        theta = np.random.randn(self.X.shape[1], 1) #Setting a random theta value
        change = np.zeros_like(theta)

        for i in range(self.n_iter):
            gradients = self.gradient(self.X, theta)
            change = self.learning_rate * gradients + self.momentum * change
            theta -= change

            if np.linalg.norm(gradients) < self.threshold:
                break
        return theta
    
# Create a sample dataset
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X = np.c_[np.ones((100, 1)), X]

learning_rate = 0.1
gd = GradientDescent(X, y, learning_rate)

theta = gd.gradient_descent()
print("Calculated theta values: ", theta)
