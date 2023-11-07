import numpy as np
from sklearn.linear_model import SGDClassifier

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_derivative(X_batch, y_batch, theta):
    predictions = sigmoid(np.dot(X_batch, theta))
    gradient = np.dot(X_batch.T, (predictions - y_batch)) / X_batch.shape[0]
    return gradient

def stochastic_gradient_descent(X, y, derivative, batch_size=10, epochs=50, learning_rate=0.01):
    n = X.shape[0]
    theta_sgd = np.random.randn(X.shape[1], 1)  # Adjusted to match feature count of X
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size].reshape(-1, 1)
            gradients = derivative(X_batch, y_batch, theta_sgd)
            theta_sgd -= learning_rate * gradients

    return theta_sgd

# Logistic regression using SGD for XOR gate
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float64)
yXOR = np.array([0, 1, 1, 0])
theta_XOR_sgd = stochastic_gradient_descent(X, yXOR, logistic_derivative, batch_size=1, epochs=10000, learning_rate=0.01)

# Logistic regression using SGD for OR gate
yOR = np.array([0, 1, 1, 1])
theta_OR_sgd = stochastic_gradient_descent(X, yOR, logistic_derivative, batch_size=1, epochs=10000, learning_rate=0.01)

# Logistic regression using SGD for AND gate
yAND = np.array([0, 0, 0, 1])
theta_AND_sgd = stochastic_gradient_descent(X, yAND, logistic_derivative, batch_size=1, epochs=10000, learning_rate=0.01)

# Print the results
print("Theta for XOR gate using SGD:", theta_XOR_sgd.flatten())
print("Theta for OR gate using SGD:", theta_OR_sgd.flatten())
print("Theta for AND gate using SGD:", theta_AND_sgd.flatten())
