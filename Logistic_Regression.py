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
    theta_sgd = np.random.randn(2, 1)
    
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

# weights = stochastic_gradient_descent(X_train, y_train, logistic_derivative, batch_size=10, epochs=50, learning_rate=0.01)

# Create an instance of SGDClassifier which applies Stochastic Gradient Descent
sgd_clf = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01, max_iter=1000)

# Fit the model to your data
# sgd_clf.fit(X_train, y_train)
# weights = sgd_logistic_regression(X_train, y_train, learning_rate=0.01, max_iter=1000)
