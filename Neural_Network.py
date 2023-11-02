

import numpy as np
import matplotlib.pyplot as plt

class Neural_Network:
    def __init__(self, X, target, n_hidden_neurons) -> None:
        self.n_categories = len(np.unique(labels))
        self.n_hidden_neurons = n_hidden_neurons
        self.X = X
        self.n_inputs, self.n_features = X.shape
        self.target = target

    def setting_parameters(self):
        self.hidden_weights =  np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))  
  
    
    def feed_forward(self):
        #wighted sum of inputs to the hidden layer
        self.z_h = np.matmul(self.X, self.hidden_weights) + self.hidden_bias

        #activation in the hidden layer
        self.a_h = self.sigmoid(self.z_h)
       

        self.z_o =  np.matmul(self.a_h, self.output_weights) + self.output_bias
        self.a_o_h = np.heaviside(self.z_o,0)
        print(self.a_o_h[:,0])
        res = self.a_o_h[:,0]
        acc = self.accuracy_score(res)
        print(acc)
        probabilities = self.sigmoid(self.z_o)
        return probabilities

    def predict(self):
        probabilities = self.feed_forward()
        self.prediction = np.argmax(probabilities, axis=1)
        return self.prediction
    
    def accuracy_score(self, result):
        return np.sum(self.target == result)/len(self.target)

    

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
load_data = load_breast_cancer()   
data = load_data.data
target = load_data.target

train_size = 0.8 
test_size = 1 - train_size

inputs = load_data.data
labels = load_data.target

X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)

NN = Neural_Network(X_train, Y_train, 2)

#NN = Neural_Network(input, target, 2)
NN.setting_parameters()
prediction = NN.predict()

#print(prediction)

#print(labels)


