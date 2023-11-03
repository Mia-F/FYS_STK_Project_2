import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StochasticGradientDecent:
    """Doc"""

    def __init__(self, X, y):
        self._X = X
        self._y = y
        self._n = X.shape[0]
        self._p = X.shape[1]

    def _eta(self, t0, t1, t):
        return t0 / (t+t1)

    def learning_schedule(self, args):
        match args[0]:
            case 'constant':
                eta = self._eta(args[1], args[2], args[3])
            case 'eig':
                xtx = self._X.T @ self._X
                hessian = (2/self.n) * xtx
                eigenval, _ = np.linalg.eig(hessian)
                eta = 1/np.max(eigenval)
            case _:
                print("Not a valid learning schedule")
                return
        return eta
    

    def fit(self, n_epoch, batch_size, learning_args):
        n_batch = self._n // batch_size
        theta = np.random.randn(2, 1)


        for e in range(n_epoch):
            for i in range(n_batch):
                rand_idx = batch_size * np.random.randint(n_batch)
                Xi = self._X[rand_idx: rand_idx+batch_size]
                yi = self._y[rand_idx: rand_idx+batch_size]
                gradients = (2/batch_size) * Xi.T @ ((Xi @ theta) - yi)
                # t = learning_args.copy()
                t = e*n_batch + i
                # eta = self.learning_schedule(t)
                eta = 5 / (t+50)
                theta = theta - eta*gradients
        return theta


    def predict(self, theta):
        X_new = np.random.rand(self._n, self._p)
        predict = X_new.dot(theta)
        return predict
    

def main():
    # X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]], dtype=np.float64)
    # XOR
    # y = np.array([ 0, 1 ,1, 0])
    # OR
    # y = np.array([0, 1 ,1, 1])
    # AND
    # y = np.array([0, 0 ,0, 1])
    x = 2*np.random.rand(100, 1)
    x = np.sort(x)
    X = np.c_[np.ones_like(x), x]
    y = 4+3*x+np.random.randn(100, 1)
    model = StochasticGradientDecent(X, y)
    learning_args = ['constant']
    theta = model.fit(n_epoch=50, batch_size=5, learning_args=learning_args)
    y_predict = model.predict(theta=theta)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="black")
    ax.scatter(x, y_predict, color="seagreen")
    plt.show()

if __name__ == '__main__':
    main()