import numpy as np

class Adaline(object):
    def __init__(self, learning_rate = 0.3, n_interactions = 60):
        self.learning_rate = learning_rate
        self.n_interactions = n_interactions

    def fit(self,x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_interactions):
            output = self.net_input(x)
            error = (y - output)
            self.w_[1:] += self.learning_rate * x.T.dot(error)
            self.w_[0] += self.learning_rate + error.sum()
            cost = (error ** 2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)
