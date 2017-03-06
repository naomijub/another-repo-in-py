import numpy as np

class Percepton(object):
    def __init__(self, learning_rate = 0.01, n_interactions = 10):
        self.learning_rate = learning_rate
        self.n_interactions = n_interactions

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_interactions):
            errors = 0
            for x_in, target in zip(x, y):
                update = self.learning_rate * (target - self.predict(x_in))
                self.w_[1:] += update * x_in
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    def set_weight(self, x):
        self.w_ = np.zeros(1 + x.shape[1])
