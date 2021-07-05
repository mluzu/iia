import numpy as np


class LogisticRegression:

    def __init__(self, bias):
        self.bias = bias
        self.model = None
        self.losses = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # si lambda está definido aplica regularización ridge
    def loss(self, y, y_hat, lamda=None):
        loss = np.mean(-y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))
        if lamda is None:
            return loss
        else:
            reg_loss = loss + (lamda * sum(self.model ** 2))
            return reg_loss

    def fit(self, X, y, lr, b, epochs, lamda, log=100, verbose=True):
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        m = X.shape[1]
        W = np.random.randn(m).reshape(m, 1)
        loss_list = []

        for j in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X_train = X[idx]
            y_train = y[idx]
            batch_size = int(len(X_train) / b)

            for i in range(0, len(X_train), batch_size):
                end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
                batch_X = X_train[i: end]
                batch_y = y_train[i: end]

                prediction = self.sigmoid(np.sum(np.transpose(W) * batch_X, axis=1))
                error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)
                grad_sum = np.sum(error * batch_X, axis=0)
                grad_mul = 1 / batch_size * grad_sum
                gradient = np.transpose(grad_mul).reshape(-1, 1)

                W = W - (lr * gradient)
                self.model = W
            l_epoch = self.loss(y_train, self.sigmoid(np.dot(X_train, W)), lamda)
            loss_list.append(l_epoch)
            self.losses = loss_list
            if verbose:
                if j % log == 0:
                    print("Epoch: {}, Loss: {}".format(j, l_epoch))

    def predict(self, X):
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        p = self.sigmoid(X @ self.model)
        mask_true = p >= 0.5
        mask_false = p < 0.5
        p[mask_true] = 1
        p[mask_false] = 0
        return p
