import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Dataset:

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('x_1', np.float), ('x_2', np.float), ('x_3', np.float),
                     ('x_4', np.float), ('x_5', np.float), ('x_6', np.float),
                     ('x_7', np.float), ('x_8', np.float), ('x_9', np.float),
                     ('x_10', np.float), ('y', np.int)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[1]), float(line.split(',')[2]), float(line.split(',')[3]),
                         float(line.split(',')[4]), float(line.split(',')[5]), float(line.split(',')[6]),
                         float(line.split(',')[7]), float(line.split(',')[8]), float(line.split(',')[9]),
                         float(line.split(',')[10]), float(line.split(',')[11]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def muestras(self):
        return len(self.dataset)

    def features(self):
        return len(self.dataset[0])

    def find_missing(self):
        return np.isnan(self.dataset)

    def split(self, percentage):
        X = np.array([
            self.dataset['x_1'], self.dataset['x_2'], self.dataset['x_3'],
            self.dataset['x_4'], self.dataset['x_5'], self.dataset['x_6'],
            self.dataset['x_7'], self.dataset['x_8'], self.dataset['x_9'],
            self.dataset['x_10']
        ]).T
        y = self.dataset['y']

        permuted_idxs = np.random.permutation(X.shape[0])

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test

    def array_X_y(self):
        X = np.array([
            self.dataset['x_1'], self.dataset['x_2'], self.dataset['x_3'],
            self.dataset['x_4'], self.dataset['x_5'], self.dataset['x_6'],
            self.dataset['x_7'], self.dataset['x_8'], self.dataset['x_9'],
            self.dataset['x_10']
        ]).T
        y = np.array(self.dataset['y'])
        return X, y


class PCA:

    def __init__(self, X):
        self.X = X
        self.eigen_values = None
        self.eigen_vectors = None

    def variance_explained(self):
        variance_explained = []
        for i in self.eigen_values:
            variance_explained.append((i / sum(self.eigen_values))*100)
        return variance_explained

    def fit(self, n_components=2):
        X = self.X - np.mean(self.X, axis=0)
        cov = np.cov(X.T)
        v, w = np.linalg.eig(cov)
        idx = v.argsort()[::-1]
        self.eigen_values = v[idx]
        self.eigen_vectors = w[:, idx]
        print("Eigenvector: \n", self.eigen_values, "\n")
        print("Eigenvalues: \n", self.eigen_vectors, "\n")
        projection_matrix = (self.eigen_vectors.T[:][:n_components]).T
        return X @ projection_matrix

    def plot(self):
        variance_explained = self.variance_explained()
        cumulative_variance_explained = np.cumsum(variance_explained)
        print(cumulative_variance_explained)
        sns.lineplot(x=range(len(self.eigen_values)), y=cumulative_variance_explained)
        plt.xlabel("Número de componentes")
        plt.ylabel("Varianza explicada")
        plt.title("Varianza explicada vs Número de componentes")
        plt.show(block=False)
        plt.show()


def split(X, y, percentage):

    permuted_idxs = np.random.permutation(X.shape[0])

    train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

    test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

    X_train = X[train_idxs]
    X_test = X[test_idxs]

    y_train = y[train_idxs]
    y_test = y[test_idxs]

    return X_train, X_test, y_train, y_test

