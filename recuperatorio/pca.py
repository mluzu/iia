import numpy as np


def pca(X, n_components=2):
    X = X - X.mean(axis=0)
    cov = np.cov(X.T)
    v, w = np.linalg.eig(cov)
    idx = v.argsort()[::-1]
    v = v[idx]
    w = w[:, idx]
    return v[:n_components], X @ w[:, :n_components]
