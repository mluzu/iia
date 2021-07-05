import numpy as np
from metrics import Accuracy, Recall, Precision
from models import LogisticRegression


# Con lamda distinto de None usa ridge
def lg_k_folds(X_train, y_train, lr, b, epochs, lamda, bias, k=5, verbose=False):
    results = {
        'accuracy': [],
        'recall': [],
        'precision': []
    }
    metric_means = {}
    accuracy = Accuracy()
    recall = Recall()
    precision = Precision()
    chunk_size = int(len(X_train) / k)

    logistic_regression = LogisticRegression(bias)

    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])
        logistic_regression.fit(new_X_train, new_y_train,  lr, b, epochs, lamda, verbose=verbose)
        predictions = logistic_regression.predict(new_X_valid)

        results['accuracy'].append(accuracy(new_y_valid, predictions))
        results['recall'].append(recall(new_y_valid, predictions))
        results['precision'].append(precision(new_y_valid, predictions))

    metric_means['accuracy'] = np.mean(results['accuracy'])
    metric_means['recall'] = np.mean(results['recall'])
    metric_means['precision'] = np.mean(results['precision'])

    return metric_means
