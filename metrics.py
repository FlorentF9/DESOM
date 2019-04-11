"""
Clustering metrics functions

@author Florent Forest
@version 1.0
"""

import numpy as np
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment

def cluster_acc(y_true, y_pred):
    """
    Calculate unsupervised clustering accuracy. Requires scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def cluster_purity(y_true, y_pred):
    """
    Calculate clustering purity

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return metrics.accuracy_score(y_pred_voted, y_true)

def quantization_error(d):
    """
    Calculate k-means quantization error (internal DESOM function)
    """
    return d.min(axis=1).mean()

def topographic_error(d, map_size):
    """
    Calculate SOM topographic error (internal DESOM function)
    Topographic error is the ratio of data points for which the two best matching units are neighbots on the map.
    """
    h,w = map_size
    def is_adjacent(k,l):
        return (abs(k//w-l//w) == 1 and abs(k%w-l%w) == 0) or (abs(k//w-l//w) == 0 and abs(k%w-l%w) == 1)
    btmus = np.argsort(d, axis=1)[:,:2] # best two matching units
    return 1.-np.mean([is_adjacent(btmus[i,0], btmus[i,1]) for i in range(d.shape[0])])
