"""
Prototype sharpness ratio metric

@author Florent Forest
"""

import numpy as np


def prototype_sharpness_ratio(X, som):
    """Prototype sharpness ratio for image datasets

    Parameters
    ----------
    X : array, shape = [n_samples, input_dim]
        input samples
    som : array
        SOM prototype vectors
    """
    size = int(np.sqrt(X.shape[-1]))
    X = X.reshape(-1, size, size, 1)  # reshape to 2D images
    som = som.reshape(-1, size, size, 1)

    data_sharpness = 0.0
    for x in X:
        gx, gy = np.gradient(x[:, :, 0])
        gnorm = np.sqrt(gx ** 2 + gy ** 2)
        data_sharpness += np.mean(gnorm)
    data_sharpness /= X.shape[0]

    som_sharpness = 0.0
    for prototype in som:
        gx, gy = np.gradient(prototype[:, :, 0])
        gnorm = np.sqrt(gx ** 2 + gy ** 2)
        som_sharpness += np.mean(gnorm)
    som_sharpness /= som.shape[0]

    return som_sharpness / data_sharpness
