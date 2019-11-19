"""
Implementation of the Kerasom model (standard SOM in Keras)
Main file

@author Florent Forest
@version 2.0
"""

# Utilities
import os
import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.utils.vis_utils import plot_model

# Dataset helper function
from datasets import load_data

# Kerasom components
from SOM import SOMLayer
from evaluation import PerfLogger


def som_loss(weights, distances):
    """SOM loss

    Parameters
    ----------
    weights : Tensor, shape = [n_samples, n_prototypes]
        weights for the weighted sum,
    distances : Tensor ,shape = [n_samples, n_prototypes]
        pairwise squared euclidean distances between inputs and prototype vectors

    Returns
    -------
    som_loss : loss
        SOM distortion loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights * distances, axis=1))


class Kerasom:
    """Kerasom model (standard SOM in Keras)

    Example
    -------
    ```
    kerasom = Kerasom(input_dim=784, map_size=(10,10))
    ```

    Parameters
    ----------
    input_dim: int
        input vector dimension
    map_size : tuple
        size of the rectangular map. Number of prototypes is map_size[0] * map_size[1]
    """

    def __init__(self, input_dim, map_size):
        self.input_dim = input_dim
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.input = None
        self.model = None
    
    def initialize(self):
        """Initialize Kerasom model"""
        self.input = Input(shape=(self.input_dim,), name='input')
        som_layer = SOMLayer(self.map_size, name='SOM')(self.input)
        self.model = Model(inputs=self.input, outputs=som_layer)

    @property
    def prototypes(self):
        """SOM code vectors"""
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, optimizer):
        """Compile Kerasom model

        Parameters
        ----------
        optimizer : str (default='adam')
            optimization algorithm
        """
        self.model.compile(loss=som_loss, optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """Load pre-trained weights of Kerasom model"""
        self.model.load_weights(weights_path)

    def init_som_weights(self, X):
        """Initialize with a sample without replacement of encoded data points.

        Parameters
        ----------
        X : array, shape = [n_samples, input_dim]
            training set or batch
        """
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        self.model.get_layer(name='SOM').set_weights([sample])

    def predict(self, x):
        """Predict best-matching unit using the output of SOM layer

        Parameters
        ----------
        x : array, shape = [n_samples, input_dim]
            input samples

        Returns
        -------
        y_pred : array, shape = [n_samples}
            index of the best-matching unit
        """
        d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)

    def map_dist(self, y_pred):
        """Calculate pairwise Manhattan distances between cluster assignments and map prototypes
        (rectangular grid topology)

        Parameters
        ----------
        y_pred : array, shape = [n_samples]
            cluster assignments

        Returns
        -------
        d : array, shape = [n_samples, n_prototypes]
            pairwise distance matrix on the map

        See also
        --------
        `somperf.utils.topology.rectangular_topology_dist`
        """
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp - labels) // self.map_size[1]
        d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
        return d_row + d_col

    @staticmethod
    def neighborhood_function(d, T, neighborhood='gaussian'):
        """SOM neighborhood function (gaussian neighborhood)

        Parameters
        ----------
        d : int
            distance on the map
        T : float
            temperature parameter (neighborhood radius)
        neighborhood : str
            type of neighborhood function ('gaussian' or 'window')

        Returns
        -------
        w : float in [0, 1]
            neighborhood weight
        See also
        --------
        `somperf.utils.neighborhood`
        """
        if neighborhood == 'gaussian':
            return np.exp(-(d ** 2) / (T ** 2))
        elif neighborhood == 'window':
            return (d <= T).astype(np.float32)
        else:
            raise ValueError('invalid neighborhood function')

    @staticmethod
    def batch_generator(X_train, y_train, X_val, y_val, batch_size):
        """Generate training and validation batches"""
        X_batch, y_batch, X_val_batch, y_val_batch = None, None, None, None

        index = 0
        if X_val is not None:
            index_val = 0

        while True:  # generate batches
            if (index + 1) * batch_size > X_train.shape[0]:
                X_batch = X_train[index * batch_size::]
                if y_train is not None:
                    y_batch = y_train[index * batch_size::]
                index = 0
            else:
                X_batch = X_train[index * batch_size:(index + 1) * batch_size]
                if y_train is not None:
                    y_batch = y_train[index * batch_size:(index + 1) * batch_size]
                index += 1
            if X_val is not None:
                if (index_val + 1) * batch_size > X_val.shape[0]:
                    X_val_batch = X_val[index_val * batch_size::]
                    if y_val is not None:
                        y_val_batch = y_val[index_val * batch_size::]
                    index_val = 0
                else:
                    X_val_batch = X_val[index_val * batch_size:(index_val + 1) * batch_size]
                    if y_val is not None:
                        y_val_batch = y_val[index_val * batch_size:(index_val + 1) * batch_size]
                    index_val += 1
            yield (X_batch, y_batch), (X_val_batch, y_val_batch)

    def fit(self,
            X_train,
            y_train=None,
            X_val=None,
            y_val=None,
            iterations=10000,
            som_iterations=10000,
            eval_interval=10,
            save_epochs=5,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential',
            neighborhood='gaussian',
            save_dir='results/tmp',
            verbose=1):
        """Training procedure

        Parameters
        ----------
        X_train : array, shape = [n_samples, input_dim]
            training set
        y_train : array, shape = [n_samples]
            (optional) training labels
        X_val : array, shape = [n_samples, input_dim]
            (optional) validation set
        y_val : array, shape = [n_samples]
            (optional) validation labels
        iterations : int (default=10000)
            number of training iterations
        som_iterations : int (default=10000)
            number of iterations where SOM neighborhood is decreased
        eval_interval : int (default=10)
            evaluate metrics on training/validation batch every eval_interval iterations
        save_epochs : int (default=5)
            save model weights every save_epochs epochs
        batch_size : int (default=256)
            training batch size
        Tmax : float (default=10.0)
            initial temperature parameter (neighborhood radius)
        Tmin : float (default=0.1)
            final temperature parameter (neighborhood radius)
        decay : str (default='exponential')
            type of temperature decay ('exponential' or 'linear')
        neighborhood : str (default='gaussian')
            type of neighborhood function ('gaussian' or 'window')
        save_dir : str (default='results/tmp'
            path to existing directory where weights and logs are saved
        verbose : int (default=1)
            verbosity level (0, 1 or 2)
        """
        save_interval = X_train.shape[0] // batch_size * save_epochs  # save every save_epochs epochs
        print('Save interval:', save_interval)

        # Initialize perf logging
        perflogger = PerfLogger(with_validation=(X_val is not None),
                                with_labels=(y_train is not None),
                                with_latent_metrics=False,
                                save_dir=save_dir)

        # Initialize batch generator
        batch = self.batch_generator(X_train, y_train, X_val, y_val, batch_size)

        # Training loop
        for ite in range(iterations):
            (X_batch, y_batch), (X_val_batch, y_val_batch) = next(batch)

            # Compute cluster assignments for batches
            d = self.model.predict(X_batch)
            y_pred = d.argmin(axis=1)
            if X_val is not None:
                d_val = self.model.predict(X_val_batch)
                y_val_pred = d_val.argmin(axis=1)

            # Update temperature parameter
            if ite < som_iterations:
                if decay == 'exponential':
                    T = Tmax * (Tmin / Tmax) ** (ite / (som_iterations - 1))
                elif decay == 'linear':
                    T = Tmax - (Tmax - Tmin) * (ite / (som_iterations - 1))
                elif decay == 'constant':
                    T = Tmax
                else:
                    raise ValueError('invalid decay function')

            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred), T, neighborhood)
            if X_val is not None:
                w_val_batch = self.neighborhood_function(self.map_dist(y_val_pred), T, neighborhood)

            # Train on batch
            loss = self.model.train_on_batch(X_batch, w_batch)

            # Evaluate and log monitored metrics
            if ite % eval_interval == 0:

                if X_val is not None:
                    val_loss = self.model.test_on_batch(X_val_batch, w_val_batch)

                batch_summary = {
                    'map_size': self.map_size,
                    'iteration': ite,
                    'T': T,
                    'loss': [loss],
                    'val_loss': [val_loss] if X_val is not None else None,
                    'd_original': np.sqrt(d),
                    'd_original_val': np.sqrt(d_val) if X_val is not None else None,
                    'prototypes': self.prototypes,
                    'X': X_batch,
                    'X_val': X_val_batch,
                    'y_true': y_batch,
                    'y_pred': y_pred,
                    'y_val_true': y_val_batch,
                    'y_val_pred': y_val_pred if X_val is not None else None,
                }

                perflogger.log(batch_summary, verbose=verbose)

            # Save intermediate model
            if ite % save_interval == 0:
                self.model.save_weights(save_dir + '/kerasom_model_' + str(ite) + '.h5')
                print('Saved model to:', save_dir + '/kerasom_model_' + str(ite) + '.h5')

        # Save the final model
        print('saving model to:', save_dir + '/kerasom_model_final.h5')
        self.model.save_weights(save_dir + '/kerasom_model_final.h5')

        # Evaluate model on entire dataset
        print('Evaluate model on training and/or validation datasets')

        d = self.model.predict(X_train)
        y_pred = d.argmin(axis=1)
        if X_val is not None:
            d_val = self.model.predict(X_val)
            y_val_pred = d_val.argmin(axis=1)

        final_summary = {
            'map_size': self.map_size,
            'iteration': iterations,
            'd_original': np.sqrt(d),
            'd_original_val': np.sqrt(d_val) if X_val is not None else None,
            'prototypes': self.prototypes,
            'X': X_train,
            'X_val': X_val,
            'y_true': y_train,
            'y_pred': y_pred,
            'y_val_true': y_val,
            'y_val_pred': y_val_pred if X_val is not None else None,
        }
        perflogger.evaluate(final_summary, verbose=verbose)
        perflogger.close()
