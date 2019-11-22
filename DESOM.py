"""
Implementation of the Deep Embedded Self-Organizing Map model
Model file

@author Florent Forest
@version 2.0
"""

# Utilities
from time import time
import numpy as np

# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model

# DESOM components
from SOM import SOMLayer
from AE import mlp_autoencoder
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


def kmeans_loss(y_pred, distances):
    """k-means loss

    Parameters
    ----------
    y_pred : array, shape = [n_samples]
        cluster assignments
    distances : array, shape = [n_samples, n_prototypes]
        pairwise squared euclidean distances between inputs and prototype vectors

    Returns
    -------
        kmeans_loss : float
            k-means reconstruction loss
    """
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])


class DESOM:
    """Deep Embedded Self-Organizing Map (DESOM) model

    Example
    -------
    ```
    desom = DESOM(encoder_dims=[784, 500, 500, 2000, 10], map_size=(10,10))
    ```

    Parameters
    ----------
    encoder_dims : list
        number of units in each layer of encoder. dims[0] is input dim, dims[-1] is the latent code dimension
    map_size : tuple
        size of the rectangular map. Number of prototypes is map_size[0] * map_size[1]
    """

    def __init__(self, encoder_dims, map_size):
        self.encoder_dims = encoder_dims
        self.input_dim = self.encoder_dims[0]
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.pretrained = False
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.model = None
    
    def initialize(self, ae_act='relu', ae_init='glorot_uniform', batchnorm=False):
        """Initialize DESOM model

        Parameters
        ----------
        ae_act : str (default='relu')
            activation for AE intermediate layers
        ae_init : str (default='glorot_uniform')
            initialization of AE layers
        batchnorm : bool (default=False)
            use batch normalization
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = mlp_autoencoder(self.encoder_dims, ae_act, ae_init, batchnorm)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create DESOM model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.autoencoder.output, som_layer])
    
    @property
    def prototypes(self):
        """SOM code vectors"""
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, gamma, optimizer='adam'):
        """Compile DESOM model

        Parameters
        ----------
        gamma : float
            coefficient of SOM loss (hyperparameter)
        optimizer : str (default='adam')
            optimization algorithm
        """
        self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                           loss_weights=[1, gamma],
                           optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """Load pre-trained weights of DESOM model

        Parameters
        ----------
        weights_path : str
            path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """Load pre-trained weights of AE

        Parameters
        ----------
        ae_weights_path : str
            path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def init_som_weights(self, X):
        """Initialize with a sample without remplacement of encoded data points.

        Parameters
        ----------
        X : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            training set or batch
        """
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])

    def encode(self, x):
        """Encoding function. Extracts latent code from hidden layer

        Parameters
        ----------
        x : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            input samples

        Returns
        -------
        z : array, shape = [n_samples, latent_dim]
            encoded (latent) samples
        """
        return self.encoder.predict(x)
    
    def decode(self, z):
        """Decoding function. Decodes encoded features from latent space

        Parameters
        ----------
        z : array, shape = [n_samples, latent_dim]
            encoded (latent) samples

        Returns
        -------
        x : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            decoded samples
        """
        return self.decoder.predict(z)

    def predict(self, x):
        """Predict best-matching unit using the output of SOM layer

        Parameters
        ----------
        x : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            input samples

        Returns
        -------
        y_pred : array, shape = [n_samples]
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose=0)
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
    
    def pretrain(self,
                 X,
                 optimizer='adam',
                 epochs=200,
                 batch_size=256,
                 save_dir='results/tmp'):
        """Pre-train the autoencoder using only MSE reconstruction loss. Saves weights in h5 format

        Parameters
        ----------
        X : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            training set
        optimizer : str (default='adam')
            optimization algorithm
        epochs : int (default=200)
            number of pre-training epochs
        batch_size : int (default=256)
            training batch size
        save_dir : str (default='results/tmp')
            path to existing directory where weights will be saved
        """
        print('Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # Begin pretraining
        t0 = time()
        self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True

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
        X_train : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
            training set
        y_train : array, shape = [n_samples]
            (optional) training labels
        X_val : array, shape = [n_samples, input_dim] or [n_samples, height, width, channels]
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
            type of temperature decay ('exponential', 'linear' or 'constant')
        neighborhood : str (default='gaussian')
            type of neighborhood function ('gaussian' or 'window')
        save_dir : str (default='results/tmp'
            path to existing directory where weights and logs are saved
        verbose : int (default=1)
            verbosity level (0, 1 or 2)
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        save_interval = X_train.shape[0] // batch_size * save_epochs  # save every save_epochs epochs
        print('Save interval:', save_interval)

        # Initialize perf logging
        perflogger = PerfLogger(with_validation=(X_val is not None),
                                with_labels=(y_train is not None),
                                with_latent_metrics=True,
                                save_dir=save_dir)

        # Initialize batch generator
        batch = self.batch_generator(X_train, y_train, X_val, y_val, batch_size)

        # Training loop
        for ite in range(iterations):
            (X_batch, y_batch), (X_val_batch, y_val_batch) = next(batch)

            # Compute cluster assignments for batch
            _, d = self.model.predict(X_batch)
            y_pred = d.argmin(axis=1)
            if X_val is not None:
                _, d_val = self.model.predict(X_val_batch)
                y_val_pred = d_val.argmin(axis=1)

            # Update temperature parameter
            if ite < som_iterations:
                if decay == 'exponential':
                    T = Tmax * (Tmin / Tmax)**(ite / (som_iterations - 1))
                elif decay == 'linear':
                    T = Tmax - (Tmax - Tmin)*(ite / (som_iterations - 1))
                elif decay == 'constant':
                    T = Tmax
                else:
                    raise ValueError('invalid decay function')
            
            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred), T, neighborhood)
            if X_val is not None:
                w_val_batch = self.neighborhood_function(self.map_dist(y_val_pred), T, neighborhood)

            # Train on batch
            loss = self.model.train_on_batch(X_batch, [X_batch, w_batch])

            # Evaluate and log monitored metrics
            if ite % eval_interval == 0:

                # Get SOM weights and decode to original space
                decoded_prototypes = self.decode(self.prototypes)
                decoded_prototypes = decoded_prototypes.reshape(decoded_prototypes.shape[0], -1)
                # Compute pairwise squared euclidean distance matrix in original space
                d_original = np.square((np.expand_dims(X_batch.reshape(X_batch.shape[0], -1), axis=1)
                                        - decoded_prototypes)).sum(axis=2)
                if X_val is not None:
                    val_loss = self.model.test_on_batch(X_val_batch, [X_val_batch, w_val_batch])
                    d_original_val = np.square((np.expand_dims(X_val_batch.reshape(X_val_batch.shape[0], -1), axis=1)
                                                - decoded_prototypes)).sum(axis=2)

                batch_summary = {
                    'map_size': self.map_size,
                    'iteration': ite,
                    'T': T,
                    'loss': loss,
                    'val_loss': val_loss if X_val is not None else None,
                    'd_latent': np.sqrt(d),
                    'd_original': np.sqrt(d_original),
                    'd_latent_val': np.sqrt(d_val) if X_val is not None else None,
                    'd_original_val': np.sqrt(d_original_val) if X_val is not None else None,
                    'prototypes': decoded_prototypes,
                    'latent_prototypes': self.prototypes,
                    'X': X_batch.reshape(X_batch.shape[0], -1),
                    'X_val': X_val_batch.reshape(X_val_batch.shape[0], -1) if X_val is not None else None,
                    'Z': self.encode(X_batch),
                    'Z_val': self.encode(X_val_batch) if X_val is not None else None,
                    'y_true': y_batch,
                    'y_pred': y_pred,
                    'y_val_true': y_val_batch,
                    'y_val_pred': y_val_pred if X_val is not None else None,
                }

                perflogger.log(batch_summary, verbose=verbose)

            # Save intermediate model
            if ite % save_interval == 0:
                self.model.save_weights(save_dir + '/DESOM_model_' + str(ite) + '.h5')
                print('Saved model to:', save_dir + '/DESOM_model_' + str(ite) + '.h5')

        # Save the final model
        print('Saving final model to:', save_dir + '/DESOM_model_final.h5')
        self.model.save_weights(save_dir + '/DESOM_model_final.h5')

        # Evaluate model on entire dataset
        print('Evaluate model on training and/or validation datasets')

        _, d = self.model.predict(X_train)
        y_pred = d.argmin(axis=1)
        if X_val is not None:
            _, d_val = self.model.predict(X_val)
            y_val_pred = d_val.argmin(axis=1)

        # Get SOM weights and decode to original space
        decoded_prototypes = self.decode(self.prototypes)
        decoded_prototypes = decoded_prototypes.reshape(decoded_prototypes.shape[0], -1)
        # Compute pairwise squared euclidean distance matrix in original space
        d_original = np.square((np.expand_dims(X_train.reshape(X_train.shape[0], -1), axis=1)
                                - decoded_prototypes)).sum(axis=2)
        if X_val is not None:
            d_original_val = np.square((np.expand_dims(X_val.reshape(X_val.shape[0], -1), axis=1)
                                        - decoded_prototypes)).sum(axis=2)

        final_summary = {
            'map_size': self.map_size,
            'iteration': iterations,
            'd_latent': np.sqrt(d),
            'd_original': np.sqrt(d_original),
            'd_latent_val': np.sqrt(d_val) if X_val is not None else None,
            'd_original_val': np.sqrt(d_original_val) if X_val is not None else None,
            'prototypes': decoded_prototypes,
            'latent_prototypes': self.prototypes,
            'X': X_train.reshape(X_train.shape[0], -1),
            'X_val': X_val.reshape(X_val.shape[0], -1) if X_val is not None else None,
            'Z': self.encode(X_train),
            'Z_val': self.encode(X_val) if X_val is not None else None,
            'y_true': y_train,
            'y_pred': y_pred,
            'y_val_true': y_val,
            'y_val_pred': y_val_pred if X_val is not None else None,
        }
        perflogger.evaluate(final_summary, verbose=verbose)
        perflogger.close()
