"""
Implementation of the Deep Embedded Self-Organizing Map model
Main file

@author Florent Forest
@version 2.0
"""

# Utilities
import os
import csv
import argparse
from time import time
import matplotlib.pyplot as plt

# Tensorflow/Keras
import tensorflow as tf
from keras.models import Model
from keras.utils.vis_utils import plot_model

# Dataset helper function
from datasets import load_data

# DESOM components
from SOM import SOMLayer
from AE import mlp_autoencoder
from metrics import *


def som_loss(weights, distances):
    """
    SOM loss

    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    return tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))


def kmeans_loss(y_pred, distances):
    """
    k-means reconstruction loss

    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
    # Return
        k-means reconstruction loss
    """
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])


class DESOM:
    """
    Deep Embedded Self-Organizing Map (DESOM) model

    # Example
        ```
        desom = DESOM(encoder_dims=[784, 500, 500, 2000, 10], map_size=(10,10))
        ```

    # Arguments
        encoder_dims: list of numbers of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer (latent dim)
        map_size: tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1]
    """

    def __init__(self, encoder_dims, map_size):
        self.encoder_dims = encoder_dims
        self.input_dim = self.encoder_dims[0]
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.pretrained = False
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.model = None
    
    def initialize(self, ae_act='relu', ae_init='glorot_uniform'):
        """
        Create DESOM model

        # Arguments
            ae_act: activation for AE intermediate layers
            ae_init: initialization of AE layers
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = mlp_autoencoder(self.encoder_dims, ae_act, ae_init)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create DESOM model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.autoencoder.output, som_layer])
    
    @property
    def prototypes(self):
        """
        Returns SOM code vectors
        """
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, gamma, optimizer):
        """
        Compile DESOM model

        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        self.model.compile(loss={'decoder_0': 'mse', 'SOM': som_loss},
                           loss_weights=[1, gamma],
                           optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DESOM model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE

        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def init_som_weights(self, X):
        """
        Initialize with a sample w/o remplacement of encoded data points.

        # Arguments
            X: numpy array containing training set or batch
        """
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer

        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)
    
    def decode(self, x):
        """
        Decoding function. Decodes encoded features from latent space

        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer

        # Arguments
            x: data point
        # Return
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1)

    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes (rectangular grid topology)
        
        # Arguments
            y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        # Return
            pairwise distance matrix (map_dist[i,k] is the Manhattan distance on the map between assigned cell of data point i and cell k)
        """
        labels = np.arange(self.n_prototypes)
        tmp = np.expand_dims(y_pred, axis=1)
        d_row = np.abs(tmp-labels) // self.map_size[1]
        d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
        return d_row + d_col

    @staticmethod
    def neighborhood_function(d, T, neighborhood='gaussian'):
        """
        SOM neighborhood function (gaussian neighborhood)

        # Arguments
            x: distance on the map
            T: temperature parameter
        # Return
            neighborhood weight
        """
        if neighborhood == 'gaussian':
            return np.exp(-(d ** 2) / (T ** 2))
        elif neighborhood == 'window':
            return (d <= T).astype(np.float32)
    
    def pretrain(self, X,
                 optimizer='adam',
                 epochs=200,
                 batch_size=256,
                 save_dir='results/tmp'):
        """
        Pre-train the autoencoder using only MSE reconstruction loss
        Saves weights in h5 format.

        # Arguments
            X: training set
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
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
    
    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            iterations=10000,
            som_iterations=10000,
            eval_interval=10,
            save_epochs=5,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential',
            save_dir='results/tmp'):
        """
        Training procedure

        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           y_val: (optional) validation labels
           iterations: number of training iterations
           som_iterations: number of iterations where SOM neighborhood is decreased
           eval_interval: evaluate metrics on training/validation batch every eval_interval iterations
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           Tmax: initial temperature parameter
           Tmin: final temperature parameter
           decay: type of temperature decay ('exponential' or 'linear')
           save_dir: path to existing directory where weights and logs are saved
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        save_interval = X_train.shape[0] // batch_size * save_epochs # save every save_epochs epochs
        print('Save interval:', save_interval)

        # Logging file
        logfile = open(save_dir + '/desom_log.csv', 'w')
        fieldnames = ['iter', 'T', 'L', 'Lr', 'Lsom', 'Lkm', 'Ltop', 'quantization_err', 'topographic_err', 'latent_quantization_err', 'latent_topographic_err']
        if X_val is not None:
            fieldnames += ['L_val', 'Lr_val', 'Lsom_val', 'Lkm_val', 'Ltop_val', 'quantization_err_val', 'topographic_err_val', 'latent_quantization_err_val', 'latent_topographic_err_val']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        if y_val is not None:
            fieldnames += ['acc_val', 'pur_val', 'nmi_val', 'ari_val']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        # Set and compute some initial values
        index = 0
        if X_val is not None:
            index_val = 0

        for ite in range(iterations):
            # Get training and validation batches
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

            # Compute cluster assignments for batches
            _, d = self.model.predict(X_batch)
            y_pred = d.argmin(axis=1)
            if X_val is not None:
                _, d_val = self.model.predict(X_val_batch)
                y_val_pred = d_val.argmin(axis=1)

            # Update temperature parameter
            if ite < som_iterations:
                if decay == 'exponential':
                    T = Tmax*(Tmin/Tmax)**(ite/(som_iterations-1))
                elif decay == 'linear':
                    T = Tmax - (Tmax-Tmin)*(ite/(som_iterations-1))
            
            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred), T)
            if X_val is not None:
                w_val_batch = self.neighborhood_function(self.map_dist(y_val_pred), T)

            # Train on batch
            loss = self.model.train_on_batch(X_batch, [X_batch, w_batch])

            if ite % eval_interval == 0:
                # Initialize log dictionary
                logdict = dict(iter=ite, T=T)

                # Get SOM weights and decode to original space
                decoded_prototypes = self.decode(self.prototypes)

                # Evaluate losses and metrics
                print('iteration {} - T={}'.format(ite, T))
                logdict['L'] = loss[0]
                logdict['Lr'] = loss[1]
                logdict['Lsom'] = loss[2]
                logdict['Lkm'] = kmeans_loss(y_pred, d)
                logdict['Ltop'] = loss[2] - logdict['Lkm']
                logdict['latent_quantization_err'] = quantization_error(d)
                logdict['latent_topographic_err'] = topographic_error(d, self.map_size)
                d_original = np.square((np.expand_dims(X_batch, axis=1) - decoded_prototypes)).sum(axis=2)
                logdict['quantization_err'] = quantization_error(d_original)
                logdict['topographic_err'] = topographic_error(d_original, self.map_size)
                print('[Train] - Lr={:f}, Lsom={:f} (Lkm={:f}/Ltop={:f}) - total loss={:f}'.format(logdict['Lr'], logdict['Lsom'], logdict['Lkm'], logdict['Ltop'], logdict['L']))
                print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err'], logdict['topographic_err']))
                if X_val is not None:
                    val_loss = self.model.test_on_batch(X_val_batch, [X_val_batch, w_val_batch])
                    logdict['L_val'] = val_loss[0]
                    logdict['Lr_val'] = val_loss[1]
                    logdict['Lsom_val'] = val_loss[2]
                    logdict['Lkm_val'] = kmeans_loss(y_val_pred, d_val)
                    logdict['Ltop_val'] = val_loss[2] - logdict['Lkm_val']
                    logdict['latent_quantization_err_val'] = quantization_error(d_val)
                    logdict['latent_topographic_err_val'] = topographic_error(d_val, self.map_size)
                    d_original_val = np.square((np.expand_dims(X_batch, axis=1) - decoded_prototypes)).sum(axis=2)
                    logdict['quantization_err_val'] = quantization_error(d_original_val)
                    logdict['topographic_err_val'] = topographic_error(d_original_val, self.map_size)   
                    print('[Val] - Lr={:f}, Lsom={:f} (Lkm={:f}/Ltop={:f}) - total loss={:f}'.format(logdict['Lr_val'], logdict['Lsom_val'], logdict['Lkm_val'], logdict['Ltop_val'], logdict['L_val']))
                    print('[Val] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err_val'], logdict['topographic_err_val']))

                # Evaluate the clustering performance using labels
                if y_train is not None:
                    logdict['acc'] = cluster_acc(y_batch, y_pred)
                    logdict['pur'] = cluster_purity(y_batch, y_pred)
                    logdict['nmi'] = metrics.normalized_mutual_info_score(y_batch, y_pred)
                    logdict['ari'] = metrics.adjusted_rand_score(y_batch, y_pred)
                    print('[Train] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc'], logdict['pur'], logdict['nmi'], logdict['ari']))
                if y_val is not None:
                    logdict['acc_val'] = cluster_acc(y_val_batch, y_val_pred)
                    logdict['pur_val'] = cluster_purity(y_val_batch, y_val_pred)
                    logdict['nmi_val'] = metrics.normalized_mutual_info_score(y_val_batch, y_val_pred)
                    logdict['ari_val'] = metrics.adjusted_rand_score(y_val_batch, y_val_pred)
                    print('[Val] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc_val'], logdict['pur_val'], logdict['nmi_val'], logdict['ari_val']))
                    
                logwriter.writerow(logdict)

            # Save intermediate model
            if ite % save_interval == 0:
                 self.model.save_weights(save_dir + '/DESOM_model_' + str(ite) + '.h5')
                 print('Saved model to:', save_dir + '/DESOM_model_' + str(ite) + '.h5')

        # Save the final model
        logfile.close()
        print('saving model to:', save_dir + '/DESOM_model_final.h5')
        self.model.save_weights(save_dir + '/DESOM_model_final.h5')


if __name__ == "__main__":

    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'fmnist', 'usps', 'reuters10k'])
    parser.add_argument('--validation', default=False, type=bool, help='use train/validation split')
    parser.add_argument('--ae_weights', default=None, help='pre-trained autoencoder weights')
    parser.add_argument('--map_size', nargs='+', default=[8, 8], type=int)
    parser.add_argument('--gamma', default=0.001, type=float, help='coefficient of self-organizing map loss')
    parser.add_argument('--pretrain_epochs', default=0, type=int)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--som_iterations', default=10000, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--save_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--Tmax', default=10.0, type=float)
    parser.add_argument('--Tmin', default=0.1, type=float)
    parser.add_argument('--decay', default='exponential', choices=['exponential', 'linear'])
    parser.add_argument('--save_dir', default='results/tmp')
    args = parser.parse_args()
    print(args)

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data
    (X_train, y_train), (X_val, y_val) = load_data(args.dataset, validation=args.validation)

    # Set default values
    pretrain_optimizer = 'adam'

    # Instantiate model
    desom = DESOM(encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10], map_size=args.map_size)
    
    # Initialize model
    optimizer = 'adam'
    desom.initialize(ae_act='relu', ae_init='glorot_uniform')
    plot_model(desom.model, to_file='desom_model.png', show_shapes=True)
    desom.model.summary()
    desom.compile(gamma=args.gamma, optimizer=optimizer)

    # Load pre-trained AE weights or pre-train
    if args.ae_weights is None and args.pretrain_epochs > 0:
        desom.pretrain(X=X_train, y=y_train, optimizer=pretrain_optimizer,
                       epochs=args.pretrain_epochs, batch_size=args.batch_size,
                       save_dir=args.save_dir)
    elif args.ae_weights is not None:
        desom.load_ae_weights(args.ae_weights)

    # Fit model
    t0 = time()
    desom.fit(X_train, y_train, X_val, y_val, args.iterations, args.som_iterations, args.eval_interval,
              args.save_epochs, args.batch_size, args.Tmax, args.Tmin, args.decay, args.save_dir)
    print('Training time: ', (time() - t0))

    # Generate DESOM map visualization using reconstructed prototypes
    if args.dataset in ['mnist', 'fmnist', 'usps']:
        img_size = int(np.sqrt(X_train.shape[1]))
        decoded_prototypes = desom.decode(desom.prototypes)
        fig, ax = plt.subplots(args.map_size[0], args.map_size[1], figsize=(10, 10))
        for k in range(args.map_size[0]*args.map_size[1]):
            ax[k // args.map_size[1]][k % args.map_size[1]].imshow(decoded_prototypes[k].reshape(img_size, img_size), cmap='gray')
            ax[k // args.map_size[1]][k % args.map_size[1]].axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig('desom_map_{}.png'.format(args.dataset), bbox_inches='tight')
