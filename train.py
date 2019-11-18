"""
Implementation of the Deep Embedded Self-Organizing Map model
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
from keras.utils.vis_utils import plot_model

# Dataset helper function
from datasets import load_data

# DESOM
from DESOM import DESOM
from ConvDESOM import ConvDESOM

if __name__ == "__main__":

    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'fmnist', 'usps', 'reuters10k'])
    parser.add_argument('--architecture', default='dense', choices=['dense', 'convolutional'])
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
    parser.add_argument('--neighborhood', default='gaussian', choices=['gaussian', 'window'])
    parser.add_argument('--save_dir', default='results/tmp')
    args = parser.parse_args()
    print(args)

    # Convolutional architecture is only designed for MNIST/FMNIST
    if args.architecture == 'convolutional' and args.dataset not in ['mnist', 'fmnist']:
        print('Convolutional architecture is only available for mnist and fmnist datasets!')
        exit(0)

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data
    flatten = (args.architecture == 'dense')
    (X_train, y_train), (X_val, y_val) = load_data(args.dataset, validation=args.validation, flatten=flatten)
    print('Training set: ', X_train.shape)
    if args.validation:
        print('Validation set: ', X_val.shape)

    # Set default values
    pretrain_optimizer = 'adam'
    optimizer = 'adam'

    # Instantiate model
    if args.architecture == 'dense':
        desom = DESOM(encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10], map_size=args.map_size)
    else:
        desom = ConvDESOM(input_shape=X_train.shape[1:],
                          latent_dim=10,
                          encoder_filters=[32, 64, 64],  # [32, 64, 128, 256],
                          filter_size=3,
                          pooling_size=2,
                          map_size=args.map_size)

    # Initialize model
    desom.initialize(ae_act='relu', ae_init='glorot_uniform')
    plot_model(desom.model, to_file=os.path.join(args.save_dir, 'desom_model.png'), show_shapes=True)
    desom.model.summary()
    desom.compile(gamma=args.gamma, optimizer=optimizer)

    # Load pre-trained AE weights or pre-train
    if args.ae_weights is None and args.pretrain_epochs > 0:
        desom.pretrain(X=X_train, optimizer=pretrain_optimizer, epochs=args.pretrain_epochs, batch_size=args.batch_size,
                       save_dir=args.save_dir)
    elif args.ae_weights is not None:
        desom.load_ae_weights(args.ae_weights)

    # Fit model
    t0 = time()
    desom.fit(X_train, y_train, X_val, y_val, args.iterations, args.som_iterations, args.eval_interval,
              args.save_epochs, args.batch_size, args.Tmax, args.Tmin, args.decay, args.neighborhood, args.save_dir)
    print('Training time: ', (time() - t0))

    # Generate DESOM map visualization using reconstructed prototypes
    if args.dataset in ['mnist', 'fmnist', 'usps']:
        img_size = int(np.sqrt(X_train.shape[1])) if args.architecture == 'dense' else X_train.shape[1]
        decoded_prototypes = desom.decode(desom.prototypes)
        fig, ax = plt.subplots(args.map_size[0], args.map_size[1], figsize=(10, 10))
        for k in range(args.map_size[0]):
            for l in range(args.map_size[1]):
                ax[k][l].imshow(decoded_prototypes[k * args.map_size[1] + l].reshape(img_size, img_size), cmap='gray')
                ax[k][l].axis('off')
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(os.path.join(args.save_dir, 'desom_map_{}.png'.format(args.dataset)), bbox_inches='tight')
