"""
@author XXX
DESOM benchmarking script
"""

import time
import subprocess
import pandas as pd
import numpy as np
from metrics import cluster_acc, cluster_purity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datasets import load_data

from sklearn.cluster import KMeans
from DESOM import DESOM

datasets = ['MNIST (K=64)', 'Fashion-MNIST (K=64)', 'USPS (K=64)', 'REUTERS-10k (K=64)',
            'MNIST (K=10)', 'Fashion-MNIST (K=10)', 'USPS (K=10)', 'REUTERS-10k (K=4)']
results = pd.DataFrame(columns=['pur', 'pur_std', 'nmi', 'nmi_std', 'ari', 'ari_std', 'acc', 'acc_std','duration',
                                'pur_clust', 'pur_clust_std', 'nmi_clust', 'nmi_clust_std', 'ari_clust', 'ari_clust_std', 'acc_clust', 'acc_clust_std'],
                                 index=datasets)

n_runs = 1
output_file = 'desom_benchmark.csv'

pur = np.zeros(n_runs)
nmi = np.zeros(n_runs)
ari = np.zeros(n_runs)
acc = np.zeros(n_runs)
pur_clust = np.zeros(n_runs)
nmi_clust = np.zeros(n_runs)
ari_clust = np.zeros(n_runs)
acc_clust = np.zeros(n_runs)
duration = np.zeros(n_runs)

# SOM hyperparameters
Tmax = 10
Tmin = 0.1
decay = 'exponential'
# DESOM training hyperparameters
gamma = 0.001
optimizer = 'adam'
batch_size = 256
iterations = 100
som_iterations = iterations
eval_interval = 100
save_epochs = 1000000

def bench_desom(X_train, y_train, dataset, map_size, encoder_dims, ae_weights=None):
    print('*** {} - desom with {} map and {} autoencoder (gamma={})***'.format(dataset, map_size, encoder_dims, gamma))
                         
    desom = DESOM(encoder_dims=encoder_dims, map_size=map_size)
    save_dir = 'results/benchmark/desom-gamma{}_{}_{}_{}x{}'.format(gamma, dataset, optimizer, map_size[0], map_size[1])
    subprocess.run(['mkdir', '-p', save_dir])

    for run in range(n_runs):
        desom.initialize()
        desom.compile(gamma=gamma, optimizer=optimizer)
        if ae_weights is not None:
            desom.load_ae_weights(ae_weights)
        # Weights initialization by randomly sampling training points
        desom.init_som_weights(X_train)
        t0 = time.time()
        desom.fit(X_train, y_train, None, None, iterations, som_iterations, eval_interval, save_epochs, batch_size, Tmax, Tmin, decay, save_dir)
        dt = time.time()-t0
        print('Run {}/{} (took {:f} seconds)'.format(run+1, n_runs, dt))
        y_pred = desom.predict(X_train)
        pur[run] = cluster_purity(y_train, y_pred)
        nmi[run] = normalized_mutual_info_score(y_train, y_pred)
        ari[run] = adjusted_rand_score(y_train, y_pred)
        acc[run] = cluster_acc(y_train, y_pred)
        duration[run] = dt
        if map_size[0] == 8:
            # Post clustering in latent space
            print('Post-clustering in latent space...')
            prototypes = desom.prototypes
            km_desom = KMeans(n_clusters=np.max(y_train), n_jobs=-1).fit(prototypes)
            km_desom_pred = km_desom.predict(desom.encode(X_train))
            pur_clust[run] = cluster_purity(y_train, km_desom_pred)
            nmi_clust[run] = normalized_mutual_info_score(y_train, km_desom_pred)
            ari_clust[run] = adjusted_rand_score(y_train, km_desom_pred)
            acc_clust[run] = cluster_acc(y_train, km_desom_pred)

    name = '{} (K={})'.format(dataset, map_size[0]*map_size[1])
    results.at[name, 'pur'] = pur.mean()
    results.at[name, 'pur_std'] = pur.std()
    results.at[name, 'nmi'] = nmi.mean()
    results.at[name, 'nmi_std'] = nmi.std()
    results.at[name, 'ari'] = ari.mean()
    results.at[name, 'ari_std'] = ari.std()
    results.at[name, 'acc'] = acc.mean()
    results.at[name, 'acc_std'] = acc.std()
    results.at[name, 'duration'] = duration.mean()
    if map_size[0] == 8:
        # Post clustering
        results.at[name, 'pur_clust'] = pur_clust.mean()
        results.at[name, 'pur_clust_std'] = pur_clust.std()
        results.at[name, 'nmi_clust'] = nmi_clust.mean()
        results.at[name, 'nmi_clust_std'] = nmi_clust.std()
        results.at[name, 'ari_clust'] = ari_clust.mean()
        results.at[name, 'ari_clust_std'] = ari_clust.std()
        results.at[name, 'acc_clust'] = acc_clust.mean()
        results.at[name, 'acc_clust_std'] = acc_clust.std()

    print(results.loc[name])

# """
# MNIST
# """
# # Load data
# (X_train, y_train), _ = load_data('mnist')
# print("Loaded MNIST, shape:", X_train.shape)
# # DESOM with 64 clusters
# bench_desom(X_train, y_train, 'MNIST', map_size=(8,8), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_mnist_epoch200.h5')
# # # DESOM with 10 clusters
# bench_desom(X_train, y_train, 'MNIST', map_size=(10,1), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_mnist_epoch200.h5')

# """
# Fashion-MNIST
# """
# # Load data
# (X_train, y_train), _ = load_data('fmnist')
# print("Loaded Fashion-MNIST, shape:", X_train.shape)
# # DESOM with 64 clusters
# bench_desom(X_train, y_train, 'Fashion-MNIST', map_size=(8,8), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_fashion-mnist_epoch200.h5')
# # DESOM with 10 clusters
# bench_desom(X_train, y_train, 'Fashion-MNIST', map_size=(10,1), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_fashion-mnist_epoch200.h5')

# """
# USPS
# """
# # Load data
# (X_train, y_train), _ = load_data('usps')
# print("Loaded USPS, shape:", X_train.shape)
# # DESOM with 64 clusters
# bench_desom(X_train, y_train, 'USPS', map_size=(8,8), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_usps_epoch200.h5')
# # DESOM with 10 clusters
# bench_desom(X_train, y_train, 'USPS', map_size=(10,1), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_usps_epoch200.h5')

"""
REUTERS-10k
"""
# Load data
(X_train, y_train), _ = load_data('reuters10k')
print("Loaded REUTERS-10k, shape:", X_train.shape)
# DESOM with 64 clusters
bench_desom(X_train, y_train, 'REUTERS-10k', map_size=(8,8), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_reuters10k_epoch200.h5')
# DESOM with 4 clusters
bench_desom(X_train, y_train, 'REUTERS-10k', map_size=(4,1), encoder_dims=[X_train.shape[-1], 500, 500, 2000, 10])#, ae_weights='../results_paper/ae_weights_reuters10k_epoch200.h5')

results.to_csv(output_file, index_label='dataset')
