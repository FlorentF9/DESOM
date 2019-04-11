# DESOM: Deep Embedded Self-Organizing Map 

This is a Keras implementation of the Deep Embedded Self-Organizing Map (DESOM) model. It is divided into 4 scripts :

* *DESOM.py*: main script containing the model (DESOM class)
* *SOM.py*: script containing the SOM layer
* *AE.py*: script for creating the autoencoder model
* *datasets.py*: script for loading the datasets benchmarked in the paper (MNIST, Fashion-MNIST, USPS and REUTERS-10k)
* *metrics.py*: script containing functions to compute metrics evaluated in the paper (purity and unsupervised clustering accuracy). NMI is already available in scikit-learn.
* *desom_benchmark.py*: script to perform benchmark runs of DESOM on 4 datasets and save results in a CSV file

The *data* directory contains USPS and REUTERS-10k datasets.

The main script has several command-line arguments that are explained with:

```shell
$ python3 DESOM.py --help
```

All arguments have default values, so DESOM training can be simply started doing:

```shell
$ python3 DESOM.py
```

For example, to train DESOM on Fashion-MNIST with a 20x20 map, the command is:

```shell
$ python3 DESOM.py --dataset fmnist --map_size 20 20
```

Training generates several outputs:

* an image of the DESOM/kerasom map to visualize the prototypes
* an image of the model architecture
* folder containing a log of training metrics and the model weights

One training run on MNIST with 10000 iterations and batch size 256 on a laptop GPU takes a couple of minutes.

A full benchmark of DESOM on the 4 datasets can be started by calling the script `desom_benchmark.py`. Parameters, number of runs and save directories are specified inside the script. Paper results were obtained using this script and number of runs equal to 10. Similar scripts were used for other compared models (minisom, kerasom and with pre-trained autoencoder weights).

The main dependencies are keras, tensorflow, scikit-learn, numpy, pandas, matplotlib.
