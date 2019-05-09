# DESOM: Deep Embedded Self-Organizing Map 

This is the official Keras implementation of the **Deep Embedded Self-Organizing Map (DESOM)** model.

DESOM is an unsupervised learning model that jointly learns representations and the code vectors of a self-organizing map (SOM) in order to survey, cluster and visualize large, high-dimensional datasets. Our model is composed of an autoencoder and a custom SOM layer that are optimized in a joint training procedure, motivated by the idea that the SOM prior could help learning SOM-friendly representations. Its training is fast, end-to-end and requires no pre-training.

<img src="./fig/desom_map_mnist.png" width=300 height=300/><img src="./fig/desom_map_fmnist.png" width=300 height=300/>

When using this code, please cite following work:

> Florent Forest, Mustapha Lebbah, Hanene Azzag and Jérôme Lacaille (2019). Deep Embedded SOM: Joint Representation Learning and Self-Organization. In European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2019).

> Florent Forest, Mustapha Lebbah, Hanene Azzag and Jérôme Lacaille (2019). Deep Architectures for Joint Clustering and Visualization with Self-Organizing Maps. In Workshop on Learning Data Representations for Clustering (LDRC), PAKDD 2019.

(see also http://florentfo.rest/publications)

## Quick start

The implementation is divided into 4 scripts:

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

One training run on MNIST with 10000 iterations and batch size 256 on a laptop GPU takes around 2 minutes.

A full benchmark of DESOM on the 4 datasets can be started by calling the script `desom_benchmark.py`. Parameters, number of runs and save directories are specified inside the script. Paper results were obtained using this script and number of runs equal to 10. Similar scripts were used for other compared models (minisom, kerasom and with pre-trained autoencoder weights).

The main dependencies are keras, tensorflow, scikit-learn, numpy, pandas, matplotlib.
