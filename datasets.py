"""
Dataset loading functions

@author Florent Forest
@version 1.0
"""

import numpy as np

def load_mnist(flatten=True, validation=False):
    # Dataset, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Divide by 255.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if flatten: # flatten to 784-dimensional vector
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    if validation: # Return train and test set
        return (x_train, y_train), (x_test, y_test)
    else: # Return only train set with all images
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        return (x, y), (None, None)

def load_fashion_mnist(flatten=True, validation=False):
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Divide by 255.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if flatten: # flatten to 784-dimensional vector
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    if validation: # Return train and test set
        return (x_train, y_train), (x_test, y_test)
    else: # Return only train set with all images
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        return (x, y), (None, None)

def load_usps(data_path='./data/usps'):
    import h5py
    with h5py.File(data_path+'/usps.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
    x = np.concatenate((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))
    print('USPS samples', x.shape)
    return (x, y), (None, None)

def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return (x, y), (None, None)

def make_reuters_data(data_dir):
    """
    NOTE: RCV1-V2 data is heavy and not included.
    The data can be downloaded from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
    Necessary files are:
        'rcv1-v2.topics.qrels'
        'lyrl2004_tokens_test_pt0.dat'
        'lyrl2004_tokens_test_pt1.dat',
        'lyrl2004_tokens_test_pt2.dat',
        'lyrl2004_tokens_test_pt3.dat',
        'lyrl2004_tokens_train.dat'
    """
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000].astype(np.float32)
    print(x.dtype, x.size)
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    print('todense succeed')

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]
    print('permutation finished')

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})

def load_data(dataset_name, validation=False):
    if dataset_name == 'mnist':
        return load_mnist(flatten=True, validation=validation)
    elif dataset_name == 'fmnist':
        return load_fashion_mnist(flatten=True, validation=validation)
    elif dataset_name == 'usps':
        if validation:
            print('Train/validation split is not available for this dataset.')
        return load_usps()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        if validation:
            print('Train/validation split is not available for this dataset.')
        return load_reuters()
    else:
        print('Dataset {} not available! Available datasets are mnist, fmnist, usps and reuters10k.'.format(dataset_name))
        exit(0)
