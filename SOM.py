"""
Implementation of the Deep Embedded Self-Organizing Map model
SOM layer

@author Florent Forest
@version 2.0
"""

import tensorflow as tf
from keras.engine.topology import Layer, InputSpec


class SOMLayer(Layer):
    """
    Self-Organizing Map layer class with rectangular topology

    # Example
    ```
        model.add(SOMLayer(map_size=(10,10)))
    ```
    # Arguments
        map_size: Tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1].
        prototypes: Numpy array with shape `(n_prototypes, latent_dim)` witch represents the initial cluster centers
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`
    # Output shape
        2D tensor with shape: `(n_samples, n_prototypes)`
    """

    def __init__(self, map_size, prototypes=None, **kwargs):
        if 'input_shape' not in kwargs and 'latent_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('latent_dim'),)
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.initial_prototypes = prototypes
        self.input_spec = InputSpec(ndim=2)
        self.prototypes = None
        self.built = False

    def build(self, input_shape):
        assert(len(input_shape) == 2)
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_dim))
        self.prototypes = self.add_weight(shape=(self.n_prototypes, input_dim), initializer='glorot_uniform', name='prototypes')
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Calculate pairwise squared euclidean distances between inputs and prototype vectors

        Arguments:
            inputs: the variable containing data, Tensor with shape `(n_samples, latent_dim)`
        Return:
            d: distances between inputs and prototypes, Tensor with shape `(n_samples, n_prototypes)`
        """
        # Note: (tf.expand_dims(inputs, axis=1) - self.prototypes) has shape (n_samples, n_prototypes, latent_dim)
        d = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2)
        return d

    def compute_output_shape(self, input_shape):
        assert(input_shape and len(input_shape) == 2)
        return input_shape[0], self.n_prototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super(SOMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
