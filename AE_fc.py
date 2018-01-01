# simplest model using one FC layer for encoding and decoding

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np

def load_mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test

class DenseAE(object):
    def __init__(self, input_dims=784, encoding_dim=[32], add_sparsity_constraint=None):
        self.input_dims= input_dims
        # this is our input placeholder
        self.input_img = Input(shape=(input_dims,))
        # The representations is only constrained by the size of the hidden layer (32). In such a situation,
        # what typically happens is that the hidden layer is learning an approximation of PCA
        # (principal component analysis).
        self.encoding_dim = encoding_dim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
        self.add_sparsity_constraint=add_sparsity_constraint
        if not add_sparsity_constraint:
            self.setup_training_model()
        else:
            self.setup_sparse_training_model(add_sparsity_constraint)

    def setup_training_model(self):
        for index,fc_dim in enumerate(self.encoding_dim):
            if index == 0:
                self.encoded = Dense(fc_dim, activation='relu')(self.input_img)
                self.decoded = Dense(fc_dim, activation='sigmoid')(self.encoded)
            else:
                self.encoded = Dense(fc_dim, activation='relu')(self.encoded)
                self.decoded = Dense(fc_dim, activation='sigmoid')(self.decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(self.input_img, self.decoded)

    def setup_sparse_training_model(self, add_sparsity_constraint='L1'):
        if add_sparsity_constraint == 'L1':
            # add a Dense layer with a L1 activity regularizer
            self.encoded = Dense(self.encoding_dim, activation='relu',
                             activity_regularizer=regularizers.l1(10e-5))(self.input_img)
        # "decoded" is the lossy reconstruction of the input
        self.decoded = Dense(self.input_dims, activation='sigmoid')(self.encoded)
        # this model maps an input to its reconstruction
        self.autoencoder_sparse = Model(self.input_img, self.decoded)

    def encoder_model(self):
        # this model maps an input to its encoded representation
        return Model(self.input_img, self.encoded)

    def decoder_model(self):
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        return Model(encoded_input, decoder_layer(encoded_input))

    def train(self, x_train, x_val, nb_epochs=50):
        if self.add_sparsity_constraint:
            self.autoencoder = self.autoencoder_sparse
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.fit(x_train, x_train,
                        epochs=nb_epochs,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_val, x_val))

def main():
    x_tr, x_val = load_mnist_data()
    fc1AE = DenseAE(input_dims=784, dense_layers=1)
    fc1AE.train(x_tr, x_val)
    print("Training AE done")