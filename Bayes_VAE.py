import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.callbacks import TensorBoard

class BayesVAE(object):
    ''' Class to build a simple variational autoencoder with Keras.
     #Reference
     - Auto-Encoding Variational Bayes
       https://arxiv.org/abs/1312.6114
    '''

    def __init__(self,  original_dim, intermediate_dim, batch_size=32,  epsilon_std=1.0, latent_dim=2):
        self.input_img = Input(batch_shape=(batch_size, original_dim))
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epsilon_std = epsilon_std
        self.setup_dense_training_model()

    def setup_dense_training_model(self):

        x = self.input_img
        h = Dense(self.intermediate_dim, activation='relu')(x)
        self.z_mean = Dense(self.latent_dim)(h)
        # self.z_log_sigma = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)
        # sample latent variable z assuming normal distribution
        z = Lambda(self.normal_sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        self.decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        y = CustomVariationalLayer(self.original_dim, self.z_mean, self.z_log_var)([x, x_decoded_mean])
        self.vae = Model(x, y)

    def normal_sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(self.z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def encoder_model(self):
        # model mapping input to its encoded representation
        return Model(self.input_img, self.z_mean)

    def decoder_model(self):
        # decoder that samples from the learned distribution and decodes the mean back to input space
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        return Model(decoder_input, _x_decoded_mean)

    def train(self, x_train, x_test, nb_epochs=50):
        self.vae.compile(optimizer='rmsprop', loss=None)
        self.nb_epochs = nb_epochs
        self.vae.fit(x_train,
                shuffle=True,
                epochs=self.nb_epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, None),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, original_dim, z_mean, z_log_var, **kwargs):
        self.is_placeholder = True
        self.original_dim = original_dim
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


def scatterplot_latent_space(encoder, x_test, y_test, batch_size):
    """
    Display a 2D plot of the digit classes in the latent space learned with VAE
    :return: None
    """
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

def plot_manifold(generator):
    """
    Display a 2D scatterplot of the input manifold learned with VAE
    :param generator:
    :return:
    """
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

def load_mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test