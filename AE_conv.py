from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

class ConvAE(object):
    def __init__(self):
        self.input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
        self.setup_training_model()

    def setup_training_model(self):

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(self.input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        self.encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(self.encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(self.input_img, self.decoded)

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
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.fit(x_train, x_train,
                        epochs=nb_epochs,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_val, x_val),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

def load_mnist_data_as_2d_images():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    print(x_train.shape)
    print(x_test.shape)
    return x_train, x_test