from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, Conv2DTranspose, BatchNormalization as BN, \
	Dropout, MaxPooling2D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from constants import *


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def getVAEModel():

	inputs = Input(shape=IMG_SHAPE, name='encoder_input')
	# dimension de l'image en entrée : (224, 320, 3)
	x = Conv2D(filters=32, kernel_size=4, strides=2, kernel_initializer='normal', padding='same', activation='relu')(inputs)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	# (112, 160, 32)
	x = Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	# (56 80, 64)
	x = Conv2D(filters=128, kernel_size=4, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	# (28, 40, 128)
	x = Conv2D(filters=256, kernel_size=4, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	# (14, 20, 128)
	x = Conv2D(filters=512, kernel_size=4, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	# x = Dropout(0.5)(x)
	x = BN()(x)
	# (7, 10, 128)
	# On retient la shape à cet endroit pour le decoder
	shape = K.int_shape(x)

	# generate latent vector Q(z|X)
	x = Flatten()(x)
	x = Dense(1024, activation='relu')(x)
	# x = Dropout(0.5)(x)
	x = BN()(x)
	z_mean = Dense(LATENT_DIM, name='z_mean')(x)
	z_log_var = Dense(LATENT_DIM, name='z_log_var')(x)

	# use reparameterization trick to push the sampling out as input
	# note that "output_shape" isn't necessary with the TensorFlow backend
	z = Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

	# instantiate encoder model
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
	# encoder.summary()

	# build decoder model
	latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')
	x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	x = Reshape((shape[1], shape[2], shape[3]))(x)

	x = Conv2DTranspose(filters=256, kernel_size=5, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	# (14, 20, 128)
	x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	# x = Dropout(0.25)(x)
	x = BN()(x)
	# (28, 40, 64)
	x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	x = BN()(x)
	# (56, 80, 64)
	x = Conv2DTranspose(filters=32, kernel_size=5, strides=2, kernel_initializer='normal', padding='same', activation='relu')(x)
	x = BN()(x)
	# (112, 160, 64)
	outputs = Conv2DTranspose(filters=3, kernel_size=5, strides=2, kernel_initializer='normal', padding='same', activation='sigmoid')(x)
	# (224, 320, 3)
	# instantiate decoder model
	decoder = Model(latent_inputs, outputs, name='decoder')
	# decoder.summary()

	# instantiate VAE model
	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name='vae')

	# Une fonction de coût classique qui permet de déterminer l'erreur entre l'image reconstituée et celle attendue
	reconstruction_loss = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2] * binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

	# Fonction de coût personnalisée, utilisée pour les VAE
	kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

	# La fonction de coût du model est la combinaison de la binary_crossentropy et de k1_loss
	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)
	vae.compile(optimizer='adam')
	return vae