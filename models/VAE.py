from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, Conv2DTranspose, BatchNormalization as BN, \
	Dropout, MaxPooling2D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from constants import *
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM), mean=0.0, stddev=1.0)
	return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE():
	def __init__(self):
		self.models = self._build()
		self.vae = self.models[0]
		self.encoder = self.models[1]
		self.decoder = self.models[2]

	def _build(self):
		inputs = Input(shape=IMG_SHAPE, name='encoder_input')
		# dimension de l'image en entrée : (224, 320, 3)
		x = Conv2D(filters=32, kernel_size=4, strides=2, kernel_initializer='normal', padding='same',
				   activation='relu')(inputs)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		# (112, 160, 32)
		x = Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer='normal', padding='same',
				   activation='relu')(x)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		# (56 80, 64)
		x = Conv2D(filters=128, kernel_size=4, strides=2, kernel_initializer='normal', padding='same',
				   activation='relu')(x)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		# (28, 40, 128)
		x = Conv2D(filters=256, kernel_size=4, strides=2, kernel_initializer='normal', padding='same',
				   activation='relu')(x)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		# (14, 20, 128)
		x = Conv2D(filters=512, kernel_size=4, strides=2, kernel_initializer='normal', padding='same',
				   activation='relu')(x)
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
		encoder = Model(inputs, z, name='encoder')
		# encoder.summary()

		# build decoder model
		latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')
		x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		x = Reshape((shape[1], shape[2], shape[3]))(x)

		x = Conv2DTranspose(filters=256, kernel_size=5, strides=2, kernel_initializer='normal', padding='same',
							activation='relu')(x)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		# (14, 20, 128)
		x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, kernel_initializer='normal', padding='same',
							activation='relu')(x)
		# x = Dropout(0.25)(x)
		x = BN()(x)
		# (28, 40, 64)
		x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, kernel_initializer='normal', padding='same',
							activation='relu')(x)
		x = BN()(x)
		# (56, 80, 64)
		x = Conv2DTranspose(filters=32, kernel_size=5, strides=2, kernel_initializer='normal', padding='same',
							activation='relu')(x)
		x = BN()(x)
		# (112, 160, 64)
		outputs = Conv2DTranspose(filters=3, kernel_size=5, strides=2, kernel_initializer='normal', padding='same',
								  activation='sigmoid')(x)
		# (224, 320, 3)
		# instantiate decoder model
		decoder = Model(latent_inputs, outputs, name='decoder')
		# decoder.summary()

		# instantiate VAE model
		outputs = decoder(encoder(inputs))
		vae = Model(inputs, outputs, name='vae')

		# Une fonction de coût classique qui permet de déterminer l'erreur entre l'image reconstituée et celle attendue
		reconstruction_loss = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2] * binary_crossentropy(K.flatten(inputs),
																							   K.flatten(outputs))

		# Fonction de coût personnalisée, utilisée pour les VAE
		kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

		# La fonction de coût du model est la combinaison de la binary_crossentropy et de k1_loss
		vae_loss = K.mean(reconstruction_loss + kl_loss)
		vae.add_loss(vae_loss)
		vae.compile(optimizer='adam')

		return (vae, encoder, decoder)

	def save_weights(self, file_path):
		self.vae.save_weights(filepath=file_path)

	def load_weights(self, file_path):
		self.vae.load_weights(filepath=file_path)

	def train(self, epochs=100, batch_size=32, validation_split=0.2):
		training_data = None
		# On parcourt tous les fichiers numpy créés précédemment
		for data_file in glob.glob(os.path.join(IMG_DIR, '*' + VAE_TRAINING_EXT + '*.npy')):
			print(data_file)
			if training_data is None:
				training_data = np.load(data_file)
			else:
				training_data = np.concatenate((training_data, np.load(data_file)))
		training_data = training_data / 255

		earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=2)
		checkpoint = ModelCheckpoint('./saved_models/VAE.h5', monitor='val_loss', verbose=2, save_best_only=True,
									 mode='min')
		callbacks_list = [checkpoint, earlyStop]

		self.vae.fit(training_data, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks_list,
					 validation_split=validation_split, shuffle=True)

	def generate_latent_images(self):
		"""
		TODO : ne pas concaténer les tableaux d'images, il faut les différentier
		"""

		# ================ training data =====================
		training_latent_images = None
		for data_file in glob.glob(os.path.join(IMG_DIR, '*' + RNN_TRAINING_EXT + '*.npy')):
			print(data_file)
			if training_latent_images is None:
				training_latent_images = np.load(data_file)
			else:
				training_latent_images = np.concatenate((training_latent_images, np.load(data_file)))
		training_latent_images = training_latent_images / 255
		training_latent_images = self.encoder.predict(training_latent_images)

		training_actions = None
		for data_file in glob.glob(os.path.join(ACTIONS_DIR, '*' + RNN_TRAINING_EXT + '*.npy')):
			print(data_file)
			if training_actions is None:
				training_actions = np.load(data_file)
			else:
				training_actions = np.concatenate((training_actions, np.load(data_file)))
		X_train = np.append(training_latent_images[:-1], training_actions[:-1], axis=1)
		Y_train = training_latent_images[1:]
		del training_latent_images, training_actions

		# ================ validation data =====================

		test_latent_images = None
		for data_file in glob.glob(os.path.join(IMG_DIR, '*' + RNN_TEST_EXT + '*.npy')):
			print(data_file)
			if test_latent_images is None:
				test_latent_images = np.load(data_file)
			else:
				test_latent_images = np.concatenate((test_latent_images, np.load(data_file)))
		test_latent_images = test_latent_images / 255
		test_latent_images = self.encoder.predict(test_latent_images)

		validation_actions = None
		for data_file in glob.glob(os.path.join(ACTIONS_DIR, '*' + RNN_TEST_EXT + '*.npy')):
			print(data_file)
			if validation_actions is None:
				validation_actions = np.load(data_file)
			else:
				validation_actions = np.concatenate((validation_actions, np.load(data_file)))
		X_test = np.append(test_latent_images[:-1], validation_actions[:-1], axis=1)
		Y_test = test_latent_images[1:]
		del test_latent_images, validation_actions

		return X_train, Y_train, X_test, Y_test

	def generate_render(self, data_path, save_path=None):
		'''
		Affiche le rendu visuel du passage d'images à travers le VAE

		:param data_path: chemin des images à faire passer par le VAE
		:param vae : le modèle (entraîné) du VAE
		:param save_path: Si définit on sauvegarde les images sorties du VAE
		:return:
		'''
		images = np.load(data_path)
		images = images / 255

		generated_images = []
		for image in images:
			img = np.reshape(image, (1, 224, 320, 3))
			img = self.vae.predict(img)
			# img = model.predict(np.random.rand(1,LATENT_DIM))
			img = img.reshape(IMG_SHAPE)
			if (save_path != None):
				generated_images.append(img)
			plt.clf()
			plt.imshow(img, vmin=0, vmax=1)
			plt.pause(0.0000001)
		if (save_path != None):
			np.save(save_path, generated_images)

	def compare_data(self, original_data_path, reconstructed_data_path):
		'''
		Compare cote à cote 2 jeux d'images
		L'intérêt est de comparer les images originales d'une session de jeu en parrallèle de celles passées par le VAE

		:param original_data_path:
		:param reconstructed_data_path:
		:return:
		'''
		origin_images = np.load(original_data_path)
		reconstructed_images = np.load(reconstructed_data_path)
		fig = plt.figure(figsize=(1, 2))
		for i in range(len(origin_images)):
			plt.clf()
			origin_img = origin_images[i]
			rec_img = reconstructed_images[i]
			fig.add_subplot(1, 2, 1)
			plt.imshow(origin_img)
			fig.add_subplot(1, 2, 2)
			plt.imshow(rec_img)
			plt.pause(0.0000001)
