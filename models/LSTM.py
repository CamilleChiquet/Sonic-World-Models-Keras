from keras import Sequential, metrics
from keras.layers import BatchNormalization, CuDNNLSTM, regularizers, Dense, Dropout
from keras.engine.saving import load_model
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import sys
from constants import *
from keras import layers
import keras.backend as K



class LSTM():

	def __init__(self, input_shape=(None, LATENT_DIM + NB_ACTIONS)):
		self.input_shape = input_shape
		self._build()

	def _build(self):
		self.model = Sequential()
		self.model.add(layers.LSTM(units=LATENT_DIM, input_shape =self.input_shape, activation='sigmoid', kernel_initializer='random_normal'))
		self.model.add(BatchNormalization())
		self.model.compile(loss='mse', optimizer='adam')

	def train(self, X_train, Y_train, X_test, Y_test, epochs=200):

		print(X_train.shape)
		print(Y_train.shape)
		print(X_test.shape)
		print(Y_test.shape)
		self.model.fit(x=X_train, y=Y_train, epochs=epochs, validation_data=(X_test, Y_test), batch_size=SEQ_LENGTH, verbose=2, shuffle=False)

	def save(self, path):
		self.model.save(path)

	def save_weights(self, file_path):
		self.model.save_weights(filepath=file_path)

	def load_weights(self, file_path):
		self.model.load_weights(filepath=file_path)

	'''
	On charge le LSTM entraîné afin de jouer dans l'environnement qu'il génère.
	On lui fournit une image (latente : 128 dimensions) issue du dataset d'entraînement en entrée et il nous prédit la prochaine.
	On lui fournit ensuite en entrée l'image (latente) prédite afin qu'il nous prédise la suivante, ainsi de suite...
	En plus de fournir l'image (latente) on donne également les actions de l'utilisateur (touches directionnelles)
	'''
	def play_in_dream(self, start_image, decoder):
		latent_image = start_image
		while True:
			# Actions du joueur
			actions = [[0, 0, 0, 0]]
			# On détecte si une des touches directionnelles est enfoncée
			try:
				if keyboard.is_pressed('up'):
					actions[0][Actions.JUMP] = 1
					print('UP')
				if keyboard.is_pressed('left'):
					actions[0][Actions.LEFT] = 1
					print('LEFT')
				if keyboard.is_pressed('right'):
					actions[0][Actions.RIGHT] = 1
					print('RIGHT')
				if keyboard.is_pressed('down'):
					actions[0][Actions.DOWN] = 1
					print('DOWN')
				if keyboard.is_pressed('escape'):
					print("exit")
					sys.exit(1)
			except Exception as e:
				print(e)
				break

			# Il faut fournir la latent_image + le tableau d'actions au LSTM
			lstm_input = np.concatenate((latent_image, actions), axis=1)
			# TODO rendre le reshape dynamic
			lstm_input = np.reshape(lstm_input, (1, 1, LATENT_DIM + NB_ACTIONS))
			# Il nous prédit la prochaine image
			latent_image = self.model.predict(lstm_input)
			# On affiche l'image à l'écran
			reconstructed_image = decoder.predict(latent_image)
			reconstructed_image = reconstructed_image.reshape(IMG_SHAPE)
			plt.clf()
			plt.imshow(reconstructed_image, vmin=0, vmax=1)
			# Nécessaire sinon rien ne s'affiche
			plt.pause(0.0000001)
