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
		# Only one lstm layer
		# The output needs to be the same size as the LATENT_DIM because the LSTM predict the future latent vector
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
	We load the trained model of the LSTM in order to play inside it.
	Its output is connected to its input so it generates continuously new latent vectors.
	The actions of the player are also given to the input layer.
	'''
	def play_in_dream(self, start_image, decoder):
		latent_image = start_image
		while True:
			# Player's actions
			actions = [[0, 0, 0, 0]]
			# We check wich keys are pressed
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

			# latent vector + actions are given to the input layer of the LSTM
			lstm_input = np.concatenate((latent_image, actions), axis=1)
			lstm_input = np.reshape(lstm_input, (1, 1, LATENT_DIM + NB_ACTIONS))
			# Futur latent vector is predicted
			latent_image = self.model.predict(lstm_input)
			# We pass the latent vector through the decoder to see the corresponding image
			reconstructed_image = decoder.predict(latent_image)
			reconstructed_image = reconstructed_image.reshape(IMG_SHAPE)
			plt.clf()
			plt.imshow(reconstructed_image, vmin=0, vmax=1)
			# Necessary to display something on screen
			plt.pause(0.0000001)
