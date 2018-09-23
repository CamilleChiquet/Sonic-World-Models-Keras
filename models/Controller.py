from keras import Sequential
from keras.layers import BatchNormalization, Dense
from keras.engine.saving import load_model
from constants import *
from keras import layers




class Controller():

	def __init__(self):
		self.input_shape = (LATENT_DIM)
		self._build()

	def _build(self):
		self.model = Sequential()
		self.model.add(layers.Dense(units=NB_ACTIONS, input_shape=self.input_shape, activation='sigmoid', kernel_initializer='random_normal'))
		self.model.add(BatchNormalization())
		self.model.compile(loss='mse', optimizer='adam')

	def train(self, X_train, Y_train, validation_data):
		self.model.fit(x=X_train, y=Y_train, validation_data=validation_data, epochs=1, batch_size=1, verbose=2)

	def save(self, path):
		self.model.save(path)

	def save_weights(self, file_path):
		self.model.save_weights(filepath=file_path)

	def load_weights(self, file_path):
		self.model.load_weights(filepath=file_path)
