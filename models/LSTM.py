from keras import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, CuDNNLSTM
from keras.optimizers import Adam, SGD

from constants import *


def getLSTMModel(input_shape):
	model = Sequential()
	model.add(CuDNNLSTM(units=LATENT_DIM, input_shape=input_shape))
	model.add(BatchNormalization())
	model.compile(loss='mse', optimizer='adam')
	return model
