from keras import Sequential
from keras.layers import LSTM, Dense


def getLSTMModel(input_shape):
	model = Sequential()
	# TODO : rendre le nombre d'unités dynamique
	model.add(LSTM(units=140, input_shape=input_shape))
	# model.add(LSTM(units=128, input_shape=input_shape, activation='relu'))
	model.add(Dense(units=140, activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
