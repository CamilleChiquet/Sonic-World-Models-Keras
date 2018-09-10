from keras import Sequential
from keras.layers import LSTM, Dense


def getLSTMModel(input_shape):
	model = Sequential()
	model.add(LSTM(units=128, input_shape=input_shape, activation='relu'))
	model.add(Dense(units=128, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model