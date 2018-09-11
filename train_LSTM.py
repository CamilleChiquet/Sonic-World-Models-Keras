from models.LSTM import getLSTMModel
import numpy as np


input_shape = (1, 140)
epochs = 10

def create_shifted_data(dataset):
	dataY = []
	for i in range(len(dataset) - 2):
		dataY.append(dataset[i + 1])
	return np.array(dataY)

training_actions = np.load('data/actions/GreenHillZone.Act1.LSTM_train.npy')
training_actions = np.concatenate((training_actions, np.load('data/actions/GreenHillZone.Act1.LSTM_train2.npy')))
print(np.shape(training_actions))

training_latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_train.npy')
training_latent_images = np.concatenate((training_latent_images, np.load('data/latent_images/GreenHillZone.Act1.LSTM_train2.npy')))
print(np.shape(training_latent_images))

trainX = np.append(training_latent_images, training_actions, axis=1)
print(np.shape(trainX))
trainY = create_shifted_data(trainX)

validation_actions = np.load('data/actions/GreenHillZone.Act1.LSTM_test.npy')
validation_latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_test.npy')
testX = np.append(validation_latent_images, validation_actions, axis=1)
testY = create_shifted_data(testX)

# Les LSTM s'attendent à recevoir des données au format : [samples, time steps, features]
# Or nous avons des données de la forme [samples, features]
# D'où la transformation qui suit
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
testY = np.reshape(testY, (testY.shape[0], 1, testY.shape[1]))

model = getLSTMModel(input_shape=input_shape)
model.fit(x=trainX, y=trainY, epochs=epochs, batch_size=1, verbose=2)
