from models.LSTM import getLSTMModel
import numpy as np


input_shape = (1, 140)
epochs = 10

training_actions = np.load('data/actions/GreenHillZone.Act1.LSTM_train.npy')
training_actions = np.concatenate((training_actions, np.load('data/actions/GreenHillZone.Act1.LSTM_train2.npy')))
training_latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_train.npy')
training_latent_images = np.concatenate((training_latent_images, np.load('data/latent_images/GreenHillZone.Act1.LSTM_train2.npy')))
X_train = np.append(training_latent_images[:-1], training_actions[:-1], axis=1)
Y_train = training_latent_images[1:]

validation_actions = np.load('data/actions/GreenHillZone.Act1.LSTM_test.npy')
validation_latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_test.npy')
X_test = np.append(validation_latent_images[:-1], validation_actions[:-1], axis=1)
Y_test = validation_latent_images[1:]

# Les LSTM s'attendent à recevoir des données au format : [samples, time steps, features]
# Or nous avons des données de la forme [samples, features]
# D'où la transformation qui suit
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(np.shape(X_train))
print(np.shape(Y_train))
print(np.shape(X_test))
print(np.shape(Y_test))

model = getLSTMModel(input_shape=input_shape)
model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=1, verbose=2, shuffle=False)
