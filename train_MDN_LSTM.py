from keras.callbacks import ModelCheckpoint
from constants import *
from models.LSTM import getLSTMModel
import numpy as np
from models.MDN_LSTM import MDN_LSTM



# Et grouper les données de façon à rendre le réseau stateful sur train puis reset puis statful sur train2
training_actions = np.load('./data/actions/GreenHillZone.Act1.LSTM_train.npy')
training_latent_images = np.load('./data/latent_images/GreenHillZone.Act1.LSTM_train.npy')
# Il n'y aura pas de label pour la dernière valeur donc on la supprime
X_train = np.append(training_latent_images[:-1], training_actions[:-1], axis=1)
# les labels Y_test sont les mêmes données que X_test mais décalées de 1 car ils correspondent à l'image à prédire
# D'où le [1:]
Y_train = training_latent_images[1:]


validation_actions = np.load('./data/actions/GreenHillZone.Act1.LSTM_test.npy')
validation_latent_images = np.load('./data/latent_images/GreenHillZone.Act1.LSTM_test.npy')
X_test = np.append(validation_latent_images[:-1], validation_actions[:-1], axis=1)
Y_test = validation_latent_images[1:]

# Les LSTM s'attendent à recevoir des données au format : [samples, time steps, features]
# Or nous avons des données de la forme [samples, features]
# D'où la transformation qui suit
X_train = np.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
Y_train = np.reshape(Y_train, (1, Y_train.shape[0], Y_train.shape[1]))
X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
Y_test = np.reshape(Y_test, (1, Y_test.shape[0], Y_test.shape[1]))
print(np.shape(X_train))
print(np.shape(Y_train))
print(np.shape(X_test))
print(np.shape(Y_test))

rnn = MDN_LSTM()
rnn.train(X_train, Y_train, (X_test, Y_test))