from keras.callbacks import ModelCheckpoint
from constants import *
from models.LSTM import getLSTMModel
import numpy as np


input_shape = (1, LATENT_DIM + NB_ACTIONS)
epochs = 100

# TODO : ce n'est pas la bonne façon de fournir les données. Il faut séparer les 2 jeux de données d'entraînement
# Et grouper les données de façon à rendre le réseau stateful sur train puis reset puis statful sur train2
training_actions = np.load('data/actions/GreenHillZone.Act1.LSTM_train.npy')
training_actions = np.concatenate((training_actions, np.load('data/actions/GreenHillZone.Act1.LSTM_train2.npy')))
training_latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_train.npy')
training_latent_images = np.concatenate((training_latent_images, np.load('data/latent_images/GreenHillZone.Act1.LSTM_train2.npy')))
X_train = np.append(training_latent_images[:-1], training_actions[:-1], axis=1)
Y_train = training_latent_images[1:]

validation_actions = np.load('data/actions/GreenHillZone.Act1.LSTM_test.npy')
validation_latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_test.npy')
# Il n'y aura pas de label pour la dernière valeur donc on la supprime
X_test = np.append(validation_latent_images[:-1], validation_actions[:-1], axis=1)
# les labels Y_test sont les mêmes données que X_test mais décalées de 1 car ils correspondent à l'image à prédire
# D'où le [1:]
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

# Création de checkpoints afin de ne sauvegarder que les meilleurs poids lors de l'entrainement
checkpoint = ModelCheckpoint('saved_models/LSTM.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=epochs, callbacks=callbacks_list, batch_size=128, verbose=2, shuffle=False)

# meilleure val_loss : 1.1000