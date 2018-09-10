import numpy as np
from keras.callbacks import ModelCheckpoint
from models.VAE import *
from models.process import *
from process import *

input_shape = (224, 320, 3)
epochs = 100
batch_size = 32

# Chargement des images d'entrainement et validation
test = np.load('../data/images/GreenHillZone.Act1.test.npy')
# Normalisation des données (valeurs entre 0 et 1)
test = test/255
train = np.load('../data/images/GreenHillZone.Act1.train1.npy')
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train2.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train3.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train4.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train5.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train6.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train7.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train8.npy')))
train = np.concatenate((train, np.load('../data/images/GreenHillZone.Act1.train9.npy')))
train = train/255

# Chargement du model de réseau
model = getVAEModel(input_shape=input_shape)

# Chargement des poids sauvegardés du dernier entrainement
# model.load_weights('../saved_models/VAE.h5')

# Création de checkpoints afin de ne sauvegarder que les meilleurs poids lors de l'entrainement
checkpoint = ModelCheckpoint('../saved_models/VAE.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Entrainement du réseau
# Pas de labels pour le VAE
history = model.fit(train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks_list, validation_data=(test, None))

# plot_graphs(history)

# Sauvegarde manuelle des poids du réseau
# model.save_weights('../saved_models/VAE.h5')

# On teste sur une image prise arbitrairement
image_to_test = test[120]
image_base = image_from_array(image_to_test)
image_base.save('../img/image_base.png')

# Le réseau s'attend à recevoir un batch d'images
# on reshape donc l'image à donner pour lui passer un batch d'une seule image
reconstructed_img = model.predict(np.reshape(image_to_test, (1, 224, 320, 3)))
reconstructed_img = reconstructed_img.reshape(input_shape)
reconstructed_img = image_from_array(reconstructed_img)
reconstructed_img.save('../img/image_reconstructed.png')

