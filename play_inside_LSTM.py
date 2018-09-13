from keras.engine.saving import load_model
from models.VAE import getVAEModel
from models.LSTM import getLSTMModel
import numpy as np
import matplotlib.pyplot as plt
import keyboard
import sys
from constants import *


class ButtonCodes:
	D_LEFT = 6
	D_RIGHT = 7
	D_UP = 4
	D_DOWN = 5

# lstm_model = getLSTMModel((1, LATENT_DIM + NB_ACTIONS))
lstm_model = load_model("saved_models/LSTM.h5")

vae_model = getVAEModel()
vae_model.load_weights('saved_models/VAE.h5')
decoder = vae_model.layers[-1]

# On commence à partir d'une latent_image du dataset
train = np.load('data/latent_images/GreenHillZone.Act1.LSTM_train.npy')
latent_image = [train[0]]

while True:
	# Action du joueur
	actions = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	# On détecte si une des touches directionnelles est enfoncée
	try:
		if keyboard.is_pressed('up'):
			actions[0][ButtonCodes.D_UP] = 1
			print('UP')
		if keyboard.is_pressed('left'):
			actions[0][ButtonCodes.D_LEFT] = 1
			print('LEFT')
		if keyboard.is_pressed('right'):
			actions[0][ButtonCodes.D_RIGHT] = 1
			print('RIGHT')
		if keyboard.is_pressed('down'):
			actions[0][ButtonCodes.D_DOWN] = 1
			print('RIGHT')
		if keyboard.is_pressed('escape'):
			print("exit")
			sys.exit(1)
	except:
		break

	# Il faut fournir la latent_image + le tableau d'actions au LSTM
	lstm_input = np.concatenate((latent_image, actions), axis=1)
	lstm_input = np.reshape(lstm_input, (1, 1, 140))
	# Il nous prédit la prochaine image
	latent_image = lstm_model.predict(lstm_input)
	# On affiche l'image à l'écran
	reconstructed_image = decoder.predict(latent_image)
	reconstructed_image = reconstructed_image.reshape(IMG_SHAPE)
	plt.clf()
	plt.imshow(reconstructed_image, vmin=0, vmax=1)
	plt.pause(0.0000001)