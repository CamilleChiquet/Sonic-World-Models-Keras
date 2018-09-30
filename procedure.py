'''
Folders of the projet :

|__ data
	|__ actions
	|__ images
	|__ latent_images
	|__ reconstructed_images
|__ models
|__ saved_models

'''
from keras.engine.saving import load_model
from data_generation import generate_data
from models.LSTM import LSTM
import numpy as np
from constants import *
from models.MDN_LSTM import MDN_LSTM
from models.VAE import VAE
from play import play
from process import create_project_folders
import neat_sonic


create_project_folders()

'''
	===============================================
	0 - Just play to Sonic :D
	===============================================
'''

# print('Try to finish a level by yourself !')
# play(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

'''
	===============================================
	1 - Generating datas for VAE
	===============================================
'''

# print('\nGenerating datas for VAE.')
# Make ~5 records (every one followed by a level reset)
# for level in LEVELS:
# 	generate_data(game='SonicTheHedgehog-Genesis', state=level, extension_name=VAE_TRAINING_EXT, frame_jump=4, save_actions=False)

'''
	===============================================
	2 - VAE training
	===============================================
'''

# vae = VAE()
# /!\ If the memory of your graphic card is too low, you can choose a smaller batch_size
# print('\nVAE training.')
# vae.train(batch_size=32, epochs=200)
# vae.save_weights(SAVED_MODELS_DIR + '/VAE_GreenHillZone.h5')
# vae.load_weights(file_path=SAVED_MODELS_DIR + '/VAE_GreenHillZone.h5')

'''
	===============================================
	3 - Visualization of the VAE
	===============================================
'''

# print('VAE visualization')
# images_array_path = IMG_DIR + '/GreenHillZone.Act2.vae_train1.npy'
# vae.generate_render(data_path=images_array_path)

'''
	===============================================
	4 - Generation of the LSTM's training dataset
	This training makes the LSTM learns the logic and physic of the game in order to predict the next (latent) frame.
	Make sure to record a lot of different situations (Sonic stuck in front of a wall, jump and move in the air...)
	===============================================
'''

# print('\nGenerating data for LSTM.')

# for level in LEVELS:
# 	print("\tDonnées d'entraînement")
# 	generate_data(state=level, extension_name=RNN_TRAINING_EXT, frame_jump=FRAME_JUMP, fixed_record_size=True)
# 	print("\tDonnées de validation")
# 	generate_data(state=level, extension_name=RNN_TEST_EXT, frame_jump=FRAME_JUMP, fixed_record_size=True)

# X_train_rnn, Y_train_rnn, X_test_rnn, Y_test_rnn = vae.generate_latent_images()

# np.save(DATA_DIR + '/X_train_rnn_GreenHillZone', X_train_rnn)
# np.save(DATA_DIR + '/X_test_rnn_GreenHillZone', X_test_rnn)
# np.save(DATA_DIR + '/Y_train_rnn_GreenHillZone', Y_train_rnn)
# np.save(DATA_DIR + '/Y_test_rnn_GreenHillZone', Y_test_rnn)

# X_train_rnn = np.load(DATA_DIR + '/X_train_rnn_GreenHillZone.npy')
# X_test_rnn = np.load(DATA_DIR + '/X_test_rnn_GreenHillZone.npy')
# Y_train_rnn = np.load(DATA_DIR + '/Y_train_rnn_GreenHillZone.npy')
# Y_test_rnn = np.load(DATA_DIR + '/Y_test_rnn_GreenHillZone.npy')

# nb_training_sequences = int(X_train_rnn.shape[0] / SEQ_LENGTH)
# nb_validation_sequences = int(X_test_rnn.shape[0] / SEQ_LENGTH)

# X_train_rnn = np.reshape(X_train_rnn, (X_train_rnn.shape[0], 1, X_train_rnn.shape[1]))
# X_test_rnn = np.reshape(X_test_rnn, (X_test_rnn.shape[0], 1, X_test_rnn.shape[1]))
# Y_train_rnn = np.reshape(Y_train_rnn, (Y_train_rnn.shape[0], Y_train_rnn.shape[1]))
# Y_test_rnn = np.reshape(Y_test_rnn, (Y_test_rnn.shape[0], Y_test_rnn.shape[1]))

'''
	===============================================
	5 - Entraînement du LSTM
	===============================================
'''

# print('\nEntraînement LSTM.')

# lstm = LSTM()
# lstm.train(X_train=X_train_rnn, Y_train=Y_train_rnn, X_test=X_test_rnn, Y_test=Y_test_rnn, epochs=200)
# lstm.save_weights(SAVED_MODELS_DIR + '/LSTM_GreenHillZone.h5')
# lstm.load_weights(SAVED_MODELS_DIR + '/LSTM_GreenHillZone.h5')

'''
	===============================================
	5 - Variante - Entraînement du MDN_LSTM
	===============================================
'''

# TODO : ne fonctionne plus, à régler, erreur non-explicite
# print('\nEntraînement MDN_LSTM.')

# mdn_lstm = MDN_LSTM()
# mdn_lstm.train(X_train=X_train_rnn, Y_train=Y_train_rnn, val_data=(X_test_rnn, Y_test_rnn), epochs=200)
# mdn_lstm.save_weights(SAVED_MODELS_DIR + '/MDN_LSTM.h5')
# mdn_lstm.load_weights(SAVED_MODELS_DIR + '/MDN_LSTM.h5')

'''
	===============================================
	6 - Jouer à sonic dans le LSTM entraîné
	===============================================
'''

# print("\nYou are playing inside LSTM's dream !")

# The LSTM begin from a random latent image
# latent_image = [np.random.rand(LATENT_DIM)]
# lstm.play_in_dream(start_image=latent_image, decoder=vae.decoder)
