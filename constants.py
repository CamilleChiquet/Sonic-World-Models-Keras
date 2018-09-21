LATENT_DIM = 128
NB_ACTIONS = 4
IMG_SHAPE = (224, 320, 3)

VAE_TRAINING_EXT = '.vae_train'
RNN_TRAINING_EXT = '.rnn_train'
RNN_TEST_EXT = '.rnn_test'

DATA_DIR = './data'
ACTIONS_DIR = DATA_DIR + '/actions'
IMG_DIR = DATA_DIR + '/images'
LATENT_IMG_DIR = DATA_DIR + '/latent_images'
SAVED_MODELS_DIR = './saved_models'

class Actions:
	JUMP = 0
	LEFT = 1
	RIGHT = 2
	DOWN = 3
