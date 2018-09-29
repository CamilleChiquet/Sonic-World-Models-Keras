LATENT_DIM = 128

# JUMP, LEFT, RIGHT, DOWN
NB_ACTIONS = 4

# (image height, image width, rgb)
IMG_SHAPE = (224, 320, 3)

# Fix recording size for the LSTM : 512 frames + actions
SEQ_LENGTH = 512

# Game runs at 60 fps, the IA plays at 15 fps (60 / 4)
FRAME_JUMP = 4

NB_LIFES_AT_START = 3

VAE_TRAINING_EXT = '.vae_train'
RNN_TRAINING_EXT = '.rnn_train'
RNN_TEST_EXT = '.rnn_test'

DATA_DIR = './data'
ACTIONS_DIR = DATA_DIR + '/actions'
IMG_DIR = DATA_DIR + '/images'
LATENT_IMG_DIR = DATA_DIR + '/latent_images'
NEAT_DIR = DATA_DIR + '/neat'
SAVED_MODELS_DIR = './saved_models'

class Actions:
	JUMP = 0
	LEFT = 1
	RIGHT = 2
	DOWN = 3
