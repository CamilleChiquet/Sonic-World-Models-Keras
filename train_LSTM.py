from models.LSTM import getLSTMModel
import numpy as np


input_shape = (224, 320, 3)
epochs = 10


actions = np.load('data/actions/GreenHillZone.Act1.LSTM_test.npy')
print(np.shape(actions))
latent_images = np.load('data/latent_images/GreenHillZone.Act1.LSTM_test.npy')
print(np.shape(latent_images))
data = np.append(latent_images, actions, axis=1)
print(np.shape(data))
# model = getLSTMModel(input_shape=input_shape)
