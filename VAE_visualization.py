import numpy as np
from keras.callbacks import ModelCheckpoint
from models.VAE import *
from process import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from constants import *


def generate_render(data_path, model_path, save_path=None):
	'''
	Affiche le rendu visuel du passage d'images à travers le VAE

	:param data_path: chemin des images à faire passer par le VAE
	:param model_path: chemin du VAE sauvegardé
	:param save_path: Si définit on sauvegarde les images sorties du VAE
	:return:
	'''
	images = np.load(data_path)
	images = images / 255

	model = getVAEModel(input_shape=IMG_SHAPE)
	model.load_weights(model_path)
	# model.summary()
	# model = model.layers[2]

	generated_images = []
	for image in images:
		img = np.reshape(image, (1, 224, 320, 3))
		img = model.predict(img)
		# img = model.predict(np.random.rand(1,LATENT_DIM))
		img = img.reshape(IMG_SHAPE)
		if(save_path != None):
			generated_images.append(img)
		plt.clf()
		plt.imshow(img, vmin=0, vmax=1)
		plt.pause(0.0000001)
	if(save_path != None):
		np.save(save_path, generated_images)


def compare_data(original_data_path, reconstructed_data_path):
	'''
	Compare cote à cote 2 jeux d'images
	L'intérêt est de comparer les images originales d'une session de jeu en parrallèle de celles passées par le VAE

	:param original_data_path:
	:param reconstructed_data_path:
	:return:
	'''
	origin_images = np.load(original_data_path)
	reconstructed_images = np.load(reconstructed_data_path)
	fig = plt.figure(figsize=(1, 2))
	for i in range(len(origin_images)):
		plt.clf()
		origin_img = origin_images[i]
		rec_img = reconstructed_images[i]
		fig.add_subplot(1, 2, 1)
		plt.imshow(origin_img)
		fig.add_subplot(1, 2, 2)
		plt.imshow(rec_img)
		plt.pause(0.0000001)


generate_render('data/images/GreenHillZone.Act1.test.npy', 'saved_models/VAE.h5')
# compare_data('data/images/GreenHillZone.Act1.test.npy', 'data/reconstructed_images/GreenHillZone.Act1.test.reconstructed.npy')