from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def array_from_image(path):
	img = Image.open(path)
	# img = img.resize((128, 128))
	data = np.asarray(img, dtype=np.uint8)
	return data


def image_from_array(array):
	reconstructed_img = array * 255
	reconstructed_img = reconstructed_img.astype(np.uint8)
	return Image.fromarray(reconstructed_img, 'RGB')


def plot_graphs(history):
    # C'est l'historique des coûts qui nous intéresse
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()

    plt.show()
