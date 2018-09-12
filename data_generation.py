import pyglet
import sys
import numpy as np
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *
import ctypes
import retro
from models.VAE import *


class ButtonCodes:
	A = 15
	B = 16
	X = 17
	Y = 18
	START = 8
	SELECT = 9
	XBOX = 14
	LEFT_BUMPER = 12
	RIGHT_BUMPER = 13
	RIGHT_STICK = 11
	LEFT_STICK = 10
	D_LEFT = 6
	D_RIGHT = 7
	D_UP = 4
	D_DOWN = 5


def generateData(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='scenario', extension_name='', frame_jump=1):
	'''
	Permet de jouer à Sonic et d'enregistrer les images et actions de la session de jeu afin de créer des données d'entraînement

	:param game: jeu à charger
	:param state: niveau à charger
	:param scenario:
	:param extension_name: extension des tableaux d'images et actions sauvegardés
	:param frame_jump: facteur de prise d'images
		ex :frame_jump = 1 : chaque image de la session de jeu est sauvegardée
			frame_jump = 3 : on sauvegarde seulement une image toutes les 3 images
	:return:
	'''
	jump = frame_jump
	# Chargement du jeu et niveau
	env = retro.make(game=game, state=state, use_restricted_actions=retro.Actions.ALL, scenario=scenario)
	obs = env.reset()


	win_width = 1200
	screen_height, screen_width = obs.shape[:2]
	win_height = win_width * screen_height // screen_width
	win = pyglet.window.Window(width=win_width, height=win_height, vsync=False)

	# Servira a détecter les touches appuyées
	key_handler = pyglet.window.key.KeyStateHandler()
	win.push_handlers(key_handler)

	key_previous_states = {}

	pyglet.app.platform_event_loop.start()

	fps_display = pyglet.clock.ClockDisplay()
	clock.set_fps_limit(60)

	glEnable(GL_TEXTURE_2D)
	texture_id = GLuint(0)
	glGenTextures(1, ctypes.byref(texture_id))
	glBindTexture(GL_TEXTURE_2D, texture_id)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

	# Tableaux d'images du jeu
	images = []
	# Tableay d'action associé aux images
	actions = []
	while not win.has_exit:
		win.dispatch_events()

		win.clear()

		keys_clicked = set()
		keys_pressed = set()
		for key_code, pressed in key_handler.items():
			if pressed:
				keys_pressed.add(key_code)

			if not key_previous_states.get(key_code, False) and pressed:
				keys_clicked.add(key_code)
			key_previous_states[key_code] = pressed

		buttons_pressed = set()

		# Fin de simu
		if keycodes.ESCAPE in keys_pressed:
			sys.exit(1)
		# Sert à réinitialiser les images et actions
		elif keycodes.C in keys_pressed:
			print('reset record')
			images = []
			actions = []
		# Enregistrement des tableaux d'images et d'actions capturés pendant la partie
		elif keycodes.R in keys_pressed:
			print('save record')
			images = np.array(images, dtype=np.uint8)
			np.save('data/images/' + state + extension_name, images)

			actions = np.array(actions, dtype=np.bool)
			np.save('data/actions/' + state + extension_name, actions)
			sys.exit(1)

		inputs = {
			'A': keycodes.Z in keys_pressed or ButtonCodes.A in buttons_pressed,
			'B': keycodes.A in keys_pressed or ButtonCodes.B in buttons_pressed,
			'C': keycodes.E in keys_pressed,
			'X': keycodes.Q in keys_pressed or ButtonCodes.X in buttons_pressed,
			'Y': keycodes.S in keys_pressed or ButtonCodes.Y in buttons_pressed,
			'Z': keycodes.D in keys_pressed,

			'UP': keycodes.UP in keys_pressed or ButtonCodes.D_UP in buttons_pressed,
			'DOWN': keycodes.DOWN in keys_pressed or ButtonCodes.D_DOWN in buttons_pressed,
			'LEFT': keycodes.LEFT in keys_pressed or ButtonCodes.D_LEFT in buttons_pressed,
			'RIGHT': keycodes.RIGHT in keys_pressed or ButtonCodes.D_RIGHT in buttons_pressed,

			'MODE': keycodes.TAB in keys_pressed or ButtonCodes.SELECT in buttons_pressed,
			'START': keycodes.ENTER in keys_pressed or ButtonCodes.START in buttons_pressed,
		}
		action = [inputs[b] for b in env.buttons]

		obs, rew, done, info = env.step(action)
		jump -= 1
		if jump == 0:
			jump = frame_jump
			images.append(obs)
			actions.append(action)

		glBindTexture(GL_TEXTURE_2D, texture_id)
		video_buffer = ctypes.cast(obs.tobytes(), ctypes.POINTER(ctypes.c_short))
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, obs.shape[1], obs.shape[0], GL_RGB, GL_UNSIGNED_BYTE, video_buffer)

		x = 0
		y = 0
		h = win.height
		w = win.width

		pyglet.graphics.draw(4,
							 pyglet.gl.GL_QUADS,
							 ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
							 ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
							 )

		fps_display.draw()

		win.flip()

		timeout = clock.get_sleep_time(False)
		pyglet.app.platform_event_loop.step(timeout)

		clock.tick()

	pyglet.app.platform_event_loop.stop()


def generate_latent_images(images_path, name):
	'''
	Charge un tableau d'images et sauvegarde sa version encodée par le VAE

	:param images_path: chemin où se trouvent les images à encoder
	:param name: nom du tableau d'images
	:return:
	'''
	input_shape = (224, 320, 3)
	# Charger le modèle que l'on souhaite
	model = getVAEModel(input_shape=input_shape)
	# Charger les poids de la partie encoder réseau entraîné
	model.load_weights('saved_models/VAE.h5')
	model = model.layers[1]

	images = np.load(images_path + name)
	images = images / 255

	# On a 3 sorties [z_mean, z_log_var, z]
	latent_images = model.predict(images)
	# Seul z nous intéresse
	latent_images = latent_images[2]
	print(np.shape(latent_images))

	np.save('data/latent_images/' + name, latent_images)


# generateData(extension_name='.LSTM_train', frame_jump=1)
generate_latent_images(images_path='data/images/', name='GreenHillZone.Act1.LSTM_train.npy')
