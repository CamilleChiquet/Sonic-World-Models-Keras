import pyglet
import sys
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *
import ctypes
import retro


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


def play(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='scenario'):
	env = retro.make(game=game, state=state, use_restricted_actions=retro.Actions.ALL, scenario=scenario)
	obs = env.reset()
	screen_height, screen_width = obs.shape[:2]

	key_handler = pyglet.window.key.KeyStateHandler()
	win_width = 1200
	win_height = win_width * screen_height // screen_width
	win = pyglet.window.Window(width=win_width, height=win_height, vsync=False)

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

		inputs = {
			'A': keycodes.Z in keys_pressed or ButtonCodes.A in buttons_pressed,
			'B': keycodes.X in keys_pressed or ButtonCodes.B in buttons_pressed,
			'C': keycodes.C in keys_pressed,
			'X': keycodes.A in keys_pressed or ButtonCodes.X in buttons_pressed,
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


play()