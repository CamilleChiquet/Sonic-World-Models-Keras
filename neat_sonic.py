# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg
# NOTE: This was run using revision 1186029827c156e0ff6f9b36d6847eb2aa56757a of CodeReclaimers/neat-python, not a release on PyPI.
import gym
import matplotlib.pyplot as plt
import multiprocessing
import neat
import numpy as np
import os
import pickle
import time
import visualize
from models.VAE import VAE
from constants import *
import retrowrapper
import retro
from keras import backend as K
import tensorflow as tf
import threading
import time
from queue import Queue

MIN_REWARD = 0
MAX_REWARD = 9000
MAX_STEPS_WITHOUT_PROGRESS = 600
MAX_STEPS = 4500
NB_THREADS = 8

score_range = []


class PooledErrorCompute(object):
	def __init__(self):
		vae = VAE()
		vae.load_weights(file_path=SAVED_MODELS_DIR + '/VAE.h5')
		self.encoder = vae.encoder
		rand_image = np.random.rand(1, 224, 320, 3)
		self.encoder.predict(rand_image)  # warmup
		self.session = K.get_session()
		self.graph = tf.get_default_graph()
		self.graph.finalize()
		self.queue = Queue()
		self.finished_runs = 0
		self.workers_created = False

	# Evaluates the fitness of one network
	def eval_net(self):
		env = retrowrapper.RetroWrapper(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1',
										use_restricted_actions=retro.ACTIONS_ALL, scenario='scenario')
		while True:
			item = self.queue.get()
			net, genome, episodes_score, net_index = item[0], item[1], item[2], item[3]
			observation = env.reset()
			with self.session.as_default():
				with self.graph.as_default():
					latent_vector = self.encoder.predict(np.array([observation]))[0]

			# The step where the network did its best score
			best_score_step = 0
			# The final score of the network (not necessary the best)
			total_score = 0.0
			# The network's best score
			best_score = 0.0
			# Number of steps without progression since the bestcore's step
			steps_without_progress = 0

			for step in range(MAX_STEPS):
				# Game runs at 60 fps or the AI plays at 15 fps (every 4 frame)
				if step % FRAME_JUMP == 0:
					action = np.zeros((12,), dtype=np.bool)

					if net is not None:
						output = net.activate(latent_vector)

						bool_output = []
						# Activation function = clamped.
						# Output goes from -1 to 1, which we'll convert into a boolean value
						for value in output:
							if value <= 0:
								bool_output.append(False)
							else:
								bool_output.append(True)

						action = np.zeros((12,), dtype=np.bool)
						action[1] = bool_output[Actions.JUMP]
						action[6] = bool_output[Actions.LEFT]
						action[7] = bool_output[Actions.RIGHT]
						action[5] = bool_output[Actions.DOWN]

					last_action = action

				# If it is a frame where the AI doesn't play, we repeat the last action
				else:
					action = last_action

				observation, reward, done, info = env.step(action)

				if info['lives'] < NB_LIFES_AT_START or done or steps_without_progress >= MAX_STEPS_WITHOUT_PROGRESS:
					break

				with self.session.as_default():
					with self.graph.as_default():
						latent_vector = self.encoder.predict(np.array([observation]))[0]

				del observation

				total_score += reward
				if total_score > best_score:
					best_score = total_score
					best_score_step = step
					steps_without_progress = 0
				else:
					steps_without_progress += 1

			# TODO : modify the final score
			# score = best_score - best_score_step
			score = best_score
			episodes_score[net_index] = score
			genome.fitness = score
			self.queue.task_done()
			self.finished_runs += 1
			print('run ' + str(self.finished_runs) + ' score : ' + str(score))
		env.close()
		del env

	def evaluate_genomes(self, genomes, config):
		t0 = time.time()
		self.finished_runs = 0

		# Creation of the population's networks
		nets = []
		for gid, g in genomes:
			nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
			g.fitness = []

		print("network creation time {0}".format(time.time() - t0))
		t0 = time.time()
		episodes_score = np.zeros(len(nets))  # Array initialization with zeros
		net_index = 0

		# We create the threads only at the first generation
		# Once created, no need to create new ones, just use those which are already created
		if not self.workers_created:
			for i in range(NB_THREADS):
				t = threading.Thread(target=self.eval_net)
				t.start()
		self.workers_created = True

		for genome, net in nets:
			self.queue.put([net, genome, episodes_score, net_index])
			net_index += 1

		self.queue.join()

		print("simulation run time {0}".format(time.time() - t0))

		scores = [s for s in episodes_score]
		score_range.append((min(scores), np.mean(scores), max(scores)))
		print('best score : ' + str(max(scores)))


def run_neat(checkpoint=None):
	env = retrowrapper.RetroWrapper(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1',
									use_restricted_actions=retro.ACTIONS_ALL, scenario='scenario')
	# Load the config file, which is assumed to live in
	# the same directory as this script.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, NEAT_DIR, 'config')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_path)

	if checkpoint != None:
		pop = neat.Checkpointer.restore_checkpoint(checkpoint)
	else:
		# Create the population, which is the top-level object for a NEAT run.
		pop = neat.Population(config)

	# Checkpoint every generation or 900 seconds.
	pop.add_reporter(neat.Checkpointer(1, 900, filename_prefix=NEAT_DIR + '/neat-checkpoint-'))
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)
	# Add a stdout reporter to show progress in the terminal.
	pop.add_reporter(neat.StdOutReporter(True))
	# Run until the winner from a generation is able to solve the environment
	# or the user interrupts the process.
	ec = PooledErrorCompute()
	while 1:
		try:
			pop.run(ec.evaluate_genomes, 1)

			visualize.plot_stats(stats, ylog=False, view=False, filename=NEAT_DIR + "/fitness.svg")

			mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
			print("Average mean fitness over last 5 generations: {0}".format(mfs))

			mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
			print("Average min fitness over last 5 generations: {0}".format(mfs))

			# Use the best genome seen so far as an ensemble-ish control system.
			best_genome = stats.best_unique_genomes(1)[0]
			best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)

			solved = False

			observation = env.reset()
			with ec.session.as_default():
				with ec.graph.as_default():
					latent_vector = ec.encoder.predict(np.array([observation]))[0]

			best_score = 0
			total_score = 0
			steps_without_progress = 0
			for step in range(MAX_STEPS):
				# Game runs at 60 fps or the AI plays at 15 fps (every 4 frame)
				if step % FRAME_JUMP == 0:
					action = np.zeros((12,), dtype=np.bool)

					if best_network is not None:
						output = best_network.activate(latent_vector)

						bool_output = []
						# Activation function = clamped.
						# Output goes from -1 to 1, which we'll convert into a boolean value
						for value in output:
							if value <= 0:
								bool_output.append(False)
							else:
								bool_output.append(True)

						action = np.zeros((12,), dtype=np.bool)
						action[1] = bool_output[Actions.JUMP]
						action[6] = bool_output[Actions.LEFT]
						action[7] = bool_output[Actions.RIGHT]
						action[5] = bool_output[Actions.DOWN]

					last_action = action

				# If it is a frame where the AI doesn't play, we repeat the last action
				else:
					action = last_action
				observation, reward, done, info = env.step(action)
				env.render()
				step += 1

				if info['lives'] < NB_LIFES_AT_START or done or steps_without_progress >= MAX_STEPS_WITHOUT_PROGRESS:
					break

				with ec.session.as_default():
					with ec.graph.as_default():
						latent_vector = ec.encoder.predict(np.array([observation]))[0]

				del observation

				total_score += reward
				if total_score > best_score:
					best_score = total_score
					steps_without_progress = 0
				else:
					steps_without_progress += 1

			if best_score >= MAX_REWARD:
				solved = True

			if solved:
				print("Solved.")

				# Save the winner
				name = NEAT_DIR + '/winner'
				with open(name + '.pickle', 'wb') as f:
					pickle.dump(best_genome, f)

				visualize.draw_net(config, best_genome, view=False, filename=name + "-net.gv")
				visualize.draw_net(config, best_genome, view=False, filename="-net-enabled.gv", show_disabled=False)
				visualize.draw_net(config, best_genome, view=False, filename="-net-enabled-pruned.gv",
								   show_disabled=False, prune_unused=True)

				break

		except KeyboardInterrupt:
			print("User break.")
			break

	env.close()


if __name__ == '__main__':
	# run_neat(checkpoint=NEAT_DIR + '/neat-checkpoint-22')
	run_neat()
