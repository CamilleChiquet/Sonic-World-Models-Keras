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
import math

MIN_REWARD = 0

levels_distances = 0
for end in LEVELS_END:
	levels_distances += end

MAX_STEPS_WITHOUT_PROGRESS = 600
MAX_STEPS = 4500
NB_THREADS = 8

score_range = []

def compute_fitness(distance, step):
	if step < 10:
		return distance
	return distance / math.log10(step)

# I played to the 2 levels and it took me about 35s and 38s to finish them
number_steps_to_beat = 35*60 + 38*60
REWARD_THRESHOLD = compute_fitness(levels_distances, number_steps_to_beat)
# REWARD_THRESHOLD = compute_fitness(9450, 35*60)
print('fitness threshold : ' + str(REWARD_THRESHOLD))

def run_net_in_env(env, session, graph, encoder, net, render=False):
	env.reset()
	latent_vector = np.zeros(LATENT_DIM)
	# The final score of the network (not necessary the best)
	cumulative_reward = 0.0
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

		# If you use the scenario.json of this project, reward represents the distance Sonic traveled since the last step
		# So, positive if he goes right and negative in the left direction
		observation, reward, done, info = env.step(action)
		if render:
			env.render()

		if info['lives'] < NB_LIFES_AT_START or done or steps_without_progress >= MAX_STEPS_WITHOUT_PROGRESS:
			break

		if (step - 1) % FRAME_JUMP == 0:
			with session.as_default():
				with graph.as_default():
					latent_vector = encoder.predict(np.array([observation]))[0]

		del observation

		cumulative_reward += reward

		# The score depends of the farthest sonic went AND the time he took to get there.
		# But in my opinion distance is most important than the time, that's why I choose to use log10.
		# Feel free to modify the way you compute the reward.
		current_score = compute_fitness(cumulative_reward, step)
		if current_score > best_score:
			best_score = current_score
			steps_without_progress = 0
		else:
			steps_without_progress += 1
	return best_score

class PopulationEvaluator(object):
	def __init__(self):
		# Loading autoencoder
		vae = VAE()
		vae.load_weights(file_path=SAVED_MODELS_DIR + '/VAE_GreenHillZone.h5')
		# We only use the encoder part
		self.encoder = vae.encoder
		# We need to initialize the network for multithreading
		# Check here for more info : https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads
		rand_image = np.random.rand(1, 224, 320, 3)
		self.encoder.predict(rand_image)
		self.session = K.get_session()
		self.graph = tf.get_default_graph()
		self.graph.finalize()

		# For multithreading, evalutations of networks will be added in this queue
		self.queue = Queue()
		# Indicates if the threads that perform fitness evaluation have been created or not
		self.workers_created = False

		self.finished_runs = 0

	# Evaluates the fitness of one network
	def eval_net(self):
		envs = []
		for level in LEVELS:
			envs.append(retrowrapper.RetroWrapper(game='SonicTheHedgehog-Genesis', state=level,
										use_restricted_actions=retro.ACTIONS_ALL, scenario='scenario'))
		while True:
			item = self.queue.get()
			net, genome, generation_scores, net_index = item[0], item[1], item[2], item[3]
			generation_scores[net_index] = 0

			for env in envs:
				best_env_score = run_net_in_env(env, self.session, self.graph, self.encoder, net)
				generation_scores[net_index] += best_env_score

			genome.fitness = generation_scores[net_index]
			self.finished_runs += 1
			print('run ' + str(self.finished_runs) + ' fitness : ' + str(genome.fitness))
			self.queue.task_done()

	def evaluate_genomes(self, genomes, config):
		t0 = time.time()
		self.finished_runs = 0

		# Creation of the population's networks
		nets = []
		for gid, g in genomes:
			nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
			g.fitness = 0

		print("network creation time {0}".format(time.time() - t0))
		t0 = time.time()
		generation_scores = np.zeros(len(nets))  # Array initialization with zeros
		net_index = 0

		# We create the threads only at the first generation
		# Once created, no need to create new ones, just use those which are already created
		if not self.workers_created:
			for i in range(NB_THREADS):
				t = threading.Thread(target=self.eval_net)
				t.start()
		self.workers_created = True

		for genome, net in nets:
			self.queue.put([net, genome, generation_scores, net_index])
			net_index += 1

		self.queue.join()

		print("simulation run time {0}".format(time.time() - t0))

		scores = [s for s in generation_scores]
		score_range.append((min(scores), np.mean(scores), max(scores)))
		print('best score : ' + str(max(scores)))


def run_neat(checkpoint=None):
	envs = []
	for level in LEVELS:
		envs.append(retrowrapper.RetroWrapper(game='SonicTheHedgehog-Genesis', state=level,
									use_restricted_actions=retro.ACTIONS_ALL, scenario='scenario'))
	# Load the config file, which is assumed to live in
	# the same directory as this script.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, NEAT_DIR, 'config')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_path)

	if checkpoint is not None:
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
	popEvaluator = PopulationEvaluator()
	while 1:
		try:
			solved = False
			total_score = 0

			# 'run' returns the best genome
			best_genome = pop.run(popEvaluator.evaluate_genomes, 1)

			visualize.plot_stats(stats, ylog=False, view=False, filename=NEAT_DIR + "/fitness.svg")

			# Contructs the network of the best genome
			best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)

			for env in envs:
				total_score += run_net_in_env(env, popEvaluator.session, popEvaluator.graph, popEvaluator.encoder, best_network, render=True)

			visualize.draw_net(config, best_genome, view=False, filename=NEAT_DIR + "/gen-" +str(pop.generation) + "-net", show_disabled=False)

			if total_score >= REWARD_THRESHOLD:
				solved = True

			print('best : ' + str(total_score))

			if solved:
				print("Solved.")

				# Save the winner
				name = NEAT_DIR + '/winner'
				with open(name + '.pickle', 'wb') as f:
					pickle.dump(best_genome, f)


				break

		except KeyboardInterrupt:
			print("User break.")
			break

	env.close()

def run_network(file_name):
	envs = []
	for level in LEVELS:
		envs.append(retrowrapper.RetroWrapper(game='SonicTheHedgehog-Genesis', state=level,
									use_restricted_actions=retro.ACTIONS_ALL, scenario='scenario'))

	vae = VAE()
	vae.load_weights(file_path=SAVED_MODELS_DIR + '/VAE_GreenHillZone.h5')
	encoder = vae.encoder
	rand_image = np.random.rand(1, 224, 320, 3)
	encoder.predict(rand_image)  # warmup
	session = K.get_session()
	graph = tf.get_default_graph()
	graph.finalize()

	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, NEAT_DIR, 'config')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_path)

	with open(file_name, 'rb') as f:
		best_genome = pickle.load(f)

	best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)

	for env in envs:
		run_net_in_env(env, session, graph, encoder, best_network, render=True)

if __name__ == '__main__':
	# run_neat(checkpoint=NEAT_DIR + '/neat-checkpoint-131')
	run_neat()
	# run_network(NEAT_DIR + '/winner.pickle')
