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
		self.encoder.predict(rand_image)		# warmup
		self.session = K.get_session()
		self.graph = tf.get_default_graph()
		self.graph.finalize()
		self.queue = Queue()

	# Evaluation d'un seul réseau
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
			step = 0
			# l'étape à laquelle l'individu à obtenu son meilleur score
			best_score_step = 0
			# le score final de l'individu (pas forcément le même que sont meilleur score sur la session)
			total_score = 0.0
			# le meilleur score de l'individu (là où il est allé le plus loin)
			best_score = 0.0
			# nombre d'étapes sans progression du meilleur score
			steps_without_progress = 0

			while 1:
				# Le jeu est en 60 fps : on ne fait jouer l'IA qu'en 15 fps (toutes les 4 frames)
				if step % 4 == 0:
					action = np.zeros((12,), dtype=np.bool)

					if net is not None:
						output = net.activate(latent_vector)

						bool_output = []
						# Fonction d'activation = clamped
						# Les valeurs sont donc bornées entre -1 et 1 que l'on va convertir en False ou True
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

				# S'il s'agit d'une des trois frames où l'IA ne prend pas de décision, elle répète simplement sa dernière action
				else:
					action = last_action

				observation, reward, done, info = env.step(action)

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

				if done or step >= MAX_STEPS or steps_without_progress >= MAX_STEPS_WITHOUT_PROGRESS:
					break

				step += 1

			# TODO : revoir les rewards
			# score = best_score - best_score_step
			score = best_score
			print(score)
			print(episodes_score)
			episodes_score[net_index] = score
			genome.fitness = score
			self.queue.task_done()
		del env

	def evaluate_genomes(self, genomes, config):
		t0 = time.time()

		# Création du réseau de chaque individu de la population
		nets = []
		for gid, g in genomes:
			nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
			g.fitness = []

		print("network creation time {0}".format(time.time() - t0))
		t0 = time.time()
		episodes_score = np.zeros(len(nets))  # Initialisation du tableau avec des zéros
		net_index = 0

		for i in range(NB_THREADS):
			t = threading.Thread(target=self.eval_net)
			t.start()

		for genome, net in nets:
			self.queue.put([net, genome, episodes_score, net_index])
			net_index += 1

		self.queue.join()

		print("simulation run time {0}".format(time.time() - t0))

		scores = [s for s in episodes_score]
		score_range.append((min(scores), np.mean(scores), max(scores)))
		print('best score : ' + max(scores))


def run():
	env = retrowrapper.RetroWrapper(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1',
					 use_restricted_actions=retro.ACTIONS_ALL, scenario='scenario')
	vae = VAE()
	vae.load_weights(file_path=SAVED_MODELS_DIR + '/VAE.h5')
	encoder = vae.encoder
	# Load the config file, which is assumed to live in
	# the same directory as this script.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config')
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_path)

	# Create the population, which is the top-level object for a NEAT run.
	pop = neat.Population(config)
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)
	# Add a stdout reporter to show progress in the terminal.
	pop.add_reporter(neat.StdOutReporter(True))
	# Checkpoint every 10 generations or 900 seconds.
	pop.add_reporter(neat.Checkpointer(10, 900))

	# Run until the winner from a generation is able to solve the environment
	# or the user interrupts the process.
	ec = PooledErrorCompute()
	while 1:
		try:
			pop.run(ec.evaluate_genomes, 1)

			visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

			if score_range:
				S = np.array(score_range).T
				plt.plot(S[0], 'r-')
				plt.plot(S[1], 'b-')
				plt.plot(S[2], 'g-')
				plt.grid()
				plt.savefig("score-ranges.svg")
				plt.close()

			mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
			print("Average mean fitness over last 5 generations: {0}".format(mfs))

			mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
			print("Average min fitness over last 5 generations: {0}".format(mfs))

			# Use the best genome seen so far as an ensemble-ish control system.
			best_genome = stats.best_unique_genomes(1)[0]
			best_network = neat.nn.FeedForwardNetwork.create(best_genome, config)

			solved = True
			best_scores = []
			# for k in range(100):
			for k in range(1):
				observation = env.reset()
				latent_vector = encoder.predict(np.array([observation]))[0]
				score = 0
				for i in range(MAX_STEPS):
					best_action = best_network.activate(latent_vector)
					observation, reward, done, info = env.step(best_action)
					latent_vector = encoder.predict(np.array([observation]))[0]
					score += reward
					env.render()
					if done:
						break

				best_scores.append(score)
				avg_score = sum(best_scores) / len(best_scores)
				print(k, score, avg_score)
				if avg_score < MAX_REWARD:
					solved = False
					break

			if solved:
				print("Solved.")

				# Save the winners.
				for n, g in best_genome:
					name = 'winner-{0}'.format(n)
					with open(name + '.pickle', 'wb') as f:
						pickle.dump(g, f)

					visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
					visualize.draw_net(config, g, view=False, filename="-net-enabled.gv", show_disabled=False)
					visualize.draw_net(config, g, view=False, filename="-net-enabled-pruned.gv",
									   show_disabled=False, prune_unused=True)

				break
		except KeyboardInterrupt:
			print("User break.")
			break

	env.close()


if __name__ == '__main__':
	run()
