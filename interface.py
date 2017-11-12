import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from agent import Agent

import progressbar
import json
import os
import copy
import time

class AEInterface:
	def __init__(self, number_of_trials=10000, number_of_episodes=200, horizon=20000):
		self.number_of_trials = number_of_trials
		# self.number_of_trials = 4
		self.number_of_episodes = number_of_episodes
		self.horizon = horizon
		self.return_history = {}
		self.system_stats = {
			'number_of_trials': 0,
			'time_taken': 0
		}

		self.hyperparameters = {
			# "alpha": 0.02,
			# "gamma": 1,
			# "epsilon": 0.25,
			# "be_degree": 3
		}


	def execute(self, debug=True, persist=True, reload=True, mode=0, filepath='stest'):
		
		agent_hparams = self.hyperparameters

		if reload:
			if os.path.isfile(filepath+'/system_stats.json'):
				with open('logs/system_stats.json', 'r') as fp:
					self.system_stats = json.load(fp)

				with open(filepath+'/return_history.json', 'r') as fp:
					self.return_history = json.load(fp)

		print self.number_of_trials
		print self.system_stats
		print filepath

		for trial in range(self.system_stats['number_of_trials'], self.number_of_trials):

			if debug:
				print "-----------------------Trial ",trial," --------------------------------------"

			tic = time.time()

			environment = Environment()
			agent = Agent(agent_hparams, environment, mode=mode)

			bar = progressbar.ProgressBar(maxval=self.number_of_episodes, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
			bar.start()

			for episode in range(self.number_of_episodes):
				try:
					self.return_history[episode]
				except KeyError:
					self.return_history[episode] = []
				
				self.episode = episode
				
				# Refresh the agent here
				agent.reset()
				self.time = 0

				while True:

					agent.run_agent()

					self.time += 1

					episode_completion_status = agent.get_episode_status()

					if episode_completion_status or self.time>=self.horizon:
						break

				self.return_history[episode].append(agent.returns)
				
				bar.update(episode)

			bar.finish()

			toc = time.time()
			time_taken = toc-tic

			print time_taken," s"
			# print self.return_history

			if persist:
				print "saving"

				self.system_stats['time_taken'] += time_taken

				with open(os.path.join(os.path.dirname(__file__),filepath+'/return_history.json'), 'w') as fp:
					json.dump(self.return_history, fp)

				self.system_stats['number_of_trials'] = trial+1

				with open(os.path.join(os.path.dirname(__file__),filepath+'/system_stats.json'), 'w') as fp:
					json.dump(self.system_stats, fp)


		print "time taken: ",self.system_stats['time_taken']