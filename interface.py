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
	def __init__(self, number_of_trials=10000, number_of_episodes=200, horizon=1000, test_horizon=100):
		self.number_of_trials = number_of_trials
		# self.number_of_trials = 4
		self.number_of_episodes = number_of_episodes
		self.horizon = horizon
		self.test_horizon = test_horizon
		self.return_history = {}
		self.crash_frequency_history = {}
		self.trust_history = {}
		self.system_stats = {
			'number_of_trials': 0,
			'time_taken': 0
		}

		self.hyperparameters = {
			"alpha": 0.002,
			"gamma": 1,
			"epsilon": 0.25,
			"be_degree": 2
		}

		self.warning_type = {
			"true_warning": [],
			"false_warning": [],
			"no_warning": []
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
			self.warning_type = {
				"true_warning": [],
				"false_warning": [],
				"no_warning": []
			}


			bar = progressbar.ProgressBar(maxval=self.number_of_episodes, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
			bar.start()

			for episode in range(self.number_of_episodes):
				try:
					self.return_history[episode]
				except KeyError:
					self.return_history[episode] = []
					self.crash_frequency_history[episode] = []
					self.trust_history[episode] = []
				
				self.episode = episode
				
				# Refresh the agent here
				agent.reset()
				self.time = 0

				while True:

					agent.run_agent()

					self.time += 1

					if self.time>=self.horizon:
						break


				# Testing
				agent.reset()
				test_time = 0

				while True:
					test_time += 1
					agent.test()

					if test_time >= self.test_horizon:
						break

				self.return_history[episode].append(agent.returns)
				self.crash_frequency_history[episode].append(environment.number_of_crashes)
				self.trust_history[episode].append(environment.trust)

				print environment.number_of_crashes

				for z in self.warning_type:
					self.warning_type[z].append(agent.warning_type[z])
				
				bar.update(episode)

			bar.finish()

			# agent.reset()

			# test_history = []

			# self.time = 0

			# while True:
			# 	agent.run_agent()

			# 	self.time += 1

			# 	if self.time>=self.horizon:
			# 		break
			

			toc = time.time()
			time_taken = toc-tic

			print time_taken," s"
			# print self.return_history

			if persist:
				print "saving"

				self.system_stats['time_taken'] += time_taken

				with open(os.path.join(os.path.dirname(__file__),filepath+'/warning_type.json'), 'w') as fp:
					json.dump(self.warning_type, fp)

				with open(os.path.join(os.path.dirname(__file__),filepath+'/trust_history.json'), 'w') as fp:
					json.dump(self.trust_history, fp)

				with open(os.path.join(os.path.dirname(__file__),filepath+'/return_history.json'), 'w') as fp:
					json.dump(self.return_history, fp)

				with open(os.path.join(os.path.dirname(__file__),filepath+'/crash_frequency.json'), 'w') as fp:
					json.dump(self.crash_frequency_history, fp)

				self.system_stats['number_of_trials'] = trial+1

				with open(os.path.join(os.path.dirname(__file__),filepath+'/system_stats.json'), 'w') as fp:
					json.dump(self.system_stats, fp)


		print "time taken: ",self.system_stats['time_taken']