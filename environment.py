from __future__ import division

import numpy as np

seed = 42
np.random.seed(seed)

class Environment:
	def __init__(self):
		
		self.action_list = [ -1, 0, 1 ]
		
		self.number_of_state_variables = 2

		self.terminal_states = [ 0.5 ]

		self.position_lbound = -1.2
		self.position_ubound = 0.5
		self.velocity_lbound = -0.07
		self.velocity_ubound = 0.07

		self.episode_completion_flag = False
		self.cur_position = 0
		self.cur_velocity = 0

		self.reset()


	def interact(self, a):
		reward = -1

		self.cur_velocity += 0.001*a - 0.0025*np.cos(3*self.cur_position)

		if self.cur_velocity > self.velocity_ubound:
			self.cur_velocity = self.velocity_ubound

		if self.cur_velocity < self.velocity_lbound:
			self.cur_velocity = self.velocity_lbound

		self.cur_position += self.cur_velocity

		if self.cur_position > self.position_ubound:
			self.episode_completion_flag = True
			self.cur_position = self.position_ubound

		if self.cur_position < self.position_lbound:
			self.cur_velocity = 0
			self.cur_position = self.position_lbound

		return reward


	def get_state(self):
		return [ self.cur_position, self.cur_velocity ]


	def get_episode_status(self):
		return self.episode_completion_flag


	def reset(self):
		self.cur_position = -0.5
		self.cur_velocity = 0
		self.episode_completion_flag = False