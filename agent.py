from __future__ import division

import numpy as np

import copy


class FourierExpansion:
	def __init__(self, i):
		self.i = i
		self.list_all_comb = []
		self.expanded = None


	def compute_sum(self, s, i):
		i-=1

		if i<0:
			return

		for x in range(self.i):
			self.expanded[i] = x
			self.compute_sum(s, i)

			if x != self.i-1 or i==s.shape[0]-1:
		
				sum_1 = np.pi*np.sum(s*self.expanded)
				self.list_all_comb.append(sum_1)


	def compute(self, x):
		self.expanded = np.zeros(x.shape[0])

		self.compute_sum(x, x.shape[0])

		self.list_all_comb = np.cos(np.array(self.list_all_comb))

		ret = self.list_all_comb

		self.expanded = None
		self.list_all_comb = []

		return ret



class Agent:
	def __init__(self, h_params, environment, mode=0):
		self.mode = mode	# SARSA-0, Q-Learning-1
		self.time = 0
		self.environment = environment
		number_of_state_variables = self.environment.number_of_state_variables

		self.state_params = {
			"x1_l": -1.2,
			"x1_u": 0.5,
			"x2_l": -0.07,
			"x2_u": 0.07
		}
		self.actions = np.array(self.environment.action_list)

		self.hyperparameters = h_params
		self.hyperparameters['alpha'] = self.hyperparameters.get('alpha', 0.05)
		self.hyperparameters['gamma'] = self.hyperparameters.get('gamma', 1)
		self.hyperparameters['epsilon'] = self.hyperparameters.get('epsilon', 0.5)
		self.hyperparameters['be_degree'] = self.hyperparameters.get('be_degree', 1)

		self.basis_expansion = FourierExpansion(self.hyperparameters['be_degree'])

		# learnable parameter
		# self.be_size = np.power(self.basis_expansion.i, number_of_state_variables)
		self.be_size = number_of_state_variables
		self.w = np.ones(self.be_size*len(self.actions))	# Check initialization of w has any effects

		# Required global variables
		self.cur_state = None
		self.cur_action = None
		self.next_state = None

		# Implementation specific variables
		self.next_action = None		# Next Action variable for SARSA
		self.greedy_action = None	# Greedy action from action distribution

		self.returns = 0

		self.number_of_true_warnings = 0
		self.number_of_false_warnings = 0
		self.number_of_no_warnings = 0
		self.warning_type = None


	def reset(self):
		self.time = 0
		self.returns = 0
		self.number_of_true_warnings = 0
		self.number_of_false_warnings = 0
		self.number_of_no_warnings = 0
		self.environment.reset()
		self.cur_action = None
		self.cur_state = None
		self.next_state = None
		self.next_action = None
		self.greedy_action = None
		self.warning_type = {
			'no_warning': 0,
			'true_warning': 0,
			'false_warning': 0
		}


	def q_value(self, s, a):
		al, au = self.get_alimit(a)

		w_masked = self.w[al:au]
		return np.sum(w_masked*s)


	def w_update(self, delta, a):
		al, au = self.get_alimit(a)

		prev = self.w.copy()

		self.w[al:au] += delta


	def get_alimit(self, a):
		al = self.be_size*a
		au = al + self.be_size

		return (al, au)


	def scale_state(self, x):
		return [ 
				(x[0]-self.state_params['x1_l'])/(self.state_params['x1_u'] - self.state_params['x1_l']), 
				(x[1]-self.state_params['x2_l'])/(self.state_params['x2_u'] - self.state_params['x2_l']) 
			]


	def get_action(self, s, greedy=False):
		
		if not greedy:
			dice = np.random.random_sample()
			# print dice
			
			if dice < self.hyperparameters['epsilon']:
				return np.random.randint(len(self.actions))

		greedy_action = []
		best_q = None
		for i in range(len(self.actions)):
			q = self.q_value(s, i)

			if best_q == None or best_q < q:
				best_q = q
				greedy_action = [ i ]
			
			elif best_q == q:
				greedy_action.append(i)

		return np.random.choice(greedy_action)


	def sarsa(self):

		self.environment.step_reset()
		self.cur_state = None
		start = True

		while(True):

			if start:
				start = False
				self.cur_state = np.array(self.environment.get_state())
				# self.cur_state = self.basis_expansion.compute(self.cur_state)
				self.cur_action = self.get_action(self.cur_state)

			reward = self.environment.interact(self.cur_action)
			self.returns += reward			

			self.next_state = np.array(self.environment.get_state())
			# self.next_state = self.basis_expansion.compute(self.next_state)

			self.next_action = self.get_action(self.next_state)

			qsa = self.q_value(self.cur_state, self.cur_action)
			qsadot = self.q_value(self.next_state, self.next_action)

			if self.environment.get_instance_status():
				qsadot = 0

			delta = reward + self.hyperparameters['gamma']*qsadot - qsa

			delta *= self.cur_state
			delta *= self.hyperparameters['alpha']

			self.w_update(delta, self.cur_action)

			self.cur_state = self.next_state
			self.cur_action = self.next_action

			if self.environment.get_instance_status():
				break


	def test(self):
		self.environment.step_reset()

		while(True):

			self.cur_state = np.array(self.environment.get_state())
			# self.cur_state = self.basis_expansion.compute(self.cur_state)

			self.cur_action = self.get_action(self.cur_state, greedy=True)

			reward = self.environment.interact(self.cur_action)
			self.returns += reward

			if self.environment.warning_type != None:
				self.warning_type[self.environment.warning_type] += 1

			self.next_state = np.array(self.environment.get_state())
			# self.next_state = self.basis_expansion.compute(self.next_state)

			self.next_action = self.get_action(self.next_state, greedy=True)

			self.cur_state = self.next_state
			self.cur_action = self.next_action

			if self.environment.get_instance_status():
				break		


	def qlearning(self):
		next_position = 0
		if self.cur_action == None:
			self.cur_state = self.environment.get_state()
			# self.cur_state = self.scale_state(self.cur_state)
			# Basis Expansion of current state
			# self.cur_state = np.array(self.basis_expansion.compute(self.cur_state, self.state_params))

		self.cur_action = self.get_action(self.cur_state)
		reward = self.environment.interact(self.actions[self.cur_action])

		self.returns += np.power(self.hyperparameters['gamma'], self.time)*reward

		self.next_state = self.environment.get_state()
		next_position, velocity = self.next_state
		self.next_state = self.scale_state(self.next_state)
		self.next_state = np.array(self.basis_expansion.compute(self.next_state, self.state_params))	# Basis Expansion of next state

		self.greedy_action = self.get_action(self.next_state, greedy=True)
		if next_position in self.environment.terminal_states:
			delta = reward - self.q_value(self.cur_state, self.cur_action)
		else:
			delta = reward + self.hyperparameters['gamma']*self.q_value(self.next_state, self.greedy_action) - self.q_value(self.cur_state, self.cur_action)
		
		delta *= self.cur_state
		delta *= self.hyperparameters['alpha']


		self.w_update(delta, self.cur_action)

		self.cur_state = self.next_state

		self.time+=1


	def run_agent(self):
		if self.mode == 0:
			self.sarsa()
		else:
			self.qlearning()


	def get_episode_status(self):
		return self.environment.get_episode_status()