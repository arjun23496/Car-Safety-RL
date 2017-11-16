from __future__ import division

import numpy as np
import pandas as pd

# seed = 42
# np.random.seed(seed)


class Environment:
	def __init__(self, filename="data/data.csv", crash_prob=0.5, ttc_delta=1, initial_trust=1.0, stopping_time=1):
		
		# Manipulatable variables
		self.stopping_time = stopping_time
		self.initial_trust = initial_trust
		self.trust_decay_rate = 0.02
		self.trust_inc_rate = 0.01
		self.max_trust = 0.8
		self.min_trust = 0.5
		self.ttc_delta = ttc_delta
		self.crash_prob = crash_prob
		self.num_stability = 1e-10

		self.action_list = [ 'warn', 'not_warn' ]

		self.probabilities = pd.read_csv(filename)
		self.crash = False
		self.end_instance = False

		self.ttc_mu = 5
		self.ttc_sigma = 1
		self.ttc = 0

		self.number_of_crashes = 0

		## Warning type
		# 0 - true warning
		# 1 - false warning
		# 2 - no warning
		self.warning_type = None
		
		# ttc limit
		self.crash_ttc = 2

		## Age - Code ##
		# 0 - below 18
		# 1 - 18 to 25
		# 2 - 25 to 65
		# 3 - above 65

		## Number of Occupants - Code ##
		# 0 - 0
		# 1 - 1
		# 2 - Greater than equal to 2
		
		self.number_of_state_variables = 9

		self.statedef_constant = {
			'age': pd.unique(self.probabilities['age']),
			'sex': pd.unique(self.probabilities['sex']),
		}

		self.statedef = {
			'alcinvol': pd.unique(self.probabilities['alcinvol']),
			'druginv': pd.unique(self.probabilities['druginv']),
			'dridistract': pd.unique(self.probabilities['dridistract']),
			'driverdrowsy': pd.unique(self.probabilities['driverdrowsy']),
			'prevacc': pd.unique(self.probabilities['prevacc']),
			'numoccs': pd.unique(self.probabilities['numoccs']) 
		}

		step_variable_states = [ 'alcinvol', 'druginv', 'dridistract', 'driverdrowsy' ]

		self.cur_state = {
			'age': None,
			'sex': None,
			'alcinvol': None,
			'druginv': None,
			'dridistract': None,
			'driverdrowsy': None,
			'prevacc': None,
			'numoccs': None 
		}

		## Globally changing variables
		self.trust = None
		self.running_crash_prob = None
		self.initial_crash_prob = None


	def bound_trust(self):
		if self.trust > self.max_trust:
			self.trust = self.max_trust
		
		if self.trust < self.min_trust:
			self.trust = self.min_trust


	def interact(self, a):

		a = self.action_list[a]

		dice = np.random.sample()

		crash_status = dice < self.running_crash_prob

		if crash_status:
			if a == 'warn':
				self.warning_type = 'true_warning'
				dice = np.random.sample()

				self.trust += self.trust_inc_rate
				
				self.bound_trust()

				if dice < self.trust:
					self.end_instance = True
					return 10
				else:
					self.crash = True
					self.number_of_crashes += 1
					self.end_instance = True
					return -10
			else:
				self.warning_type = 'no_warning'

				self.trust -= self.trust_decay_rate
				self.bound_trust()
				
				self.crash = True
				self.number_of_crashes += 1
				self.end_instance = True
				return -10
		else:
			if a == 'warn':
				self.warning_type = 'false_warning'
				self.trust -= self.trust_decay_rate
				self.bound_trust()
				self.end_instance = True
				return -10

		ttc_scaled = (1-self.initial_crash_prob)*(self.ttc - 2.5)/5.5

		self.running_crash_prob += (ttc_scaled - 1)**2

		self.ttc -= self.ttc_delta

		if self.ttc < self.stopping_time:
			self.crash = True
			self.number_of_crashes += 1
			self.end_instance = True

			return -10

		return 0


	def get_state(self):
		temp_state = self.cur_state.copy()
		temp_state['ttc'] = self.ttc

		for x in self.statedef_constant:
			maximum = np.max(self.statedef_constant[x])
			minimum = np.min(self.statedef_constant[x])

			if maximum != minimum:
				temp_state[x] = (temp_state[x] - minimum)/( maximum - minimum )
			else:
				temp_state[x] = 1

		for x in self.statedef:
			maximum = np.max(self.statedef[x])
			minimum = np.min(self.statedef[x])
			
			if maximum != minimum:
				temp_state[x] = (temp_state[x] - minimum)/( maximum - minimum )
			else:
				temp_state[x] = 1

		temp_state['ttc'] = (temp_state['ttc'] - 2.5)/5.5

		return temp_state.values()


	def get_instance_status(self):
		return self.end_instance


	def get_prob(self):
		prob = np.ones(self.probabilities.shape[0])
		
		for x in self.cur_state:
			prob *= (self.probabilities[x] == self.cur_state[x]).astype('int64')

		prob = prob.astype('bool')

		prob = self.probabilities[prob]

		if prob.shape[0] == 0:
			return 0
		else:
			return prob['probability'].iloc[0]


	def step_reset(self):
		self.warning_type = None
		self.end_instance = False
		self.crash = False

		for key in self.statedef:
			self.cur_state[key] = np.random.choice(self.statedef[key])

		self.ttc = np.random.normal(self.ttc_mu, self.ttc_sigma)

		if self.ttc < self.stopping_time:
			self.ttc = self.stopping_time + 1.5
		elif self.ttc > 8.0:
			self.ttc = 8.0

		p_x_crash = self.get_prob()

		initial_crash_prob = (self.crash_prob*p_x_crash)/np.power(1-self.crash_prob, 2)
		initial_crash_prob *= ( 1 - self.crash_prob - ((self.crash_prob*p_x_crash)*(np.log(self.num_stability+((self.crash_prob*p_x_crash + 1 - self.crash_prob)/(self.crash_prob*p_x_crash + self.num_stability))) ) ))

		self.initial_crash_prob = initial_crash_prob
		self.running_crash_prob = self.initial_crash_prob


	def reset(self):
		self.number_of_crashes = 0
		self.trust = self.initial_trust

		for key in self.statedef_constant:
			self.cur_state[key] = np.random.choice(self.statedef_constant[key])