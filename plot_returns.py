import matplotlib.pyplot as plt
import numpy as np

import json

import seaborn as sns

return_history = None

def plot_values(data, title, system_stats):
	# Plot Return History
	ax = sns.tsplot(data=data, ci="sd")

	plt.title(title+' for '+str(system_stats['number_of_trials'])+' trials')
	plt.xlabel("Number of episodes")
	plt.ylabel("Mean Undiscounted Returns(Solid Line)")

	# plt.ylim([-1000, 0])
	manager = plt.get_current_fig_manager()
	manager.window.showMaximized()

	plt.grid(True)

	plt.tight_layout()
	plt.show()


mode = 's'
directory = 'stest'

with open(directory+'/return_history.json', 'r') as fp:
	return_history = json.load(fp)

with open(directory+'/trust_history.json', 'r') as fp:
	trust_history = json.load(fp)

with open(directory+'/crash_frequency.json', 'r') as fp:
	crash_frequency = json.load(fp)

with open(directory+'/warning_type.json', 'r') as fp:
	warning_type = json.load(fp)

with open(directory+'/system_stats.json', 'r') as fp:
	system_stats = json.load(fp)

number_of_episodes = len(warning_type['true_warning'])

# system_stats['number_of_trials'] = 2

preturn_history = np.zeros((number_of_episodes, system_stats['number_of_trials']))
pcrash_frequency = np.zeros((number_of_episodes, system_stats['number_of_trials']))
ptrust_history = np.zeros((number_of_episodes, system_stats['number_of_trials']))

for x in return_history:
	pcrash_frequency[int(x)] += np.array(crash_frequency[x][:system_stats['number_of_trials']])
	preturn_history[int(x)] += np.array(return_history[x][:system_stats['number_of_trials']])
	ptrust_history[int(x)] += np.array(trust_history[x][:system_stats['number_of_trials']])

preturn_history = preturn_history.T
pcrash_frequency = pcrash_frequency.T
ptrust_history = ptrust_history.T

plot_values(preturn_history, 'return_history', system_stats)
plot_values(pcrash_frequency, 'crash_frequency', system_stats)
plot_values(ptrust_history, 'trust_history', system_stats)

plt.plot(range(number_of_episodes), warning_type['true_warning'], color='b', label="true_warning")
plt.plot(range(number_of_episodes), warning_type['false_warning'], color='y', label="false_warning")
plt.plot(range(number_of_episodes), warning_type['no_warning'], color='r', label="no_warning")

plt.legend()

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)
plt.show()