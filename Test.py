from Table import Table
import numpy as np

def TestTable(table, epochs = 100):
	#Run the agent in the environment with the given table with the epochs passed as parameter
	rewards = []
	for _ in range(epochs):
		table.Run(exploreChance = 0)
		rewards = np.append(rewards, table.reward)

	table.avgReward = np.mean(rewards)
	table.stdReward = np.std(rewards)

	return rewards[rewards > -200].size


def Compare(t1, t2):
	if(t1.avgReward == t2.avgReward):
		if(t1.reward == t2.reward):
			if(t1.closest <= t2.closest):
				return t1
			else:
				return t2

		elif(t1.reward > t2.reward):
			return t1
		else:
			return t2

	elif(t1.avgReward > t2.avgReward):
		return t1
	else:
		return t2
