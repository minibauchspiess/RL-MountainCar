from Table import Table

def TestTable(table, epochs = 100):
	#Run the agent in the environment with the given table with the epochs passed as parameter
	winTimes = 0
	rewardSum = 0
	for _ in range(epochs):
		table.Run(exploreChance = 0)
		rewardSum += table.reward

		if(table.reward > -200):
			winTimes += 1

	avgRwd = (rewardSum / epochs)
	table.avgReward = avgRwd

	return winTimes


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