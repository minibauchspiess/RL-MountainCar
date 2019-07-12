from Table import Table
import numpy as np
import sys
import gym

TABLES_FOLDER = "TrainedTables/"

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


def RunSingleTable(tableFile, showTable=False):
	#Load desired table to test
	try:
		table = np.load(TABLES_FOLDER+tableFile)
	except:
		print("No file with that name.\nCheck if it is inside "+TABLES_FOLDER+" folder, or if it has .npy extension")
		exit()

	#Show table, if requested
	if(showTable):
		print(table)

	#Create environment
	env = gym.make('MountainCar-v0')

	#Create table object, to receive q table
	numPos = np.size(table,0)
	numVel = np.size(table,1)
	tableObj = Table(env, numPos, numVel)
	tableObj.table = table

	#Run table, showing results
	tableObj.Run(show=True)

	#Close environment after finished
	env.close()
	



inputs = sys.argv
if(len(inputs) >= 2):
	table = inputs[1]
	if(len(inputs) >= 3):
		RunSingleTable(table, inputs[2])
	else:
		RunSingleTable(table)