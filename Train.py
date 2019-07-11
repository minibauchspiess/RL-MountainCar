from Table import Table
from Test import TestTable, Compare

import gym
import numpy as np
from time import time
import csv

DEFAULT_SIZEPOS = 10
DEFAULT_SIZEVEL = 10


def TrainManyTables(env, expChance=0, epochs=1, numTables=2000, sizePos=DEFAULT_SIZEPOS, sizeVel=DEFAULT_SIZEVEL, verbose=True):
	#Check starting time executing training and end time, to determine elapsed time during training
	startTime = time()

	#Create the specified number of tables, with no chance of changing it's already set values to a new random number, and select the table with the best result
	bestTable = Table(env, sizePos, sizeVel)
	bestTable.Run(exploreChance=0)
	TestTable(bestTable, 5)

	for i in range(numTables-1):
		t = Table(env, sizePos, sizeVel)
		for _ in range(epochs):
			t.Run(exploreChance=expChance)

		#Test table, getting it's average reward
		TestTable(t, 5)

		bestTable = Compare(bestTable, t)

		if((verbose) and ((i%100) == 0)):
			print("Done with ", i, " tables")

	#Get values to validade created table
	timesWon = TestTable(bestTable)
	#bestTable.Run(exploreChance=0, show=True)

	elapsedTime = time() - startTime
	return bestTable, bestTable.avgReward, timesWon, elapsedTime

def ManySingleFixed(env, numTables=2000, sizePos=DEFAULT_SIZEPOS, sizeVel=DEFAULT_SIZEVEL, verbose=True, saveTable=False, tableFile="", saveStatistics=False, statisticsFile='StatisticData.csv'):
	print("Starting training ManySingleFixed method\nCreating ", numTables, " tables with size ", sizePos, "x", sizeVel)

	qStar, avgReward, timesWon, elapsedTime = TrainManyTables(env=env, numTables=numTables, sizePos=sizePos, sizeVel=sizeVel, verbose=verbose)

	if(saveTable):
		np.save(tableFile, qStar.table)


	winArray = []
	for _ in range(15):
		winArray = np.append(winArray, TestTable(qStar))
	meanWin = np.mean(winArray)
	stdWin = np.std(winArray)


	print("Finished training")
	print("Average Reward: ", qStar.avgReward, "; Reward Deviation: ", qStar.stdReward, "; Times won: ", meanWin, "+-", stdWin, "%; Training time: {:.2f}".format(elapsedTime), " seconds" )

	if(saveStatistics):
		SaveToCSV(['ManySingleFixed', str(sizePos)+'x'+str(sizeVel), qStar.avgReward, qStar.stdReward, meanWin, stdWin, elapsedTime], statisticsFile)


	return qStar

def ManySingleExploring(env, expChance=0.2, numTables=2000, sizePos=DEFAULT_SIZEPOS, sizeVel=DEFAULT_SIZEVEL, verbose=True, saveTable=False, tableFile="", saveStatistics=False, statisticsFile='StatisticData.csv'):
	print("Starting training ManySingleExploring method\nCreating ", numTables, " tables with size ", sizePos, "x", sizeVel, " and with ", expChance*100, "% chance of exploring")

	qStar, avgReward, timesWon, elapsedTime = TrainManyTables(env=env, expChance=expChance, numTables=numTables, sizePos=sizePos, sizeVel=sizeVel, verbose=verbose)

	if(saveTable):
		np.save(tableFile, qStar.table)


	winArray = []
	for _ in range(15):
		winArray = np.append(winArray, TestTable(qStar))
	meanWin = np.mean(winArray)
	stdWin = np.std(winArray)

	print("Finished training")
	print("Average Reward: ", qStar.avgReward, "; Reward Deviation: ", qStar.stdReward, "; Times won: ", meanWin, "+-", stdWin, "%; Training time: {:.2f}".format(elapsedTime), " seconds" )

	if(saveStatistics):
		SaveToCSV(['ManySingleExploring', str(sizePos)+'x'+str(sizeVel), qStar.avgReward, qStar.stdReward, meanWin, stdWin, elapsedTime], statisticsFile)

	return qStar

def ManyMultipleExploring(env, expChance=0.2, epochs=5, numTables=2000, sizePos=DEFAULT_SIZEPOS, sizeVel=DEFAULT_SIZEVEL, verbose=True, saveTable=False, tableFile="", saveStatistics=False, statisticsFile='StatisticData.csv'):
	print("Starting training ManyMultipleExploring method\nCreating ", numTables, " tables with size ", sizePos, "x", sizeVel,", training ", epochs, " times each table and with ", expChance*100, "% chance of exploring")

	qStar, avgReward, timesWon, elapsedTime = TrainManyTables(env=env, expChance=expChance, epochs=epochs, numTables=numTables, sizePos=sizePos, sizeVel=sizeVel, verbose=verbose)

	if(saveTable):
		np.save(tableFile, qStar.table)


	winArray = []
	for _ in range(15):
		winArray = np.append(winArray, TestTable(qStar))
	meanWin = np.mean(winArray)
	stdWin = np.std(winArray)

	print("Finished training")
	print("Average Reward: ", qStar.avgReward, "; Reward Deviation: ", qStar.stdReward, "; Times won: ", meanWin, "+-", stdWin, "%; Training time: {:.2f}".format(elapsedTime), " seconds" )

	if(saveStatistics):
		SaveToCSV(['ManyMultipleExploring', str(sizePos)+'x'+str(sizeVel), qStar.avgReward, qStar.stdReward, meanWin, stdWin, elapsedTime], statisticsFile)

	return qStar


def Crossing(env, crossingTimes, numCreatedTables=100, sizePos = DEFAULT_SIZEPOS, sizeVel = DEFAULT_SIZEVEL):

	t = []
	for i in range(crossingTimes):
		t[i] = TrainManyTables(env = env, numTables = numCreatedTables, sizePos = sizePos, sizeVel = sizeVel)

	son = []
	for i in range(crossingTimes-1):
		son[i] = t[i].Crossing(t[i+1])


	t1 = TrainManyTables(env = env, numTables = numCreatedTables, sizePos = sizePos, sizeVel = sizeVel)
	t2 = TrainManyTables(env = env, numTables = numCreatedTables, sizePos = sizePos, sizeVel = sizeVel)
	son = t1.Crossing(t2)

	TestTable(son)
	worst = min(son.avgReward, t1.avgReward, t2.avgReward)

	if(son.avgReward == worst):
		pass
		#son = t1.Crossing(t2,2)
	elif(t1.avgReward == worst): 
		return t2.Crossing(son)
	else:
		return son.Crossing(t1)



def SaveToCSV(row, statisticsFile):
	csvFile = open(statisticsFile,'a')
	writer = csv.writer(csvFile)
	writer.writerow(row)
	csvFile.close()


