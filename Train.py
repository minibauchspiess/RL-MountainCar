from Table import Table
from Test import TestTable, Compare

import gym
import numpy as np
from time import time
import csv
from copy import deepcopy

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


def Crossing(env, numCreatedTables=500, sizePos = DEFAULT_SIZEPOS, sizeVel = DEFAULT_SIZEVEL, saveTable=False, tableFile="", saveStatistics=False, statisticsFile='StatisticData.csv'):
	print("Starting training with Crossing method\nCreating table with size ", sizePos, "x", sizeVel)
	startTime = time()

	#Generate 4 good tables, from wich the crossing will be made
	t = []
	for i in range(4):
		aux,_,_,_ = TrainManyTables(env=env, expChance=0.2, numTables=numCreatedTables, sizePos=sizePos, sizeVel=sizeVel, verbose=False)
		t.append(aux)

	#Execute possible half/half crossing
	son=[]
	son.append(t[0].Crossing(t[1]))
	son.append(t[1].Crossing(t[0]))

	son.append(t[2].Crossing(t[3]))
	son.append(t[3].Crossing(t[2]))

	#Check wich is the best crossing result in both crossing proccesses (among t's and son's)
	best1 = Compare(son[0], son[1])
	best1 = Compare(best1,t[0])
	best1 = Compare(best1,t[1])

	best2 = Compare(son[2], son[3])
	best2 = Compare(best2,t[2])
	best2 = Compare(best2,t[3])

	#Execute quarter crossing among the bests, resulting in 4 grandsons
	offset = int(sizePos/2)
	grandson = []

	grandson.append(best1.Crossing(other=best2, offset=offset))
	grandson.append(best1.Crossing(other=best2, offset=-offset))
	grandson.append(best2.Crossing(other=best1, offset=offset))
	grandson.append(best2.Crossing(other=best1, offset=-offset))


	#Find best result among best's and grandson's
	TestTable(best1)
	TestTable(best2)
	qStar = Compare(best1, best2)

	for i in range(4):
		qStar = Compare(qStar, grandson[i])




	#Execute extra proccessing in the table, if requested
	#Save it
	if(saveTable):
		np.save(tableFile, qStar.table)

	winArray = []
	for _ in range(15):
		winArray = np.append(winArray, TestTable(qStar))
	meanWin = np.mean(winArray)
	stdWin = np.std(winArray)

	elapsedTime = time() - startTime

	print("Finished training")
	print("Average Reward: ", qStar.avgReward, "; Reward Deviation: ", qStar.stdReward, "; Winning Rate: ", meanWin, "+-", stdWin, "%; Training time: {:.2f}".format(elapsedTime), " seconds" )

	if(saveStatistics):
		SaveToCSV(['Crossing', str(sizePos)+'x'+str(sizeVel), qStar.avgReward, qStar.stdReward, meanWin, stdWin, elapsedTime], statisticsFile)

	return qStar


def SaveToCSV(row, statisticsFile):
	csvFile = open(statisticsFile,'a')
	writer = csv.writer(csvFile)
	writer.writerow(row)
	csvFile.close()


