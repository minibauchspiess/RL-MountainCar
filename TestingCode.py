'''
Code used to test the training functions

'''

import gym
import numpy as np
from Table import Table, FindNearest
from Test import TestTable, Compare
import Train
import time


TRAINING_FOLDER = "TrainedTables/"


MATSIZE = 15

env = gym.make('MountainCar-v0')


for i in range(5):
	Train.ManySingleFixed(env=env, sizePos=MATSIZE, sizeVel=MATSIZE, verbose=False, saveTable=True, tableFile=(TRAINING_FOLDER+"SingleFixed_"+str(MATSIZE)+"x"+str(MATSIZE)+"_"+str(i+1)), saveStatistics=True)

for i in range(5):
	Train.ManySingleExploring(env=env, expChance=0.2, sizePos=MATSIZE, sizeVel=MATSIZE, verbose=False, saveTable=True, tableFile=(TRAINING_FOLDER+"SingleExploring_"+str(MATSIZE)+"x"+str(MATSIZE)+"_"+str(i+1)), saveStatistics=True)

for i in range(5):
	Train.ManyMultipleExploring(env=env, expChance=0.2, epochs=5, sizePos=MATSIZE, sizeVel=MATSIZE, verbose=False, saveTable=True, tableFile=(TRAINING_FOLDER+"MultipleExploring_"+str(MATSIZE)+"x"+str(MATSIZE)+"_"+str(i+1)), saveStatistics=True)



MATSIZE = 8

for i in range(5):
	Train.ManySingleFixed(env=env, sizePos=MATSIZE, sizeVel=MATSIZE, verbose=False, saveTable=True, tableFile=(TRAINING_FOLDER+"SingleFixed_"+str(MATSIZE)+"x"+str(MATSIZE)+"_"+str(i+1)), saveStatistics=True)

for i in range(5):
	Train.ManySingleExploring(env=env, expChance=0.2, sizePos=MATSIZE, sizeVel=MATSIZE, verbose=False, saveTable=True, tableFile=(TRAINING_FOLDER+"SingleExploring_"+str(MATSIZE)+"x"+str(MATSIZE)+"_"+str(i+1)), saveStatistics=True)

for i in range(5):
	Train.ManyMultipleExploring(env=env, expChance=0.2, epochs=5, sizePos=MATSIZE, sizeVel=MATSIZE, verbose=False, saveTable=True, tableFile=(TRAINING_FOLDER+"MultipleExploring_"+str(MATSIZE)+"x"+str(MATSIZE)+"_"+str(i+1)), saveStatistics=True)




env.close()


