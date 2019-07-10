import gym
import numpy as np
from random import random
from copy import deepcopy


class Table():
	def __init__(self, env, numPos, numVel):
		self.env = env

		self.numPos = numPos
		self.numVel = numVel

		self.posResol = (env.observation_space.high[0] - env.observation_space.low[0]) / numPos
		self.velResol = (env.observation_space.high[1] - env.observation_space.low[1]) / numVel

		self.table = np.ones((numPos, numVel), int) * (-1)


		self.reward = 0
		self.closest = (env.observation_space.high[0] - env.observation_space.low[0])

		self.avgReward = -200

	def GetAction(self, pos, vel):
		#Using the inputs and the discretization range (passed in the creation of the object), find the corresponding index in the table, and return the value in that position
		lin = (pos - self.env.observation_space.low[0])//self.posResol
		if(lin==self.numPos):
			lin = self.numPos-1

		col = (vel - self.env.observation_space.low[1])//self.velResol
		if(col==self.numVel):
			col = self.numVel-1

		return self.table[int(lin),int(col)]

	def SaveAction(self, pos, vel, val):
		lin = (pos - self.env.observation_space.low[0])//self.posResol
		if(lin==self.numPos):
			lin = self.numPos-1

		col = (vel - self.env.observation_space.low[1])//self.velResol
		if(col==self.numVel):
			col = self.numVel-1

		self.table[int(lin),int(col)] = val

	def Run(self, show=False, updateTable=True, exploreChance = 0.1):
		#Set initial state for environment
		state = self.env.reset()

		self.reward = 0

		#Perform interactions with the environment until completion condition is reached
		done = False
		while(not done):
			#Extract state values to more mnemonic variable
			pos = state[0]		#position of the car (in range of -1.2 to 0.5)
			vel = state[1]		#velocity of the car (in range of -0.7 to 0.7)

			#If the current position is the closest the car has gotten to the goal, update the closest atribute
			if(self.closest > (self.env.observation_space.high[0] - pos)):
				self.closest = (self.env.observation_space.high[0] - pos)

			#Select action according to the table built so far
			act = self.GetAction(pos, vel)
			if((act == -1) or (random() < exploreChance)):			#If no action in this state has been taken before, chose random action
				act = self.env.action_space.sample()
				if(updateTable):
					self.SaveAction(pos, vel, act)

			#Execute action
			state, reward, done, info = self.env.step(act)
			if(show):
				self.env.render()

			#Update the reward
			self.reward += reward

	def InterpolNearest(self, reach):
		#Make a copy, to aply results only after proccess is finished
		aux = deepcopy(self.table)

		[posy, posx] = np.where(self.table == -1)
		for i in range(posy.size):
			aux[posy[i], posx[i]] = FindNearest(self.table, posy[i], posx[i], reach)

		self.table = aux

	def Merge(self, other):
		#Check whitch object has the table with best result
		if(Compare(self, other) == self):
			#Change only the values not uncovered of it self
			self.table[self.table==-1] = other.table[self.table==-1]

			return 1


		else:
			#Take every value already discovered from the other table and put it in itself
			aux = other.table
			aux[aux==-1] = self.table[aux==-1]
			self.table = aux

			#Update reward with the reward given by the best table
			self.reward = other.reward
			self.closest = other.closest

			return 2

	def Crossing(self, other, offset=0):
		#Changes upper diagonal values of self table to values of other table
		indUpperTr = np.triu_indices(self.numPos, offset)	#Get index values for upper triangle in self.table matrix

		son = Table(self.env, self.numPos, self.numVel)
		son.table = deepcopy(self.table)

		son.table[indUpperTr] = other.table[indUpperTr]

		return son

	def Mutate(self, conf):
		new = Table(self.env, self.numPos, self.numVel)
		new.table = deepcopy(self.table)
		#Changes some values inside self table randomly (but considering given confiability in it self to change the probability of change)
		new.table[np.random.rand(*new.table.shape) < conf] = new.env.action_space.sample()
		return new




def FindNearest(table, y, x, reach):
	for i in range(1, reach+1):
		aux = deepcopy(table[max(0,(y-i)):min(table.size, (y+i+1)), max(0,(x-i)):min(table.size, (x+i+1))])
		[posy, posx] = np.where(aux != -1)		#find the positions where the auxiliar table has values different than -1
		if(posy.size > 0):						#Condition where at least one value different than -1 was found
			return round(int((np.sum(aux[posy,posx]) / posy.size)))

	return -1



