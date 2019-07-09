import gym
import numpy as np
from random import random


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

	def Run(self, random, show):
		#If random is active, table training uses no memory of other tables to guide it's own training
		if(random):
			#Set initial state for environment
			state = self.env.reset()

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
				if(act == -1):			#If no action in this state has been taken before, chose random action
					act = self.env.action_space.sample()
					self.SaveAction(pos, vel, act)

				#Execute action
				state, reward, done, info = self.env.step(act)
				if(show):
					self.env.render()

				#Update the reward
				self.reward += reward

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

	def Crossing(self, other, selfConf):
		pass
		for i in range(self.numPos):
			for j in range(self.numVel):
				pass 
		#Changes some values of self table to the values of the other table, given the confiability selfConf in the self table


	def Mutate(self, conf):
		pass
		#Changes some values inside self table randomly (but considering given confiability in it self to change the probability of change)
		for i in range(self.numPos):
			for j in range(self.numVel):
				if((self.table[i,j] != -1) and (conf < random())):
					action = self.env.action_space.sample()
					self.table[i, j] = action


		




def Compare(t1, t2):
	if(t1.reward == t2.reward):
		if(t1.closest <= t2.closest):
			return t1
		else:
			return t2

	elif(t1.reward > t2.reward):
		return t1
	else:
		return t2

