import gym
import numpy as np
from Table import Table, Compare, FindNearest

POSNUM = 12
VELNUM = 12
class test():
	"""docstring for test"""
	def __init__(self, arg):
		self.a = arg
		pass

		

env = gym.make('MountainCar-v0')

#Create optimal table
qStar = Table(env,POSNUM,VELNUM)
qStar.Run()

#Create tables and update optimal one
for _ in range(1500):
	t = Table(env,POSNUM,VELNUM)
	t.Run(exploreChance = 0.1)
	#t.Run(exploreChance = 0.1)
	#t.Run(exploreChance = 0.1)
	#t.reward /= 3
	#print(t.reward)
	qStar = Compare(qStar, t)
	#if(qStar.Merge(t) == 2):
	#	pass

		#print("Atualizei best")
		#best = t
	#print(qStar.table)

#print("Mutating now")
#for _ in range(20):
#	qStar.Mutate(0.5)
#	print(qStar.table)
'''
'''
#Current values for q Star
#print("qStar reward: ", qStar.reward)
#print("qStar closest: ", qStar.closest)
#print(qStar.table)

#print("Best:\n", best.table)
#print("\n\nqStar now:\n", qStar.table)

#print(best.table == qStar.table)

score = 0
#Run q Star a few times
print("Running tests")
for i in range(10):
	print("Test ", i)
	qStar.reward = 0
	qStar.Run(show = True, exploreChance = 0)
	#print("qStar now:\n", qStar.table)
	if(qStar.reward != -200):
		score += 1

print("Score: ", score)
''''''
#New values for q Star
#print("qStar reward: ", qStar.reward)
#print("qStar closest: ", qStar.closest)
#print(qStar.table)


env.close()

