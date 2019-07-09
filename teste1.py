import gym
import numpy as np
from Table import Table, Compare

POSNUM = 8
VELNUM = 8


env = gym.make('MountainCar-v0')

#Create optimal table
qStar = Table(env,POSNUM,VELNUM)
qStar.Run(True, False)

best = qStar

#Create tables and update optimal one
for _ in range(500):
	t = Table(env,POSNUM,VELNUM)
	t.Run(True, False)
	best = Compare(qStar, t)
	if(qStar.Merge(t) == 2):
		pass

		#print("Atualizei best")
		#best = t
	#print(qStar.table)

#Current values for q Star
#print("qStar reward: ", qStar.reward)
#print("qStar closest: ", qStar.closest)
#print(qStar.table)

print("Best:\n", best.table)
print("\n\nqStar now:\n", qStar.table)

print(best.table == qStar.table)
'''
score = 0
#Run q Star a few times
print("Running tests")
for i in range(10):
	print("Test ", i)
	qStar.reward = 0
	qStar.Run(True, True)
	#print("qStar now:\n", qStar.table)
	if(qStar.reward != -200):
		score += 1

print("Score: ", score)
'''
#New values for q Star
#print("qStar reward: ", qStar.reward)
#print("qStar closest: ", qStar.closest)
#print(qStar.table)


env.close()

