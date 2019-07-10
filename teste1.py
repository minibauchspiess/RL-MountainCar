import gym
import numpy as np
from Table import Table, FindNearest
from Test import TestTable, Compare
import Train
import time

MATSIZE = 15

env = gym.make('MountainCar-v0')


qStar = Train.ManyIndepFixed(env, sizePos=MATSIZE, sizeVel=MATSIZE);

env.close()


