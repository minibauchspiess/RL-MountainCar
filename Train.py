from Table import Table
from Test import TestTable, Compare

import gym
import numpy as np
from time import time

DEFAULT_SIZEPOS = 10
DEFAULT_SIZEVEL = 10

'''
Métodos de treinamento (vai adicionando aí):
- Cria várias tabelas independentes, sem atualizar estados já aprendidos, e seleciona a melhor pra validar
- Mesmo caso de cima, mas com uma chance x de escolher um valor aleatório em vez de consultar a tabela
- Cria várias tabelas, cada tabela sendo desenvolvida em um número de épocas e com chance x de escolher um valor aleatório em vez de consultar a tabela e escolhe a melhor
- Utiliza um dos métodos acima pra gerar tabelas, e escolhe um número n das melhores tabelas pra fazer um crossing entre si
- Utiliza um dos métodos citados pra criar tabelas, faz mutações nas melhores, valida o resultado entre elas e escolhe a melhor
- Utiliza um dos métodos citados pra criar tabelas, faz mutações nas melhores, valida os melhores resultados, pega esses melhores resultados, faz crossing e seleciona de novo o melhor resultado
'''



def ManyIndepFixed(env, numTables = 2000, sizePos = DEFAULT_SIZEPOS, sizeVel = DEFAULT_SIZEVEL, feedback = True):
	#Check starting time executing training and end time, to determine elapsed time during training
	startTime = time()

	#Create the specified number of tables, with no chance of changing it's already set values to a new random number, and select the table with the best result
	bestTable = Table(env, sizePos, sizeVel)
	bestTable.Run(exploreChance=0)
	TestTable(bestTable, 5)

	for i in range(numTables-1):
		t = Table(env, sizePos, sizeVel)
		t.Run(exploreChance=0)

		#Test table, getting it's average reward
		TestTable(t, 5)

		bestTable = Compare(bestTable, t)

		if((feedback) and ((i%100) == 0)):
			print("Done with ", i, " epochs")

	#Get values to validade created table
	timesWon = TestTable(bestTable)
	bestTable.Run(exploreChance=0, show=True)

	elapsedTime = time() - startTime

	print("Table created through function ManyIndepFixed, after {:.2f}".format(elapsedTime), " seconds")
	print("Average reward: ", bestTable.avgReward, "\nTimes completed, out of 1000: ", timesWon)


	return bestTable


