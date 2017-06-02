#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import time
import numpy as np



class Force():
	"""docstring for Force"""

	def __init__(self, objetivo):
		self.objetivo = objetivo


	def calcularFitness(self):
		self.listaFitness = []
		for genoma in self.listaGenomas:
			fitness = self.getFitness(genoma)			
			if fitness == 0:
				self.found = True
				self.solucion = genoma
				pass
			self.listaFitness.append(fitness)		

	def getFitness(self, genoma):
		diferencia = np.absolute(genoma - self.objetivo)
		fitness = -np.sum(diferencia)
		return fitness

	def generateNumbers(self):
		vector = []
		for x in xrange(0,256):
			binario = bin(x)[2:].zfill(8)
			binario = binario.replace("",",")[1:-1];
			aux = map(int, binario.split(","))
			vector.append(np.array(aux))
			pass
		self.listaGenomas = np.array(vector)

	def findSolution(self):
		solution = []
		for x in xrange(0,256):
			if self.listaFitness[x] == 0:
				solution = self.listaGenomas[x]
				pass
			pass
		return solution

start_time = time.time()
force = Force([1,1,1,1,1,1,1,1])
force.generateNumbers()
force.calcularFitness()

print("--- %s seconds ---" % (time.time() - start_time))
print("solucion")
print(force.findSolution())
