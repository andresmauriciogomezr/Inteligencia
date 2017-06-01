#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import time
import numpy as np



class Force():
	"""docstring for Force"""

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
		for x in xrange(0,255):
			vector.append(bin(x)[2:] )
			pass

		self.listaGenomas = np.array(vector)
		print self.listaGenomas

force = Force()
force.generateNumbers()
force.calcularFitness()
