#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import time
import numpy as np

class Genetic:

	found = False

	def __init__(self, objetivo, poblacionGrupal):
		self.listaFitness = []
		self.objetivo = objetivo
		self.poblacionGrupal = poblacionGrupal

	def inicializar(self, poblacionInicial):
		# Genomas iniciales		
		self.listaGenomas = np.random.randint(2, size=( poblacionInicial, len(self.objetivo) ))

		# Inicializa fitness para cada genoma
		self.calcularFitness()


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

	def seleccion(self):
		cantidadEliminar = len(self.listaGenomas) - self.poblacionGrupal
		for x in xrange(0,cantidadEliminar):
			index = np.argmin(self.listaFitness)
			self.listaFitness.pop(index)
			self.listaGenomas = np.delete(self.listaGenomas, index, 0 )
			pass
		
		indexMayor = np.argmax(self.listaFitness)
		
		parejas = []
		for genoma in self.listaGenomas:

			if np.array_equal(genoma, self.listaGenomas[indexMayor]) == False:
				parejas.append([self.listaGenomas[indexMayor], genoma])

		return parejas


	def corssover(self, parejas):
		limite = random.randint(1,(len(self.objetivo) -1))
		genomas = []
		for pareja in parejas:
			primero = pareja[0]
			segundo = pareja[1]

			izquierdaPrimero = primero[0:limite]
			derechaPrimero = primero[limite:]

			izquierdaSegundo = segundo[0:limite]
			derechaSegundo = segundo[limite:]

			primero = np.concatenate([izquierdaSegundo, derechaPrimero])
			segundo = np.concatenate([izquierdaPrimero, derechaSegundo])

			genomas.append(primero)
			genomas.append(segundo)

		self.listaGenomas =  np.array(genomas)
		self.calcularFitness()

	def mutar(self):
		probabilidadMutacion = random.randint( 0,( len( self.listaGenomas )-1 ) )

		for x in xrange(0,probabilidadMutacion):
			genoma = self.listaGenomas[x]
			indice = random.randint( 0, ( len( genoma )-1 ) )

			bit = genoma[indice]

			if bit == 1:
				genoma[indice] = 0				
			else:
				genoma[indice] = 1



start_time = time.time()

genetic = Genetic( np.array( [1, 1, 0, 1, 0, 0, 1, 0] ) , 4)
genetic.inicializar(5)

while genetic.found == False:
	genetic.calcularFitness()
	parejas = genetic.seleccion()
	genetic.corssover(parejas)
	genetic.mutar()

print("--- %s seconds ---" % (time.time() - start_time))
print "Solucion "
print genetic.solucion
#genetic.getFitness()