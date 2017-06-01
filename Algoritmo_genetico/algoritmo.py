import numpy as np

class Genetic:


	def __init__(self, objetivo, poblacionGrupal):
		self.listaFitness = []
		self.objetivo = objetivo
		self.poblacionGrupal = poblacionGrupal

	def inicializar(self, poblacionInicial):
		# Genomas iniciales		
		self.listaGenomas = np.random.randint(2, size=( poblacionInicial, len(self.objetivo) ))

		# Inicializa fitness para cada genoma
		for genoma in self.listaGenomas:
			fitness = self.getFitness(genoma)
			self.listaFitness.append(fitness)

	def getFitness(self, genoma):
		diferencia = np.absolute(genoma - self.objetivo)
		fitness = -np.sum(diferencia)
		return fitness

	def seleccion(self):
		cantidadEliminar = len(self.listaGenomas) - self.poblacionGrupal
		print self.listaFitness
		print self.listaGenomas
		for x in xrange(0,cantidadEliminar):
			index = np.argmin(self.listaFitness)
			self.listaFitness.pop(index)
			#self.listaGenomas = np.delete(self.listaGenomas[ine], index)
			self.listaGenomas = np.delete(self.listaGenomas, self.listaGenomas[index] )
			pass
		print self.listaFitness
		print self.listaGenomas



genetic = Genetic( np.array( [1, 1, 0, 1, 0, 0, 1, 0] ) , 4)
genetic.inicializar(5)
genetic.seleccion()
#genetic.getFitness()