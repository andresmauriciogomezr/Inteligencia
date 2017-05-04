#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
# como ocurre la propagacion inversa por medio del gradiente descendiente? Revisar el link
# Sugerencia: primero leer y entender la parte de la definición general (if __name__ == "__main__":) y despues entrar en el modelo del perceprtron (class NeuralNetwork():)
# -*- coding: utf-8 -*-

from numpy import exp, array, random, dot
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():

    def __init__(self):
        # Inicializar la semilla del generador aleatorio para que siempre de los mismos numeros
        # cada vez que el programa corra
        random.seed(1)

        # Modelo de una sola neurona, con una conexion de salida y tres de entrada.
        # asignamos pesos aleatorios a una matriz 3 x 1, con los valores en el rango -1 a 1
        # y media 0.
        self.synaptic_weights = 2 * random.random((35, 1)) - 1

    # La función sigmoidea, que describe una función en forma de s, es la función de activación.
    # Nosotros hacemos pasar la suma de los pesos a través de dicha función
    # para normalizarla entre 0 y 1 (porque eso es lo que esperamos) y así
    # poder dar un resultado
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # La derivada de la función sigmoidea
    # es el gradiente descendiente de la función sigmoidea
    # Indica qué tanto "le creemos" a los pesos resultantes, revisar link.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Entrenamos a la red neuronal a través de un proceso de prueba y error
    # Ajustamos los pesos sinápticos en cada iteración
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pasamos el conjunto de entrenamiento a través de la red neuronal
            # (una única neurona).
            output = self.think(training_set_inputs)

            # Calculamos el error (La diferencia entre el valor que esperamos obtener realmente
            # y la salida predicha).
            error = training_set_outputs - output

            # Multiplique el error por la entrada y de nuevo por el gradiente descendiente de la función sigmoidea.
            # Esto significa que los pesos menos confiables se ajustan más (filtrado)
            # Esto significa que las entradas, que son cero, no causan cambio a
            # los pesos.
            adjustment = dot(training_set_inputs.T, error *
                             self.__sigmoid_derivative(output))

            # Ajustar los pesos.
            self.synaptic_weights += adjustment

    # Proceso de aprendizaje de la red neuronal:
    def think(self, inputs):
        # Pasamos las entradas a través de la red neuronal (una única neurona).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))



def discrimine(value):
    if value >= 0.5:
        return 1
    else:
        return 0

def classify(results):
    dictionary = {
        "A" : [0, 0, 0, 0],
        "B" : [0, 0, 0, 1],
        "C" : [0, 0, 1, 0],
        "D" : [0, 0, 1, 1],
        "E" : [0, 1, 0, 0],
        "F" : [0, 1, 0, 1],
        "G" : [0, 1, 1, 0],
        "H" : [0, 1, 1, 1],
        "I" : [1, 0, 0, 0],
        "J" : [1, 0, 0, 1],
        "K" : [1, 0, 1, 0],
        "L" : [1, 0, 1, 1],
        "M" : [1, 1, 0, 0],
        "N" : [1, 1, 0, 1],
        "O" : [1, 1, 1, 0],
        "P" : [1, 1, 1, 1]
    }
    for name, array in dictionary.iteritems():
     #   print array
        if array == results:
            return name
    return "No encuentra coincidencia"

if __name__ == "__main__":

    # Inicialice una red neuronal de una sola neurona. Quizas no sea
    # propiamente una red...
    neural_network_1 = NeuralNetwork()
    neural_network_2 = NeuralNetwork()
    neural_network_3 = NeuralNetwork()
    neural_network_4 = NeuralNetwork()

    #print("Pesos sinapticos iniciales generados aleatoriamente: ")
    #print(neural_network_1.synaptic_weights," pesos para 1")
    #print(neural_network_2.synaptic_weights," pesos para 2")
    #print(neural_network_3.synaptic_weights," pesos para 3")
    #print(neural_network_4.synaptic_weights," pesos para 4")

    # El conjunto de entrenamiento. Tenemos 4 ejemplos, cada uno consistente
    # de tres valores de entrada con su respectiva salida (una salida)
    training_set_inputs = array([[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
     [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
     [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
     [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
     [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
     [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
     [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
     [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
     [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    training_set_outputs_1 = array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    training_set_outputs_2 = array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    training_set_outputs_3 = array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]).T
    training_set_outputs_4 = array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]).T

    # Entrene a la red neuronal usando un conjunto de entrenamiento.
    # lo iteramos 10,000 veces, haciendo pequeños ajustes de pesos en cada
    # iteración
    neural_network_1.train(training_set_inputs, training_set_outputs_1, 7000)
    neural_network_2.train(training_set_inputs, training_set_outputs_2, 7000)
    neural_network_3.train(training_set_inputs, training_set_outputs_3, 7000)
    neural_network_4.train(training_set_inputs, training_set_outputs_4, 7000)

    print("Nuevos pesos sinapticos después del entremaniento: ")
    print(neural_network_1.synaptic_weights," pesos para 1")
    print(neural_network_2.synaptic_weights," pesos para 2")
    print(neural_network_3.synaptic_weights," pesos para 3")
    print(neural_network_4.synaptic_weights," pesos para 4")
   

    prueba =     np.array([
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]
                    ])
    prueba = np.reshape(prueba, 35)



    result = []
    # Pruebe la red neuronal con una situacion desconocida.
    print("Considerando las entradas  -> ?: ")
    perceptron1 = neural_network_1.think(prueba)
    perceptron2 = neural_network_2.think(prueba)
    perceptron3 = neural_network_3.think(prueba)
    perceptron4 = neural_network_4.think(prueba)

    result.append(discrimine(perceptron1))
    result.append(discrimine(perceptron2))
    result.append(discrimine(perceptron3))
    result.append(discrimine(perceptron4))

    print "Resultado de la predicción : " + str(classify(result))

    plt.imshow(np.reshape(prueba, (7,5)), 'gray')
    plt.show()
