#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Ensemble des fonctions essentielles à l'implémentation en Python de l'algorithme de Kohonen en utilisant NumPy

Code implémenté en vue du projet Ensta IN104-Python 2014-2015

@author Thomas Hecht
@version 0.2.3
"""

#===============================================================================
# Importation des modules nécessaires
#===============================================================================
import numpy
import matplotlib.pyplot as plt
#===============================================================================
# Déclaration des fonctions
#===============================================================================

def constrainedExponentialDecay(curr_iter, start_iter, stop_iter, max_value, min_value):
    """Renvoie la valeur d'une décroissance exponentielle entre bornes : avant start_iter, max_value ; après stop_iter, min_value ; entre les deux, on passe exponentiellement de max_value à min_value

    @param curr_iter (entier) itération courante
    @param start_iter (entier) itération de démarrage de la décroissance exponentielle
    @param stop_iter (entier) itération de fin de la décroissance exponentielle   
    @param max_value (flottant) valeur du plateau initial    
    @param min_value (flottant) valeur du plateau final    
    @return (flottant) valeur de la décroissance exponentielle bornée
    """
    ##Calcul du paramètre de décroissance
    par_lambda = numpy.log(max_value/min_value)/(stop_iter-start_iter)
    ##valeur de la décroissance
    decay = max_value * numpy.exp((-1)*par_lambda*curr_iter)    
    return decay    

def nearestVector(input_vector, vectors):
    """Renvoie le vecteur nearest_vector (issu de vectors) établi comme "le plus proche" du vecteur input_vector au sens de la distance euclidienne

    @param input_vector (numpy.ndarray) vecteur d'entrée unique de dimension n
    @param vectors (numpy.ndarray) vecteur de vecteurs de dimension m * n
    @return (entier, flottant) indice de la BMU, score de la BMU
    """
    ##Vecteur des distances par rapport au vecteur d'entrée
    dist = numpy.linalg.norm(vectors-input_vector, axis = 1)
    ##Calcul de la BMU et de son score
    ind_min = numpy.argmin(dist)
    mini = numpy.ndarray.min(dist)
    
    return ind_min,mini

def twoDimensionGaussian(space_shape, gaussian_position, gaussian_sigma):
    """Renvoie un noyau gaussien à une certain position sur une grille en 2D avec une variance de gaussian_variance de maximum valant 1.0.

    @param space_shape (tuple) taille de l'espace 2D au format (m, n)
    @param gaussian_position (tuple) position du pic de la gaussienne au format (x,y)
    @param gaussian_sigma (entier) écart-type de la gaussienne, en termes de l'espace 2D (c-à-d en pixels si l'on assimile espace 2D à une image)
    @see http://fr.wikipedia.org/wiki/Fonction_gaussienne
    @see http://docs.scipy.org/doc/numpy/reference/generated/numpy.ogrid.html
    @return (numpy.ndarray) vecteur de vecteurs représentant une 'bulle' gaussienne (à aplatir avant de renvoyer)
    """
    kernel = numpy.zeros(space_shape)
    for i in range(space_shape[0]):
        for j in range(space_shape[1]):
            kernel[i,j] = numpy.exp(- ((i - gaussian_position[0])**2 + (j - gaussian_position[1])**2) / (2 * gaussian_sigma**2))
    
    ###En travaux. Tentative d'obtenir le même résultat avec numpy
##    x = numpy.array([i for i in range(space_shape[0])])
##    x = numpy.exp(-1/(2*gaussian_sigma**2) * numpy.square(x-gaussian_position[0]))
##    y = numpy.array([j for j in range(space_shape[1])])
##    y = numpy.exp(-1/(2*gaussian_sigma**2) * numpy.square(x-gaussian_position[1]))
    
##    XX, YY = numpy.meshgrid(x,y)
##    kernel = XX*YY  
    
    return kernel.flatten()
    
    

def updateKohonenWeights(input_vector, weights, learning_rate, neighborhood):
    """Renvoie les prototypes mis à jour d'après l'algorithme de Kohonen. - Fonction à effets de bord

    @param input_vector (numpy.ndarray) vecteur d'entrée unique de dimension n
    @param weights (numpy.ndarray) vecteur des poids de dimension m * n
    @param learning_rate (flottant) taux d'apprentissage [eta]
    @param neighborhood (numpy.ndarray) gaussienne sur un espace 2D ('aplatie' à m * n) intégrant implicitement la fonction de voisinage
    @see twoDimensionGaussian()
    @see nearestVector()
    @return None
    """
    trash = numpy.copy(weights)
    
    #print(neighborhood.shape)
    
    for i in range(len(weights)):
        weights[i] = weights[i] + learning_rate * neighborhood[i] * (input_vector - weights[i])
    
    return None
    
    
    
#output = twoDimensionGaussian((280,280), (140,140), 30)
#x = numpy.linspace(0,280,280)
#plt.plot(x,output[:, 140])
#plt.show()

