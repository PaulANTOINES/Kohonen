#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@brief Implémentation de l'algorithme de Kohonen et entraînement d'une Carte auto-organisatrice sur un exemple simple : regrouper des couleurs RGB aléatoires

Code implémenté en vue du projet Ensta IN104-Python 2014-2015

@author Thomas Hecht
@version 0.1.2
"""

#===============================================================================
# Importations nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy
import kohonen
import time
from read_mnist import training_samples, training_labels
x = time.clock()
#===============================================================================
# Paramètres généraux de la simulation
#===============================================================================
## nombre total d'itérations d'apprentissage
iterations = 1000#training_samples.shape[0]
## affichage console d'information ou non
verbose = True

#===============================================================================
# Paramètres concernant les données d'apprentissage
#===============================================================================
## dimension d'un vecteur d'entrée
data_shape = (1,784)
## nombre de couleurs différentes disponibles
data_number = 10
## dimensions des données : 'data_number' couleurs de taille 'data_shape' générées aléatoirement
data_dimension = (data_number, numpy.prod(data_shape))
## génération des données
data = training_samples

#===============================================================================
# Paramètres concernant la carte auto-organisatrice et l'algorithme de Kohonen
#===============================================================================
## taille de la carte auto-organisatrice (COA)
map_shape = (20, 20)
## valeur constante du rayon de voisinage gaussien (sigma)
sigma = 2.
## valeur constante du taux d'apprentissage (eta)
eta = 2.8 
## dimensions des prototypes de la COA : une carte de MxM' [map_shape] vecteurs de dimension PxP' [data_dimension]
weights_dimension = (numpy.prod(map_shape), numpy.prod(data_shape))
## initialisation aléatoire des prototypes de la COA : distribution uniforme entre 0. et 1.
weights = numpy.random.uniform(low=0., high=10., size=weights_dimension)

#===============================================================================
# Boucle d'apprentissage suivant l'algorithme de Kohonen
#===============================================================================
eta_var = eta
sigma_var = sigma
BMU = 0
for curr_iter in range(iterations):
    ## choisir un indice aléatoirement
    random_idx = numpy.random.randint(training_samples.shape[0])
    ## instancier l'exemple d'apprentissage courant
    sample = data[random_idx]
    ## trouver la best-matching unit (BMU) et son score (plus petite distance)
    bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
    BMU = BMU + bmu_score
    ## traduire la position 1D de la BMU en position 2D dans la carte
    bmu_2D_idx = (bmu_idx//map_shape[0], bmu_idx%map_shape[0])
    ## gaussienne de taille sigma à la position 2D de la BMU
    gaussian_on_bmu = kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma)
    ## mettre à jour les prototypes d'après l'algorithme de Kohonen (fonction à effets de bord)
    kohonen.updateKohonenWeights(sample, weights, eta, gaussian_on_bmu)
    #eta = kohonen.constrainedExponentialDecay(curr_iter, 0, iterations, eta, 0.01)
    #sigma = kohonen.constrainedExponentialDecay(curr_iter, 0, iterations, sigma, 0.01)
    ## afficher l'itération courante à l'écran
    if verbose: 
        print('Iteration %d/%d'%(curr_iter+1, iterations))
    
#===============================================================================
# Affichage graphique
#===============================================================================
weights_reshaped = weights.reshape((map_shape[0],map_shape[1],28,28))
img = numpy.zeros((map_shape[0]*28,map_shape[1]*28))
for i in range(map_shape[0]):
    for j in range(map_shape[1]):
        img[i*28:(i+1)*28,j*28:(j+1)*28] = weights_reshaped[i,j,:,:]
plt.imshow(img, cmap = 'Greys')
print(time.clock()-x)
print(BMU/iterations)
plt.show()

    
#===============================================================================
# Sauvegarde des données
#===============================================================================
numpy.save("data", weights)

