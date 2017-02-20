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

#===============================================================================
# Paramètres généraux de la simulation
#===============================================================================
## nombre total d'itérations d'apprentissage
iterations = 10000
## affichage console d'information ou non
verbose = True

#===============================================================================
# Paramètres concernant les données d'apprentissage
#===============================================================================
## dimension d'un vecteur d'entrée : l'espace colorimétrique utilisé ici est RVB, soit des vecteurs de dimension 3
data_shape = (1, 3)
## nombre de couleurs différentes disponibles
data_number = 1000
## dimensions des données : 'data_number' couleurs de taille 'data_shape' générées aléatoirement
data_dimension = (data_number, numpy.prod(data_shape))
## génération des données
data = numpy.random.random(size=data_dimension)

#===============================================================================
# Paramètres concernant la carte auto-organisatrice et l'algorithme de Kohonen
#===============================================================================
## taille de la carte auto-organisatrice (COA)
map_shape = (20, 20)
## valeur constante du rayon de voisinage gaussien (sigma)
sigma = 2.
## valeur constante du taux d'apprentissage (eta)
eta = .1
## dimensions des prototypes de la COA : une carte de MxM' [map_shape] vecteurs de dimension PxP' [data_dimension]
weights_dimension = (numpy.prod(map_shape), numpy.prod(data_shape))
## initialisation aléatoire des prototypes de la COA : distribution uniforme entre 0. et 1.
weights = numpy.random.uniform(low=0., high=1., size=weights_dimension)

#===============================================================================
# Boucle d'apprentissage suivant l'algorithme de Kohonen
#===============================================================================
for curr_iter in range(iterations):
    ## choisir un indice aléatoirement
    random_idx = numpy.random.randint(data_number)
    ## instancier l'exemple d'apprentissage courant
    sample = data[random_idx]
    ## trouver la best-matching unit (BMU) et son score (plus petite distance)
    bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
    ## traduire la position 1D de la BMU en position 2D dans la carte
    bmu_2D_idx = #?????--------------------------------------------------------- 
    ## gaussienne de taille sigma à la position 2D de la BMU
    gaussian_on_bmu = #?????---------------------------------------------------- 
    ## mettre à jour les prototypes d'après l'algorithme de Kohonen (fonction à effets de bord)
    kohonen.updateKohonenWeights(sample, weights, eta, gaussian_on_bmu)
    ## afficher l'itération courante à l'écran
    if verbose: 
        print('Iteration %d/%d'%(curr_iter+1, iterations))

#===============================================================================
# Affichage graphique
#===============================================================================
## paramètrage d'un graphique affichant les prototypes de couleurs appris par la COA
## création d'une nouvelle figure
weights_plot = plt.figure('Couleurs associées au prototypes de la carte')
## création d'un axe matplotlib
ax_weights = weights_plot.add_subplot(111)
## chargement dans la figure des prototypes comme matrice de pixels RVB
ax_weights.imshow(weights.reshape(map_shape[0], map_shape[1], numpy.prod(data_shape)), interpolation='nearest')
## affichage des graphiques
plt.show()
