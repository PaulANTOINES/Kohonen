#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Script permettant de lire les données contenues dans mnist.pkl.gz

Code implémenté en vue du projet Ensta IN104-Python 2014-2015

@author Thomas Hecht
@version 0.2.3
"""

#===============================================================================
# Importation des modules nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy
import pickle
import gzip

#===============================================================================
# Paramètres concernant les données
#===============================================================================
## chemin d'accès local au fichier mnist.pkl.gz
data_path = "/home/a/antoine/IN104/mnist.pkl.gz"
## taille des images en entrée
image_shape = (28,28)

#===============================================================================
# Chargement des données
#===============================================================================
file_handler = gzip.open(data_path, 'rb')
data = pickle.load(file_handler, encoding='latin1')
file_handler.close()
    
## répartition des sous-ensembles (de type 'tuple')
training_set, labelling_set, testing_set = data
## répartition des sous-sous-ensembles (de type 'numpy.ndarray')
training_samples, training_labels = training_set
labelling_samples, labelling_labels = labelling_set
testing_samples, testing_labels = testing_set


if __name__ == '__main__':     
#===============================================================================
# Affichage de statistiques liées aux données 
#=============================================================================== 
## création de liste de chaînes de caractères pour la lisibilité
    subset_names = ['training set', 'labelling set', 'testing set']

# Nombre d'exemples disponibles par sous-ensemble 
    print("Nombre d'exemples dans ",subset_names[0]," : ",training_labels.shape[0])
    print("Nombre d'exemples dans ",subset_names[1]," : ",labelling_labels.shape[0])
    print("Nombre d'exemples dans ",subset_names[2]," : ",testing_labels.shape[0])

# Format des données
    print("Format des exemples d'apprentissage :", training_samples.shape)
    print("Format des exemples de labellisation :", labelling_samples.shape)
    print("Format des exemples de test :", testing_samples.shape)

# Nombre de classes différentes
    nb_class = 10
# Répartition des valeurs de classe (équirépartition ?)
    repartition = [training_labels[training_labels == k].size for k in range(nb_class)]
    print(repartition)
# Affichage de l'image, moyenne/médiane + écart-type par classe d'entrainement
    plt.imshow(numpy.reshape(training_samples[numpy.random.randint      (0,training_labels.shape[0])], image_shape), interpolation = "nearest", cmap = 'Greys')
    plt.show()


    print("Moyenne de labelling_set : ", numpy.mean(labelling_set[1]))
    print("Médiane de labelling_set : ",numpy.median(labelling_set[1]))
    print("Ecart-type de labelling_set : ",numpy.std(labelling_set[1]))
    print("Moyenne de training_set : ", numpy.mean(training_set[1]))
    print("Médiane de training_set : ",numpy.median(training_set[1]))
    print("Ecart-type de training_set : ",numpy.std(training_set[1]))
    print("Moyenne de testing_set : ", numpy.mean(testing_set[1]))
    print("Ecart-type de testing_set : ",numpy.std(testing_set[1]))
    print("Médiane de testing_set : ",numpy.median(testing_set[1]))

