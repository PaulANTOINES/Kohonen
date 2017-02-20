#===============================================================================
# Importation des modules nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy
from read_mnist import testing_samples, testing_labels
import kohonen

W_labels = numpy.load("labels.npy")
data = numpy.load("data.npy")

Rendu = 0
Nombre_total = testing_labels.shape[0]
Nombre_exclus = 0
Raté = 0
'''
for j in range(numpy.shape(W_labels)[0]):
    if W_labels[j] == -1:
        data[j] = numpy.zeros((1,784))
'''

for itt in range(testing_labels.shape[0]):
    sample = testing_samples[itt]
    bmu_idx, bmu_score = kohonen.nearestVector(sample, data)        
    label_predicted = W_labels[bmu_idx]
    real_label = testing_labels[itt]
    print("Itération :",itt,"/",testing_labels.shape[0])
    
    if W_labels[bmu_idx]!=-1:# and bmu_score<7.5:
        if real_label == label_predicted:
            Rendu = Rendu + 1
            
        else : Raté = Raté + 1
    else:
        Nombre_exclus = Nombre_exclus + 1
    
        
        
print("Le rendu de ce programme est de : ", (Rendu/(Nombre_total-Nombre_exclus))*100, "%")
print("Nombre d'exclus :",(Nombre_exclus/Nombre_total)*100, "%")
print(Raté)


#Résultat obtenu à l'issue de la séance du 05/04 : 73.35%



    



