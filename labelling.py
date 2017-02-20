#===============================================================================
# Importation des modules nécessaires
#===============================================================================
import matplotlib.pyplot as plt
import numpy
from read_mnist import labelling_samples, labelling_labels
import kohonen

data = numpy.load("data.npy")

W_associated = [[0 for i in range(10)] for j in range(numpy.shape(data)[0])]

for itt in range(labelling_labels.shape[0]):
    sample = labelling_samples[itt]
    bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
    W_associated[bmu_idx][labelling_labels[itt]] += 1
    print("Itération :",itt,"/",labelling_labels.shape[0])
    
W_labels = [0]*numpy.shape(data)[0]
for i in range(numpy.shape(data)[0]):
    ##Solution 1
    '''
    if W_associated[i][numpy.argmax(numpy.array(W_associated[i]))]>0.3*numpy.sum(W_associated[i]) and numpy.sum(W_associated[i])!=0:
        W_labels[i] = numpy.argmax(numpy.array(W_associated[i]))
    else :
        W_labels[i] = -1
    
    ##Solution 2
    '''
    '''
    if W_associated[i][numpy.argmax(numpy.array(W_associated[i]))]<0.3*numpy.sum(W_associated[i]):
        data[i] = numpy.zeros((1,784))
        bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
        W_labels[i] = numpy.argmax(numpy.array(W_associated[bmu_idx]))
    else:
        W_labels[i] = numpy.argmax(numpy.array(W_associated[i]))
    '''    
    
    
    W_labels[i] = numpy.argmax(numpy.array(W_associated[i]))
    print(W_associated[i][numpy.argmax(numpy.array(W_associated[i]))])
    
print(W_labels)


numpy.save("labels", W_labels)

