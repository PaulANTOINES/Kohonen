import matplotlib.pyplot as plt
import numpy
import kohonen
import time
from read_mnist import training_samples, training_labels

data = numpy.load("rendu_optimal.npy")

print(data)
print(numpy.mean(data))
print(numpy.var(data))


