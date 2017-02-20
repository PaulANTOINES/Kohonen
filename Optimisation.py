import matplotlib.pyplot as plt
import numpy
import kohonen
from read_mnist import training_samples, training_labels, labelling_samples, labelling_labels, testing_samples, testing_labels

def COA(iterations, eta, sigma,y,x):
    

#===============================================================================
# Paramètres généraux de la simulation
#===============================================================================
## nombre total d'itérations d'apprentissage
    #iterations = 1000 #training_samples.shape[0]
## affichage console d'information ou non
    verbose = False

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
    map_shape = (20,20)
## valeur constante du rayon de voisinage gaussien (sigma)
    #sigma = 2
## valeur constante du taux d'apprentissage (eta)
    #eta = 1. #METTRE 10^-3
## dimensions des prototypes de la COA : une carte de MxM' [map_shape] vecteurs de dimension PxP' [data_dimension]
    weights_dimension = (numpy.prod(map_shape), numpy.prod(data_shape))
## initialisation aléatoire des prototypes de la COA : distribution uniforme entre 0. et 1.
    weights = numpy.random.uniform(low=y, high=x, size=weights_dimension)

    decay_start_iter = 0.2*iterations
    decay_stop_iter = 0.6*iterations

## paramètres du rayon de voisinage gaussien (sigma)
    sigma_max_value = 4.
    sigma_min_value = .9
## paramètres du taux d'apprentissage (eta)
    eta_max_value = .3
    eta_min_value = .001

#===============================================================================
# Boucle d'apprentissage suivant l'algorithme de Kohonen
#===============================================================================
    eta_var = eta
    sigma_var = sigma
    sum_bmu = 0
    for curr_iter in range(iterations):
    ## choisir un indice aléatoirement
        random_idx = numpy.random.randint(training_samples.shape[0])
    ## instancier l'exemple d'apprentissage courant
        sample = data[random_idx]
        ## trouver la best-matching unit (BMU) et son score (plus petite distance)
        bmu_idx, bmu_score = kohonen.nearestVector(sample, weights)
        sum_bmu += bmu_score
        ## traduire la position 1D de la BMU en position 2D dans la carte
        bmu_2D_idx = (bmu_idx//map_shape[0], bmu_idx%map_shape[0])
        ## gaussienne de taille sigma à la position 2D de la BMU
        gaussian_on_bmu = kohonen.twoDimensionGaussian(map_shape, bmu_2D_idx, sigma)
        ## mettre à jour les prototypes d'après l'algorithme de Kohonen (fonction à effets de bord)
        kohonen.updateKohonenWeights(sample, weights, eta, gaussian_on_bmu)
        sigma = kohonen.constrainedExponentialDecay(curr_iter, decay_start_iter, decay_stop_iter, sigma_max_value, sigma_min_value)
        eta = kohonen.constrainedExponentialDecay(curr_iter, decay_start_iter, decay_stop_iter, eta_max_value, eta_min_value)
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
    #plt.imshow(img, cmap = 'Greys')
    #plt.show()

        
    #===============================================================================
    # Sauvegarde des données
    #===============================================================================
    return weights,sum_bmu/iterations
    #numpy.save("data", weights)
    #mean_bmu = sum_bmu/iterations
    #numpy.save("mean_bmu", mean_bmu)


def label(data,mean_bmu):
    
    exclus = 0
    W_associated = [[0 for i in range(10)] for j in range(numpy.shape(data)[0])]

    for itt in range(labelling_labels.shape[0]):
        sample = labelling_samples[itt]
        bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
        if (bmu_score/mean_bmu) * 100 < 200: 
            W_associated[bmu_idx][labelling_labels[itt]] += 1
        
    W_labels = [0]*numpy.shape(data)[0]
    for i in range(numpy.shape(data)[0]):
        if numpy.sum(W_associated[i])!=0:
            W_labels[i] = numpy.argmax(numpy.array(W_associated[i]))
        else :
            W_labels[i] = -1 
            exclus += 1
       
    print("Nombre d'exclus : ", exclus)


    #numpy.save("labels", W_labels)
    return W_labels

def eval(W_labels,data,mean_bmu):
    
    exclus = 0
    Rendu = 0
    fail = 0

    for itt in range(testing_labels.shape[0]):
        sample = testing_samples[itt]
        bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
        label_predicted = W_labels[bmu_idx]
        real_label = testing_labels[itt]
        
        if W_labels[bmu_idx]!=-1 and bmu_score/mean_bmu < 2 : #and bmu_score<8.5:  Cette condition ne marche pas
            if real_label == label_predicted:
                Rendu = Rendu + 1
                
            else : fail += 1
        else:
            exclus = exclus + 1
            
            
    #print("Le rendu de ce programme est de : ", (Rendu/testing_labels.shape[0])*100, "%")
    #print("Nombre d'exclus :",(exclus/testing_labels.shape[0])*100, "%")
    
    return (Rendu/testing_labels.shape[0])*100,(exclus/testing_labels.shape[0])*100




###TEST###
#eta = 1
#sigma = 1
SIGMA = []
Rendu = []
EXCLUS = []
rend = 0
exc = 0
L=[1.*i for i in range(10)]
print(L)


for j in range(len(L)):
    y = L[j]
    x = y+1. 
    print(j)
    (data,mean_bmu) = COA(10000,1.2622,1.14,y,x)
    W_labels = label(data,mean_bmu)
    (rend,exc) = eval(W_labels,data,mean_bmu)
    SIGMA.append(y)
    EXCLUS.append(exc)
    Rendu.append(rend)
    numpy.save("sauvegarde_sigma", SIGMA)
    numpy.save("sauvegarde_exclus", EXCLUS)
    numpy.save("sauvegarde_rendu", Rendu)
    
plt.plot(SIGMA,Rendu)
plt.show()

print(SIGMA)
print(Rendu)
print(EXCLUS)
'''

    
