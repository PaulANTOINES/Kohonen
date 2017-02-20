###Programme pour l'affichage d'une interface graphique 

'''
    Programme permettant d'afficher une interface graphique interactive permettant la réalisation de CAO grâce à l'algorithme de Kohonen, son labelling ainsi que son évaluation, tout en faisant varier les paramètres d'entrée.
    
    Entrée : le fichier lu par la fonction read_mnist
    
    Sortie : aucune à part l'interface graphique, et la création de fichiers npy pour les paramètres
    
    Pour plus d'informations sur la programmation de l'interface : 
    http://apprendre-python.com/page-tkinter-interface-graphique-python-tutoriel
    
'''

###Importations :

from tkinter import * 

import matplotlib.pyplot as plt
import numpy
import kohonen
from read_mnist import training_samples, training_labels, labelling_samples, labelling_labels, testing_samples, testing_labels
import os

###Programmes liés à l'appui des différents boutons :

## Tout au long de ces programmes, on utilisera des boucles de sécurité du type (try,except) au lieu de tests assert, afin de ne pas perdre l'intérêt de l'interface graphique que l'utilisateur est censé utiliser. De même, on ne pourra pas effectuer de test lorsque les fonctions sont appelés, car elles ne contiennent aucun argument d'entrée (pour pouvoir être utilisés sur les boutons). On pourra se référer aux asserts et aux tests effectués dans les fonctions réelles CAO, label et évaluation.

##Puisque les fonctions associées à des boutons sur Tkinter ne peuvent prendre aucun argument en entrée, nous sommes forcés de réécrire les fonctions principales, sous une forme différente pour prendre en compte les choix de l'utilisateur

###Fonctions principales : 

def graph():
    
    ''' 
    Fonction qui permet de réaliser l'affichage graphique d'une CAO déjà réalisée, et mise en mémoire
    
        Entrée : pas d'entrée, car fonction sur bouton, on utilise donc des sauvegardes, ici celle des poids, et de la taille de la               carte
        Sortie : affichage graphique de la carte grâce à matplotlib
    '''
    
    try:#Boucle de sécurité : si la CAO n'est pas en mémoire, on ne fait rien.
    
        weights = numpy.load("data.npy")
        map_shape = numpy.load("map_shape.npy")
        
        #On transforme les données pour les faire correspondre à une matrice de même taille que map_shape.
        
        weights_reshaped = weights.reshape((map_shape[0],map_shape[1],28,28))
        img = numpy.zeros((map_shape[0]*28,map_shape[1]*28))
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                img[i*28:(i+1)*28,j*28:(j+1)*28] = weights_reshaped[i,j,:,:]
        
        plt.imshow(img, cmap = 'Greys')
        plt.show()
        
        return "Done"
    except:
        #On affiche le message d'erreur sur l'interface graphique.
        
        affichageCAO['text'] = "Veuillez d'abord faire la CAO"
        
    
def allfunction():
    
    '''
    
    Fonction qui va effectuer tous les calculs d'un coup lors de l'appui du bouton correspondant : CAO, labelling et évaluation. Se référer aux fonctions ainsi nommées, placées plus bas, pour plus de précision.
    
        Entrée : pas d'entrée, car fonction sur bouton, on utilise donc des sauvegardes, ici les paramètres rentrés par l'utilisateur (sauf map_shape) ou s'il n'en a pas rentré, les paramètres de base (voir plus bas)
        
        Sortie : pas de sortie, le programme sauvegarde uniquement les données de la CAO, ainsi que le résultat de l'évaluation
        
    '''
    try:#Boucle de sécurité : si le fichier rentré dans le programme n'a pas le bon format ou les bonnes composantes, la fonction ne va rien faire et afficher un message d'erreur sur l'interface graphique.
        
        #On charge les données :
        
        iterations = numpy.load("iterations.npy")
        iterations = int(iterations.tolist())#Permet de transformer le format des données, car elles sont sauvegardées sous format numpy.
        
        sigma = numpy.load("sigma.npy")
        sigma = float(sigma.tolist())
        
        eta = numpy.load("eta.npy")
        eta = float(eta.tolist())
        
        sigma_max = numpy.load("sigma_max.npy")
        sigma_max_value = float(sigma_max.tolist())
        
        sigma_min = numpy.load("sigma_min.npy")
        sigma_min_value = float(sigma_min.tolist())
        
        eta_max = numpy.load("eta_max.npy")
        eta_max_value = float(eta_max.tolist())
        
        eta_min = numpy.load("eta_min.npy")
        eta_min_value = float(eta_min.tolist())
        
    #===============================================================================
    # Paramètres concernant les données d'apprentissage
    #===============================================================================
    ## dimension d'un vecteur d'entrée
        data_shape = (1,784)
    ## nombre de classes différentes disponibles
        data_number = 10
    ## dimensions des données 
        data_dimension = (data_number, numpy.prod(data_shape))
    ## récupération des données
        data = training_samples
    
    #===============================================================================
    # Paramètres concernant la carte auto-organisatrice et l'algorithme de Kohonen
    #===============================================================================
    ## taille de la carte auto-organisatrice (COA)
        map_shape = (20,20)
        numpy.save("map_shape", map_shape)  #Sauvegardé car utilisé dans d'autres fonctions.
    ## dimensions des prototypes de la COA : une carte de MxM' [map_shape] vecteurs de dimension PxP' [data_dimension]
        weights_dimension = (numpy.prod(map_shape), numpy.prod(data_shape))
    ## initialisation aléatoire des prototypes de la COA : distribution uniforme entre 0. et 1.
        weights = numpy.random.uniform(low=0., high=1., size=weights_dimension)
    
        decay_start_iter = 0.2*iterations
        decay_stop_iter = 0.6*iterations
    
    ###Kohonen :
    
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
        
        #===============================================================================
        # Sauvegarde des données
        #===============================================================================
        numpy.save("data", weights) # On en a besoin dans d'autres fonctions (graph par exemple).
        mean_bmu = sum_bmu/iterations
        numpy.save("mean_bmu", mean_bmu) # De même si on veut refaire le labelling ou encore l'évaluation.
        
        ###Labelling :
        
        data = weights #changement de nom : ici, la donnée devient invariante
        
    
        W_associated = [[0 for i in range(10)] for j in range(numpy.shape(data)[0])]#Matrice qui va compter les différentes correspondances des données de labelling avec les poids de la CAO.
    
        for j in range(labelling_labels.shape[0]):
            sample = labelling_samples[j]
            bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
            
            if (bmu_score/mean_bmu) * 100 < 200: #Test supplémentaire destiné à ne pas compter les cas qui sont trop loin des neurones de la CAO, qui viendrait polluer le labelling.
            
                W_associated[bmu_idx][labelling_labels[j]] += 1
            
        W_labels = [0]*numpy.shape(data)[0]
        for i in range(numpy.shape(data)[0]):
            if numpy.sum(W_associated[i])!=0:
                W_labels[i] = numpy.argmax(numpy.array(W_associated[i]))
            else :#On ne pas mettre une étiquette sur les neuronnes qui n'ont pas été utilisés dans le labelling.
                W_labels[i] = -1 #Coefficient qui indique que l'on ne pourra pas utiliser ce neurone pour les évaluations.
            
        
    
        numpy.save("labels", W_labels)
        
        
        ###Évaluation :
        
        #Paramètres permettant de rendre compte de l'efficacité de la CAO analysée.
        
        exclus = 0
        rendu = 0
        fail = 0
    
        for j in range(testing_labels.shape[0]):
            sample = testing_samples[j]
            bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
            label_predicted = W_labels[bmu_idx]
            real_label = testing_labels[j]
            
            if W_labels[bmu_idx]!=-1 and bmu_score/mean_bmu < 2 : #Dernier critère supplémentaire : les données qui sont trop loin de tous les neurones utilisés seront rejetés.
                if real_label == label_predicted:
                    rendu = rendu + 1
                    
                else : fail += 1
            else:
                exclus = exclus + 1   
        
        numpy.save("results", (rendu/testing_labels.shape[0])*100) #On sauvegarde pour que l'utilisateur puisse la consulter même après avoir lancé une autre CAO.
        
    except:
        affichageAllfunction['text'] = "Erreur fichier d'entrée ou paramètres"
        
def coa():
    
    '''
    
    Fonction qui réalise une carte auto-organisatrice (CAO) à partir de l'algorithme de kohonen et de la base de donnée entrée par l'utilisateur à travers la fonction du fichier read_mnist. Voir le lien plus bas pour plus de précision.
    
    
    Entrées : (sous forme de sauvegardes)
    
        -les paramètres que l'utilisateur souhaite appliquer (iterations, sigma, eta, sigma_max, sigma_min, eta_max et eta_min
        
        
    Sorties : (sous forme de sauvegardes)
    
    -l'ensemble des poids des neurones générés par la fonction
    -mean_bmu : la moyenne de la valeur des bmu(best matching unit) calculés lors de la réalisation de la CAO
    
    
    Lien wikipédia : https://fr.wikipedia.org/wiki/Carte_auto_adaptative
    
    
    '''
    
    
    try:#Boucle de sécurité : si le fichier rentré dans le programme n'a pas le bon format ou les bonnes composantes, la fonction ne va rien faire et afficher un message d'erreur sur l'interface graphique.
        
        #On charge les données :
        
        iterations = numpy.load("iterations.npy")
        iterations = int(iterations.tolist())#Permet de transformer le format des données, car elles sont sauvegardées sous format numpy.
        
        sigma = numpy.load("sigma.npy")
        sigma = float(sigma.tolist())
        
        eta = numpy.load("eta.npy")
        eta = float(eta.tolist())
        
        sigma_max = numpy.load("sigma_max.npy")
        sigma_max_value = float(sigma_max.tolist())
        
        sigma_min = numpy.load("sigma_min.npy")
        sigma_min_value = float(sigma_min.tolist())
        
        eta_max = numpy.load("eta_max.npy")
        eta_max_value = float(eta_max.tolist())
        
        eta_min = numpy.load("eta_min.npy")
        eta_min_value = float(eta_min.tolist())
        
    #===============================================================================
    # Paramètres concernant les données d'apprentissage
    #===============================================================================
    ## dimension d'un vecteur d'entrée
        data_shape = (1,784)
    ## nombre de classes différentes disponibles
        data_number = 10
    ## dimensions des données 
        data_dimension = (data_number, numpy.prod(data_shape))
    ## récupération des données
        data = training_samples
    
    #===============================================================================
    # Paramètres concernant la carte auto-organisatrice et l'algorithme de Kohonen
    #===============================================================================
    ## taille de la carte auto-organisatrice (COA)
        map_shape = (20,20)
        numpy.save("map_shape", map_shape)  #Sauvegardé car utilisé dans d'autres fonctions
    ## dimensions des prototypes de la COA : une carte de MxM' [map_shape] vecteurs de dimension PxP' [data_dimension]
        weights_dimension = (numpy.prod(map_shape), numpy.prod(data_shape))
    ## initialisation aléatoire des prototypes de la COA : distribution uniforme entre 0. et 1.
        weights = numpy.random.uniform(low=0., high=1., size=weights_dimension)
    
        decay_start_iter = 0.2*iterations
        decay_stop_iter = 0.6*iterations
    
    ###Kohonen
    
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
        
        #===============================================================================
        # Sauvegarde des données
        #===============================================================================
        numpy.save("data", weights) # On en a besoin dans d'autres fonctions (graph par exemple).
        mean_bmu = sum_bmu/iterations
        numpy.save("mean_bmu", mean_bmu) # De même si on veut refaire le labelling ou encore l'évaluation.
        
    except:
        affichageCAO['text'] = "Veuillez rentrer tous les paramètres"
    

def label():
   
    '''
    Associe un label à chaque neurone de la carte. Le label est choisi comme étant la classe qui correspond le mieux au neurone. Si la correspondance n'est pas assez claire, le label du neurone est mis à -1 et le neurone est dit exclus.
    
    Entrées : (sous forme de sauvegardes)
    - data (numpy.ndarray) : matrice des poids de la carte créée
    - mean_bmu (entier): moyenne des scores des bmu lors de la création de la carte
    
    Sortie : (sous forme de sauvegardes)
    - W_labels (numpy.ndarray) : liste des labels correspondant aux neurones
    
    '''
    
    try:
        data = numpy.load("data.npy") 
        mean_bmu = numpy.load("mean_bmu.npy")
    
        W_associated = [[0 for i in range(10)] for j in range(numpy.shape(data)[0])]#Matrice qui va compter les différentes correspondances des données de labelling avec les poids de la CAO.
    
        for j in range(labelling_labels.shape[0]):
            sample = labelling_samples[j]
            bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
            
            if (bmu_score/mean_bmu) * 100 < 200: #Test supplémentaire destiné à ne pas compter les cas qui sont trop loin des neurones de la CAO, qui viendrait polluer le labelling.
            
                W_associated[bmu_idx][labelling_labels[j]] += 1
            
        W_labels = [0]*numpy.shape(data)[0]
        for i in range(numpy.shape(data)[0]):
            if numpy.sum(W_associated[i])!=0:
                W_labels[i] = numpy.argmax(numpy.array(W_associated[i]))
            else :#On ne pas mettre une étiquette sur les neuronnes qui n'ont pas été utilisés dans le labelling.
                W_labels[i] = -1 #Coefficient qui indique que l'on ne pourra pas utiliser ce neurone pour les évaluations.
            
        
    
        numpy.save("labels", W_labels)
    except:
        affichageLabel['text'] = "Veuillez d'abord faire la CAO"
        
      

def eval():
    
    '''
    Evalue la qualité de la carte en comparant, pour chaque exemple, la classe associé à l'exemple et le chiffre donné par la carte.
    
    Entrées : (sous forme de sauvegardes)
    - data (numpy.ndarray) : matrices des poids de la carte créée
    - mean_bmu (entier): moyenne des scores des bmu lors de la création de la carte
    - W_labels (numpy.ndarray) : liste des labels correspondant aux neurones
    
    Sortie : (sous forme de sauvegarde)
    - rendu (entier) : nombre de réponses correctes données par la carte
    
    
    '''
    
    try:
        W_labels = numpy.load("labels.npy")
        data = numpy.load("data.npy")
        mean_bmu = numpy.load("mean_bmu.npy")
        
        #Paramètres permettant de rendre compte de l'efficacité de la CAO analysée.
        
        exclus = 0
        rendu = 0
        fail = 0
    
        for j in range(testing_labels.shape[0]):
            sample = testing_samples[j]
            bmu_idx, bmu_score = kohonen.nearestVector(sample, data)
            label_predicted = W_labels[bmu_idx]
            real_label = testing_labels[j]
            
            if W_labels[bmu_idx]!=-1 and bmu_score/mean_bmu < 2 : #Dernier critère supplémentaire : les données qui sont trop loin de tous les neurones utilisés seront rejetés.
                if real_label == label_predicted:
                    rendu = rendu + 1
                    
                else : fail += 1
            else:
                exclus = exclus + 1   
        
        numpy.save("results", (rendu/testing_labels.shape[0])*100) #On sauvegarde pour que l'utilisateur puisse la consulter même après avoir lancé une autre CAO.
        
    except:
        affichageEval['text'] = "Veuillez d'abord faire le labelling"
        
    
def result():
    '''
    
    Fonction qui permet, lors de l'appui sur le bouton correspond, d'afficher le dernier résultat en mémoire de l'évaluation d'une CAO 
    
    Entrée : pas d'entrée
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    try:#Il faut qu'il y ait un résultat en mémoire.
        res = numpy.load("results.npy")
        affichageResult['text'] = res
    except:
        affichageResult['text'] = "Evaluer d'abord la CAO"

def erase():
    '''
    
    Fonction qui permet, lors de l'appui sur le bouton correspond, d'effacer les fichiers sauvegardés en mémoire sous format npy, contenant les paramètres
    
    Entrée : pas d'entrée
    
    Sortie : pas de sortie 
    
    '''
    
    try:
        os.remove("mean_bmu.npy")
    except:
        trtr=0#Ceci n'est pas une variable importante, elle est juste présente pour que Python comprenne que si le fichier n'existe pas, il n'a rien à faire.
    
    try:
        os.remove("data.npy")
    except:  
        trtr=0
        
    try:
        os.remove("sigma_max.npy")
    except:    
        trtr=0
        
    try:
        os.remove("sigma_min.npy")
    except:    
        trtr=0
        
    try:    
        os.remove("sigma.npy")
    except: 
        trtr=0
        
    try:    
        os.remove("eta_min.npy")
    except:
        trtr=0
        
    try:    
        os.remove("eta_max.npy")
    except: 
        trtr=0
        
    try:    
        os.remove("eta.npy")
    except:  
        trtr=0
        
    try:    
        os.remove("iterations.npy")
    except: 
        trtr=0
        
    try:    
        os.remove("results.npy")
    except:   
        trtr=0
        
    try:    
        os.remove("labels.npy")
    except: 
        trtr=0
    
###Fonctions permettant de sauvegarder les paramètres rentré par l'utilisateur (associées à un bouton chacune):


def iteration():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "itérations" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    iterations = reponseIteration.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "itérations" doit être un entier. 
        if int(iterations)>0:# "itérations" doit être positif.
            
            numpy.save("iterations", iterations)
            affichageIteration['text'] = iterations#Affichage graphique.
        else:
            affichageIteration['text'] = "Erreur : doit être un entier positif"
    except:
        affichageIteration['text'] = "Erreur : doit être un entier"
            

def sigmas():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "sigma" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    sigma = reponseSigma.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "sigma" doit être un réel. 
        if float(sigma)>0:# "sigma" doit être positif. 
            numpy.save("sigma", sigma)
            affichageSigma['text'] = sigma
        else:
            affichageSigma['text'] = "Erreur : doit être positif"
            
    except:
        affichageSigma['text'] = "Erreur : doit être un réel"

def etas():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "eta" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    eta = reponseEta.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "eta" doit être un réel. 
        if float(eta)>0:# "eta" doit être positif. 
            numpy.save("eta", eta)
            affichageEta['text'] = eta
        else:
            affichageEta['text'] = "Erreur : doit être positif"
    except:
        affichageEta['text'] = "Erreur : doit être un réel"
    
    
def etas_max():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "eta_max_value" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    eta_max = reponseEtaMax.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "eta_max_value" doit être un réel. 
    
        if float(eta_max)>0:# "eta_max_value" doit être positif.
        
            try:#Au cas où l'utilisateur à supprimé le fichier du paramètre eta_min_value.
                eta_min = numpy.load("eta_min.npy")
                eta_min_value = float(eta_min.tolist())
            except:
                dede=1
                
            if eta_min_value < float(eta_max):# "eta_max_value" doit être supérieur à "eta_min_value".
                numpy.save("eta_max", eta_max)
                affichageEtaMax['text'] = eta_max
            else:
                affichageEtaMax['text'] = "Doit être supérieur à eta_min"
        else:
            affichageEtaMax['text'] = "Erreur : doit être positif"
    except:    
        affichageEtaMax['text'] = "Erreur : doit être un réel"
    
    
def etas_min():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "eta_min_value" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    eta_min = reponseEtaMin.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "eta_min_value" doit être un réel. 
        if float(eta_min)>0:# "eta_min_value" doit être positif.
        
            try:#Au cas où l'utilisateur à supprimé le fichier du paramètre eta_max_value.
                eta_max = numpy.load("eta_max.npy")
                eta_max_value = float(eta_max.tolist())
            except:
                dede=1
                
            if float(eta_min) < eta_max_value:# "eta_min_value" doit être inférieur à "eta_max_value".
                numpy.save("eta_min", eta_min)
                affichageEtaMin['text'] = eta_min
            else:
                affichageEtaMin['text'] = "Doit être inférieur à eta_max"
        else:
            affichageEtaMin['text'] = "Erreur : doit être positif"
    except:
        affichageEtaMin['text'] = "Erreur : doit être un réel"
    
    
def sigmas_max():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "sigma_max_value" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    sigma_max = reponseSigmaMax.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "sigma_max_value" doit être un réel. 
        if float(sigma_max)>0:# "sigma_max_value" doit être positif.
        
            try:
                sigma_min = numpy.load("sigma_min.npy")#Au cas où l'utilisateur à supprimé le fichier du paramètre sigma_min_value.
                sigma_min_value = float(sigma_min.tolist())
            except:
                dede=1
                
            if sigma_min_value < float(sigma_max):# "sigma_max_value" doit être supérieur à "sigma_min_value".
                numpy.save("sigma_max", sigma_max)
                affichageSigmaMax['text'] = sigma_max
            else:
                affichageSigmaMax['text'] = "Doit être inférieur à sigma_max"
        else:
            affichageSigmaMax['text'] = "Erreur : doit être positif"
    except:
        affichageSigmaMax['text'] = "Erreur : doit être un réel"
            
    
def sigmas_min():
    
    '''
    Fonction qui permet de met en mémoire la valeur du paramètre "sigma_min" si celui-ci correspond effectivement au bon format de donnée.
    
    Entrée : (sous forme de sauvegardes)
    -le paramètre entré par l'utilisateur
    
    Sortie : pas de sortie autre que celle visuelle, affichée à l'utilisateur
    
    '''
    
    sigma_min = reponseSigmaMin.get()#On récupère la valeur entrée par l'utilisateur dans l'interface graphique.
    
    try:# "sigma_min_value" doit être un réel. 
        if float(sigma_min)>0:# "sigma_min_value" doit être positif.
        
            try:#Au cas où l'utilisateur à supprimé le fichier du paramètre sigma_max_value.
                sigma_max = numpy.load("sigma_max.npy")
                sigma_max_value = float(sigma_max.tolist())
            except:
                dede=1
                
            if float(sigma_min) < sigma_max_value:# "sigma_min_value" doit être inférieur à "sigma_max_value".
                numpy.save("sigma_min", sigma_min)
                affichageSigmaMin['text'] = sigma_min
            else:
                affichageSigmaMin['text'] = "Doit être inférieur à sigma_max"
        else:
            affichageSigmaMin['text'] = "Erreur : doit être positif"
    except:
        affichageSigmaMin['text'] = "Erreur : doit être un réel"
        
    



###Paramètres de base optimisés pour une CAO de taille 20x20:

iterations = 500
numpy.save("iterations", iterations)

sigma = 7.24
numpy.save("sigma", sigma)
sigma_max_value = 0.8
numpy.save("sigma_max", sigma_max_value)
sigma_min_value = 0.055
numpy.save("sigma_min", sigma_min_value)

eta = 1.26
numpy.save("eta", eta)
eta_max_value = 0.45
numpy.save("eta_max", eta_max_value)
eta_min_value = 0.1
numpy.save("eta_min", eta_min_value)

### Interface graphique :

fenetre = Tk()

### Frames et Boutons pour modifier et afficher les paramètres :

##Frame itérations :

FrameIterations = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameIterations.pack(side=TOP, padx=30, pady=30)

iter = Label(FrameIterations, text = "Nombre d'itérations :")#Affichage de texte.
reponseIteration = Entry(FrameIterations)#Prend la valeur du paramètre entrée par l'utilisateur.
votre_iteration=Label(FrameIterations, text="est l'itération choisie")#Affichage de texte.
valeurIteration = Button(FrameIterations, text =' Valider', command=iteration)
affichageIteration = Label(FrameIterations, width=25)#Affichage du paramètre ou du message d'erreur.

iter.pack(side = LEFT)
reponseIteration.pack(side = LEFT)
valeurIteration.pack(side = LEFT)
affichageIteration.pack(side = LEFT)
votre_iteration.pack(side = LEFT)

##Frame sigma :

FrameSigma = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameSigma.pack(side=TOP, padx=30, pady=30)

sig = Label(FrameSigma, text = "Valeur du sigma :")#Affichage de texte.
reponseSigma = Entry(FrameSigma)#Prend la valeur du paramètre entrée par l'utilisateur.
valeurSigma = Button(FrameSigma, text =' Valider', command=sigmas)
votre_sigma = Label(FrameSigma, text='est le sigma choisie')#Affichage de texte.
affichageSigma = Label(FrameSigma, width=25)#Affichage du paramètre ou du message d'erreur.


sig.pack(side = LEFT)
reponseSigma.pack(side = LEFT)
valeurSigma.pack(side = LEFT)
affichageSigma.pack(side = LEFT)
votre_sigma.pack(side = LEFT)

##Frame eta :

FrameEta = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameEta.pack(side=TOP, padx=30, pady=30)

et = Label(FrameEta, text = "Valeur du eta :")#Affichage de texte.
reponseEta = Entry(FrameEta)#Prend la valeur du paramètre entrée par l'utilisateur.
valeurEta = Button(FrameEta, text =' Valider', command=etas)
votre_eta = Label(FrameEta, text='est le eta choisie')#Affichage de texte.
affichageEta = Label(FrameEta, width=25)#Affichage du paramètre ou du message d'erreur.


et.pack(side = LEFT)
reponseEta.pack(side = LEFT)
valeurEta.pack(side = LEFT)
affichageEta.pack(side = LEFT)
votre_eta.pack(side = LEFT)


##Frame eta_max_value : 

FrameEtaMax = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameEtaMax.pack(side=TOP, padx=30, pady=30)

et_max = Label(FrameEtaMax, text = "Valeur du eta_max :")#Affichage de texte.
reponseEtaMax = Entry(FrameEtaMax)#Prend la valeur du paramètre entrée par l'utilisateur.
valeurEtaMax = Button(FrameEtaMax, text =' Valider', command=etas_max)
votre_eta_max = Label(FrameEtaMax, text='est le eta_max choisie')#Affichage de texte.
affichageEtaMax = Label(FrameEtaMax, width=25)#Affichage du paramètre ou du message d'erreur.


et_max.pack(side = LEFT)
reponseEtaMax.pack(side = LEFT)
valeurEtaMax.pack(side = LEFT)
affichageEtaMax.pack(side = LEFT)
votre_eta_max.pack(side = LEFT)


##Frame eta_min_value :

FrameEtaMin = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameEtaMin.pack(side=TOP, padx=30, pady=30)

et_min = Label(FrameEtaMin, text = "Valeur du eta_min :")#Affichage de texte.
reponseEtaMin = Entry(FrameEtaMin)#Prend la valeur du paramètre entrée par l'utilisateur.
valeurEtaMin = Button(FrameEtaMin, text =' Valider', command=etas_min)
votre_eta_min = Label(FrameEtaMin, text='est le eta_min choisie')#Affichage de texte.
affichageEtaMin = Label(FrameEtaMin, width=25)#Affichage du paramètre ou du message d'erreur.


et_min.pack(side = LEFT)
reponseEtaMin.pack(side = LEFT)
valeurEtaMin.pack(side = LEFT)
affichageEtaMin.pack(side = LEFT)
votre_eta_min.pack(side = LEFT)


##Frame sigma_max_value : 

FrameSigmaMax = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameSigmaMax.pack(side=TOP, padx=30, pady=30)

sig_max = Label(FrameSigmaMax, text = "Valeur du sigma_max :")#Affichage de texte.
reponseSigmaMax = Entry(FrameSigmaMax)#Prend la valeur du paramètre entrée par l'utilisateur.
valeurSigmaMax = Button(FrameSigmaMax, text =' Valider', command=sigmas_max)
votre_sigma_max = Label(FrameSigmaMax, text='est le sigma_max choisie')#Affichage de texte.
affichageSigmaMax = Label(FrameSigmaMax, width=25)#Affichage du paramètre ou du message d'erreur.


sig_max.pack(side = LEFT)
reponseSigmaMax.pack(side = LEFT)
valeurSigmaMax.pack(side = LEFT)
affichageSigmaMax.pack(side = LEFT)
votre_sigma_max.pack(side = LEFT)


##Frame sigma_min_value :

FrameSigmaMin = Frame(fenetre, borderwidth=2, relief=GROOVE)
FrameSigmaMin.pack(side=TOP, padx=30, pady=30)

sig_min = Label(FrameSigmaMin, text = "Valeur du sigma_min :")#Affichage de texte.
reponseSigmaMin = Entry(FrameSigmaMin)#Prend la valeur du paramètre entrée par l'utilisateur.
valeurSigmaMin = Button(FrameSigmaMin, text =' Valider', command=sigmas_min)
votre_sigma_min = Label(FrameSigmaMin, text='est le sigma_min choisie')#Affichage de texte.
affichageSigmaMin = Label(FrameSigmaMin, width=25)#Affichage du paramètre ou du message d'erreur.


sig_min.pack(side = LEFT)
reponseSigmaMin.pack(side = LEFT)
valeurSigmaMin.pack(side = LEFT)
affichageSigmaMin.pack(side = LEFT)
votre_sigma_min.pack(side = LEFT)


### Frames et Boutons pour lancer les différentes étapes du programme :


##Frame 1 :

Frame1 = Frame(fenetre, borderwidth=2, relief=GROOVE)#Boutons "Création de la COA", "Labelisation" et "Evaluation".
Frame1.pack(side=LEFT, padx=50, pady=110)


bouton=Button(Frame1, text="Création de la COA", command=coa)
bouton.pack(padx =8,pady = 8 )
affichageCAO = Label(Frame1, width=25)#Affichage du message d'erreur.
affichageCAO.pack()

bouton3=Button(Frame1, text="Labelisation", command=label)
bouton3.pack(padx =8,pady = 8 )
affichageLabel = Label(Frame1, width=25)#Affichage du message d'erreur.
affichageLabel.pack()

bouton4=Button(Frame1, text="Evaluation", command=eval)
bouton4.pack(padx =8,pady = 8 )
affichageEval = Label(Frame1, width=25)#Affichage du message d'erreur.
affichageEval.pack()


##Frame 2 :

Frame2 = Frame(fenetre, borderwidth=2, relief=GROOVE)#Boutons : "Affichage COA", "Effacer les fichiers temporaires", "Quitter".
Frame2.pack(side=LEFT, padx=40, pady=75)

bouton2=Button(Frame2, text="Affichage COA", command=graph)
bouton2.pack(padx =10,pady = 10 )

bouton7=Button(Frame2, text="Effacer les fichiers temporaires", command=erase)
bouton7.pack(padx =15,pady = 15 )

bouton5=Button(Frame2, text="Quitter", command=fenetre.destroy)#La fonction associée permet (sous windows) de fermer l'interface graphique Tkinter lorsque l'utilisateur, mais sans pour autant supprimer les fichiers npy sauvegardés.
bouton5.pack(padx =15,pady = 15 )

##Frame 3 :

Frame3 = Frame(fenetre, borderwidth=2, relief=GROOVE)#Boutons : "Création, labelling et évaluation", "Résultat".
Frame3.pack(side=LEFT, padx=40, pady=75)

bouton6=Button(Frame3, text="Création, labelling et évaluation", command=allfunction)
bouton6.pack(padx =20,pady = 20 )
affichageAllfunction = Label(Frame3, width=35)#Affichage du message d'erreur.
affichageAllfunction.pack()

bouton6=Button(Frame3, text="Résultat : ", command=result)
bouton6.pack(padx =20,pady = 20 )#Affichage du message d'erreur.
affichageResult = Label(Frame3, width=20)
affichageResult.pack()











###Boucle de l'interface graphique

fenetre.mainloop()
