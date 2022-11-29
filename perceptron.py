"""
Dans cette classe, on implémente la méthode du perceptron, K classes et d dimensions.
"""

from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self,lamb):
        """
        Forme de W : matrice dxK : W = (w1...wK), où wi = T(w1i,w2i,...,wdi).
        """
        self.W = None #matrice des paramètres (biais NON inclus)
        self.W_0 = None #biais
        self.lamb = lamb #terme de régularisation (hyper-paramètre)

    def entrainement(self, x_train, t_train, recherche_hyper_parametres, reference, loss, penalty, learning_rate, eta):
        """
        Paramètres à inclure dans la fonction SGDCClassifier.
        Dans le cas d'un MAP, E(W) = fonction_perte + lamb*R(W).
        --> Fonction_perte = celle du perceptron => loss="perceptron"
        --> R(W) : terme de régularisation : on prend la norme 2 ici => penalty = l2
        --> lamb : constante devant le terme de régularisation => alpha = lamb
        --> eta0 : le learning rate dans la descente de gradient  

        cross_validation = True si on veut une recherche d'hyper-paramètres, False sinon.      
        """
        if recherche_hyper_parametres:
            self.recherche_hyper_parametres(x_train, t_train, reference, loss, penalty, learning_rate, eta)
        modele = SGDClassifier(loss=loss,penalty=penalty,alpha=self.lamb,learning_rate=learning_rate,eta0=eta,max_iter=1000) #SGDClassifier : Stochastic Gradient Descent Classifier
        #modele = SGDClassifier(loss="perceptron",penalty="l2",alpha=self.lamb,learning_rate="constant",eta0=0.001,max_iter=1000)
        modele.fit(x_train,t_train)
        self.W = modele.coef_ 
        self.W_0 = modele.intercept_
    
    def prediction(self,x,reference):
        """
        Détermine y = np.transpose(W)*x où x est une entrée 1D
        """
        y = self.W_0 + np.dot(self.W,np.transpose(x))
        indice = np.argmax(y)
        return reference[indice]

    def erreur(self,t,pred):
        """
        Retourne 1 si t!=pred, 0 sinon.
        pred est l'espèce prédite pour une entrée x donnée, t est la cible attendue pour ce x.
        """
        if t==pred:
            return 0
        else:
            return 1

    def recherche_hyper_parametres(self,x_tab,t_tab,reference, loss, penalty, learning_rate, eta):
        """
        Recherche le meilleur hyperparamètre lamb à l'aide de la K-fold cross validation.
        RAPPEL : k-fold cross validation :
        1. Mélanger les données d'entraînement
        2. Découper les données d'entraînement en k groupes distincts
        3. Pour chaque valeur possible de l'hyper-paramètre :
            a. Faire k fois :
                i. Prendre 1 groupe sur les k pour la validation (à chaque fois différent d'une itération à l'autre), le reste pour l'apprentissage
                ii. Entraîner le modèle sur D_apprentissage
                iii. Prédire sur D_valid, et déterminer l'erreur de validation associée
            b. Déterminer l'erreur moyenne de validation sur toutes les itérations.
        4. Retenir la valeur de l'hyper-paramètre pour laquelle l'erreur moyenne de validation est minimale
        """
        # Mélange des données d'entraînement :
        shuffler = np.random.permutation(len(x_tab)) #permute les index aléatoirement
        x_tab_shuffled = x_tab[shuffler] #on associe x_tab aux nouveaux index
        t_tab_shuffled = t_tab[shuffler] #on associe t_tab aux nouveaux index

        # Découpage des données d'entraînement en k=10 groupes distincts
        k=10
        x_split = np.split(x_tab_shuffled,k)
        t_split = np.split(t_tab_shuffled,k)

        # Extraction des données d'entraînement et de validation :
        X_trains = [] #tableau à k=10 éléments : stocke les X_train pour les 10 itérations de la k fold cross validation
        t_trains = []
        X_valids = []
        t_valids = []

        for j in range(k):
            # Recherche de l'ensemble de validation (un groupe sur les k groupes (à chaque fois différent))
            X_valid = x_split[j]
            t_valid = t_split[j]
            
            # Recherche de l'ensemble d'apprentissage (tout sauf le groupe de validation)
            x_split1 = x_split[:j]
            x_split2 = x_split[j+1:]
            t_split1 = t_split[:j]
            t_split2 = t_split[j+1:]

            if len(x_split1) != 0 and len(x_split2)!=0:
                X_train = np.array(np.concatenate((x_split1,x_split2)))
                X_train = X_train.reshape((-1,len(x_tab[0])))
                t_train = np.array(np.concatenate((t_split1,t_split2)))
                t_train = t_train.reshape((1,-1))[0]
            elif len(x_split1) != 0 and len(x_split2) == 0:
                X_train = np.concatenate(x_split1)
                t_train = np.concatenate(t_split1)
            elif len(x_split1) == 0 and len(x_split2) !=0:
                X_train = np.concatenate(x_split2)
                t_train = np.concatenate(t_split2)
            else:
                X_train = []
                t_train = []

            X_trains.append(1*X_train)
            t_trains.append(1*t_train)
            X_valids.append(1*X_valid)
            t_valids.append(1*t_valid)
        
        # Candidats hyper-paramètre lamb :
        candidats_lamb = np.arange(start=0.000000001,stop=2,step=0.01,dtype=float)
        arg_min_lamb = candidats_lamb[0]
        erreur_min = 100      

        #Pour chaque valeur de l'hyper-paramètre        
        for l in range(len(candidats_lamb)):
            self.lamb = candidats_lamb[l]

            erreurs_differents_groupes = np.empty(k) #tableau contenant les erreurs moyennes pour chaque groupe de données

            # Pour chaque groupe de données :
            for j in range(k):
                x_apprentissage, x_validation = X_trains[j], X_valids[j]
                t_apprentissage, t_validation = t_trains[j], t_valids[j]

                # Entraînement sur x_apprentissage et t_apprentissage :
                self.entrainement(x_apprentissage, t_apprentissage, False, reference, loss, penalty, learning_rate, eta) #False car sinon il va faire un appel récursif infini

                # Prédiction sur valid :
                erreur_valid = np.empty(len(x_validation))
                for v in range(len(x_validation)): #pour chaque donnée de D_valid
                    pred = self.prediction(x_validation[v],reference)
                    erreur_pred = self.erreur(t_validation[v],pred) #0 si bien prédit, 1 sinon
                    erreur_valid[v] = erreur_pred

                erreurs_differents_groupes[j] = np.mean(erreur_valid)

            # Erreur pour l'hyper-paramètre (erreur de validation moyenne avec cet hyper-paramètre)
            erreur_hyper_parametre = np.mean(erreurs_differents_groupes)
            if erreur_hyper_parametre<erreur_min:
                arg_min_lamb = self.lamb
                erreur_min =erreur_hyper_parametre

        # On retient le self.lamb pour lequel l'erreur de validation moyenne est la plus faible    
        self.lamb = arg_min_lamb
    
    def erreur_train_and_valid(self, x_tab, t_tab, reference, k, recherche_hyper_parametres, loss, penalty, learning_rate, eta):
        """
        Détermine l'erreur d'entraînement et de validation. Ce calcul se fait à l'aide d'une cross validation : de façon semblable à la recherche d'hyper-paramètres, 
        on découpe x_tab en k groupes distincts, puis, on entraîne le modèle sur k-1 groupes et on teste sur un des groupes, et on fait ça k fois, en prenant à chaque
        fois un ensemble de validation différent.
        """
        # Mélange des données d'entraînement :
        shuffler = np.random.permutation(len(x_tab)) #permute les index aléatoirement
        x_tab_shuffled = x_tab[shuffler] #on associe x_tab aux nouveaux index
        t_tab_shuffled = t_tab[shuffler] #on associe t_tab aux nouveaux index

        # Découpage des données d'entraînement en k groupes distincts
        x_split = np.split(x_tab_shuffled,k)
        t_split = np.split(t_tab_shuffled,k)

        # Extraction des données d'entraînement et de validation :
        X_trains = [] #tableau à k=10 éléments : stocke les X_train pour les 10 itérations de la k fold cross validation
        t_trains = []
        X_valids = []
        t_valids = []

        for j in range(k):
            # Recherche de l'ensemble de validation (un groupe sur les k groupes (à chaque fois différent))
            X_valid = x_split[j]
            t_valid = t_split[j]
            
            # Recherche de l'ensemble d'apprentissage (tout sauf le groupe de validation)
            x_split1 = x_split[:j]
            x_split2 = x_split[j+1:]
            t_split1 = t_split[:j]
            t_split2 = t_split[j+1:]

            if len(x_split1) != 0 and len(x_split2)!=0:
                X_train = np.array(np.concatenate((x_split1,x_split2)))
                X_train = X_train.reshape((-1,len(x_tab[0])))
                t_train = np.array(np.concatenate((t_split1,t_split2)))
                t_train = t_train.reshape((1,-1))[0]
            elif len(x_split1) != 0 and len(x_split2) == 0:
                X_train = np.concatenate(x_split1)
                t_train = np.concatenate(t_split1)
            elif len(x_split1) == 0 and len(x_split2) !=0:
                X_train = np.concatenate(x_split2)
                t_train = np.concatenate(t_split2)
            else:
                X_train = []
                t_train = []

            X_trains.append(1*X_train)
            t_trains.append(1*t_train)
            X_valids.append(1*X_valid)
            t_valids.append(1*t_valid)
        
        erreurs_train = np.empty(k) #tableau stockant l'erreur d'entraînement pour chaque k
        erreurs_valid = np.empty(k) #tableau stockant l'erreur de validation pour chaque k    

        # Pour chaque groupe de données :
        for j in range(k):
            x_apprentissage, x_validation = X_trains[j], X_valids[j]
            t_apprentissage, t_validation = t_trains[j], t_valids[j]

            # Entraînement sur x_apprentissage et t_apprentissage :
            self.entrainement(x_apprentissage, t_apprentissage, recherche_hyper_parametres, reference, loss, penalty, learning_rate, eta)

            # Prédiction et erreur pour l'entraînement :
            pred_app = np.array([self.prediction(x,reference) for x in x_apprentissage]) #tableau des prédictions faites
            erreur_app_array = np.array([self.erreur(t,pred) for t,pred in zip(t_apprentissage,pred_app)]) #tableau des erreurs
            erreurs_train[j] = round(100*np.count_nonzero(erreur_app_array)/len(erreur_app_array),1)

            # Prédiction et erreur pour la validation :
            pred_valid = np.array([self.prediction(x,reference) for x in x_validation]) #tableau des prédictions faites
            erreur_valid_array = np.array([self.erreur(t,pred) for t,pred in zip(t_validation,pred_valid)]) #tableau des erreurs
            erreurs_valid[j] = round(100*np.count_nonzero(erreur_valid_array)/len(erreur_valid_array),1)
        
        erreur_train = round(np.mean(erreurs_train),1)
        erreur_valid = round(np.mean(erreurs_valid),1)

        return erreur_train, erreur_valid