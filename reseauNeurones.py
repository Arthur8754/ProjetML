"""
Dans cette classe, on implémente un réseau de neurones multicouches, d dimensions pour l'entrée et K classes
"""

from sklearn.neural_network import MLPClassifier
import numpy as np

class reseauNeurones:

    def __init__(self, lamb):
        self.lamb = lamb
        self.clf = MLPClassifier(hidden_layer_sizes=(10,5), activation="tanh", solver="lbfgs", alpha=lamb, learning_rate="constant")

    def entrainement(self, x_train, t_train):
        self.clf.fit(x_train, t_train)
    
    def prediction(self, x_tab):
        """
        Détermine les classes pour chaque entrée de l'ensemble x_tab, grâce à des forward pass
        """
        y = self.clf.predict(x_tab)
        return y

    def erreur(self, pred, t):
        """
        Détermine l'erreur associée à une prédiction (0 si bien prédit, 1 sinon)
        """
        if t==pred:
            return 0
        else:
            return 1

    def erreur_train_and_valid(self, x_tab, t_tab, k):
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
            self.entrainement(x_apprentissage, t_apprentissage)

            # Prédiction et erreur pour l'entraînement :
            #print(x_apprentissage)
            pred_app = self.prediction(x_apprentissage)
            #pred_app = np.array([self.prediction(x) for x in x_apprentissage]) #tableau des prédictions faites
            #print(pred_app)
            #print("")
            erreur_app_array = np.array([self.erreur(t,pred) for t,pred in zip(t_apprentissage,pred_app)]) #tableau des erreurs
            erreurs_train[j] = round(100*np.count_nonzero(erreur_app_array)/len(erreur_app_array),1)

            # Prédiction et erreur pour la validation :
            pred_valid = self.prediction(x_validation)
            #pred_valid = np.array([self.prediction(x) for x in x_validation]) #tableau des prédictions faites
            erreur_valid_array = np.array([self.erreur(t,pred) for t,pred in zip(t_validation,pred_valid)]) #tableau des erreurs
            erreurs_valid[j] = round(100*np.count_nonzero(erreur_valid_array)/len(erreur_valid_array),1)
        
        erreur_train = round(np.mean(erreurs_train),1)
        erreur_valid = round(np.mean(erreurs_valid),1)

        return erreur_train, erreur_valid