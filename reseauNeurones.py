"""
Dans cette classe, on implémente un réseau de neurones multicouches, d dimensions pour l'entrée et K classes
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import lireDonnees
import numpy as np
import matplotlib.pyplot as plt

class reseauNeurones:

    def __init__(self, lamb):
        self.lamb = lamb
        self.clf = MLPClassifier(alpha=self.lamb)

    def entrainement(self, x_train, t_train, recherche_hyp_param):
        self.clf.fit(x_train, t_train)
    
    def prediction(self, x_tab):
        classes = self.clf.predict(x_tab)
        return classes
    
    def erreur(self, x_tab, t_tab):
        err = 1-self.clf.score(x_tab, t_tab)
        return err

    def erreur_train_valid(self, x_tab, t_tab, k):
        # 1. EXTRACTION DES ENSEMBLES D'APPRENTISSAGE ET DE VALIDATION

        kf = KFold(n_splits=k, shuffle=True) #va shuffle les données et les séparer en k groupes distincts

        # 2. DÉTERMINATION DES ERREURS D'ENTRAÎNEMENT ET DE VALIDATION :

        erreurs_app = [] #tableau stockant l'erreur d'entraînement pour chaque k
        erreurs_valid = [] #tableau stockant l'erreur de validation pour chaque k    

        # Pour chaque groupe de données :
        for app_index, valid_index in kf.split(x_tab):
            x_app, x_valid = x_tab[app_index], x_tab[valid_index]
            t_app, t_valid = t_tab[app_index], t_tab[valid_index]

            # Entraînement sur x_app et t_app :
            self.entrainement(x_app, t_app, recherche_hyp_param=False)

            # Erreur d'entraînement :
            erreurs_app.append(self.erreur(x_app, t_app)) #erreur moyenne sur x_app.

            # Erreur de validation :
            erreurs_valid.append(self.erreur(x_valid, t_valid)) #erreur moyenne sur x_valid.

        E_train = np.mean(erreurs_app)
        E_valid = np.mean(erreurs_valid)

        plt.figure(0) 
        plt.bar(["E_train","E_valid"],[E_train, E_valid])
        plt.ylabel("Taux d'erreur")
        plt.ylim(0,1)
        plt.title(f"Erreur d'entraînement et de validation, lambda = {self.lamb}")
        plt.savefig("figures/reseauNeuronesErreur.png")
        return E_train, E_valid

def main():
    # Lecture des données :
    rd = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
    x_train,t_train = rd.extract_train_data()
    
    # Modèle :
    RN = reseauNeurones(lamb=0.01)
    E_train, E_valid = RN.erreur_train_valid(x_train, t_train, 10)
    print("Taux d'erreur d'entraînement : % .2f" % E_train)
    print("Taux d'erreur de validation : % .2f" % E_valid)

if __name__=="__main__":
    main()
