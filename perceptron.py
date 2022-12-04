"""
Dans cette classe, on implémente la méthode du perceptron, K classes et d dimensions.

Rq : pour l'instant on ne gère que le cas linéaire --> rajouter le phi pour gérer le cas non linéaire (à voir)

"""

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
import lireDonnees
import numpy as np
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self, lamb):
        self.lamb = lamb #terme de régularisation (hyper-paramètre)
        self.clf = SGDClassifier(loss="perceptron",alpha=self.lamb, eta0=0.01)

    def entrainement(self, x_train, t_train, recherche_hyp_param):
        """
        Entraîne le classifieur à l'aide de x_train et t_train.
        """
        if recherche_hyp_param:
            self.recherche_hyper_parametres(x_train, t_train, k=10)
        self.clf.fit(x_train, t_train)
    
    def prediction(self,x_tab):
        """
        Détermine les classes pour chaque entrée x de x_tab.
        """
        classes = self.clf.predict(x_tab)
        return classes

    def erreur(self, x_tab, t_tab):
        """
        Détermine le taux d'erreur sur la prédiction faite sur x_tab par rapport à t_tab
        """
        err = 1-self.clf.score(x_tab, t_tab)
        return err

    def recherche_hyper_parametres(self, x_tab, t_tab, k):
        """
        Recherche la valeur optimale pour l'hyper-paramètre self.lamb, à l'aide d'une k-fold cross validation.
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
        # 1. EXTRACTION DES ENSEMBLES D'APPRENTISSAGE ET DE VALIDATION

        kf = KFold(n_splits=k, shuffle=True) #va shuffle les données et les séparer en k groupes distincts

        # 2. DÉTERMINATION DES ERREURS D'ENTRAÎNEMENT ET DE VALIDATION :

        candidats_lamb = np.arange(start=0.0001,stop=0.1,step=0.0001,dtype=float)  
        lamb_optimal = candidats_lamb[0]
        erreur_min = 1
        erreurs_lamb_array_train = [] # pour plot : tableau contenant les erreurs d'entraînement moyenne, pour chaque valeur de l'hyper-param
        erreurs_lamb_array_valid = [] # pour plot : tableau contenant les erreurs de validation moyenne, pour chaque valeur de l'hyper-param
        
        # Pour chaque valeur de l'hyper-paramètre :
        for lamb in candidats_lamb: 
            self.lamb = lamb
            self.clf.alpha = self.lamb
            print(self.clf.alpha)

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
            erreurs_lamb_array_train.append(E_train)    
            erreurs_lamb_array_valid.append(E_valid)
            if E_valid < erreur_min:
                lamb_optimal = self.lamb
                erreur_min = E_valid
        self.lamb = lamb_optimal
        self.clf.alpha = self.lamb

        # Plottage des erreurs d'entraînement et de validation :
        plt.figure(0) 
        plt.plot(candidats_lamb, erreurs_lamb_array_train, color='blue',label="E_train")
        plt.plot(candidats_lamb, erreurs_lamb_array_valid, color='red', label="E_valid")
        plt.xlabel("lambda")
        plt.ylabel("Taux d'erreur")
        plt.title("Évolution E_train et E_valid en fonction de lambda")
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/perceptronCrossValidation.png")


    def erreur_train_valid(self, x_tab, t_tab, k):
        """
        Détermine l'erreur d'entraînement moyenne et l'erreur de validation moyenne, grâce à une cross validation.
        Concrètement, on découpe x_train en k groupes distincts. Puis, on fait k fois :
        --> sélection d'un groupe pour la validation (à chaque fois différent d'une itération à l'autre) ;
        --> sélection du reste pour l'apprentissage.
        --> entraînement puis prédiction du modèle sur l'ensemble d'apprentissage --> erreur_app ;
        --> prédiction du modèle sur l'ensemble de validation --> erreur_valid
        Puis on prend la moyenne sur erreur_app (entraînement), puis la moyenne sur erreur_valid (validation)
        Ainsi, on peut savoir comment réagit notre modèle face à des données qu'il n'a jamais vues (avec la validation).
        """
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
        plt.savefig("figures/perceptronErreur.png")
        return E_train, E_valid

def main():
    # Lecture des données :
    rd = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
    x_train,t_train = rd.extract_train_data()
    x_test = rd.extract_test_data()
    x_train_normalized, x_test_normalized = rd.normalize_data(x_train, x_test)
    
    # Modèle :
    perc = perceptron(lamb=0.01)
    #perc.entrainement(x_train, t_train, recherche_hyp_param=True) #pour recherche hyper-paramètres
    #print(perc.lamb)
    E_train, E_valid = perc.erreur_train_valid(x_train, t_train, 10)
    print("Taux d'erreur d'entraînement : % .2f" % E_train)
    print("Taux d'erreur de validation : % .2f" % E_valid)

if __name__=="__main__":
    main()