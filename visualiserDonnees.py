"""
Dans cette classe, on crée les méthodes permettant d'afficher les données en 2D
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from sklearn.decomposition import PCA 

class visualiserDonnees:
    def __init__(self,x_train,x_test,t_train,reference):
        self.x_train = x_train
        self.x_test = x_test
        self.t_train = t_train
        self.reference = reference
        self.dico_map = {} #clé : espèce, valeur : couleur

    def creerDicoMap(self):
        # Mapping : on associe une espèce à une couleur
        colors = colors = random.sample(list(mcolors.CSS4_COLORS),99) #sélection aléatoire de 99 couleurs différentes (99 car 99 espèces)
        dico_map = {} #clé : espèce, valeur : sa couleur
        for i in range(len(self.reference)): 
            dico_map[self.reference[i]] = colors[i] 
        return dico_map

    def visualiserEntrainement(self, normalized):
        """
        On utilise une ACP pour visualiser les points en 2D.
        """
        # Mapping : on associe une espèce à une couleur
        dico_map = self.creerDicoMap()
        tab_colors = [] #tab_colors[i] correspond à la couleur associée à l'espèce t_train[i]
        for i in range(len(self.t_train)):
            tab_colors.append(dico_map[self.t_train[i]])

        # Application de l'ACP sur x_train
        pca = PCA(2) #on veut afficher en 2D, on retiendra les 2 composantes principales de x_train
        x_train_2d = pca.fit_transform(self.x_train)

        # Affichage :
        plt.figure(0)
        plt.scatter(x_train_2d[:,0],x_train_2d[:,1],c=tab_colors)
        plt.xlabel("1ère composante principale")
        plt.ylabel("2ème composante principale")
        if normalized:
            plt.title("Données d'entraînement normalisées (vues en 2D à l'aide d'une ACP)")
        else:
            plt.title("Données d'entraînement non normalisées (vues en 2D à l'aide d'une ACP)")
        plt.show()

    def visualiserTest(self):
        """
        Là aussi, on fait une ACP. Mais pas de mapping, car on ne connaît pas la cible
        """
        # Application de l'ACP sur x_train
        pca = PCA(2) #on veut afficher en 2D, on retiendra les 2 composantes principales de x_train
        x_test_2d = pca.fit_transform(self.x_test)

        # Affichage :
        plt.figure(1)
        plt.scatter(x_test_2d[:,0],x_test_2d[:,1],c="black")
        plt.xlabel("1ère composante principale")
        plt.ylabel("2ème composante principale")
        plt.title("Données de test (vues en 2D à l'aide d'une ACP)")
        plt.show()
