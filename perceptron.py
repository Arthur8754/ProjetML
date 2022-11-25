"""
Dans cette classe, on implémente la méthode du perceptron, K classes et d dimensions.
"""

from sklearn.linear_model import SGDClassifier
import numpy as np

class perceptron:

    def __init__(self,lamb):
        """
        Forme de W : matrice dxK : W = (w1...wK), où wi = T(w1i,w2i,...,wdi).
        """
        self.W = None #matrice des paramètres (biais NON inclus)
        self.W_0 = None #biais
        self.lamb = lamb #terme de régularisation (hyper-paramètre)

    def entrainement(self, x_train, t_train):
        """
        Paramètres à inclure dans la fonction SGDCClassifier.
        Dans le cas d'un MAP, E(W) = fonction_perte + lamb*R(W).
        --> Fonction_perte = celle du perceptron => loss="perceptron"
        --> R(W) : terme de régularisation : on prend la norme 2 ici => penalty = l2
        --> lamb : constante devant le terme de régularisation => alpha = lamb
        --> eta0 : le learning rate dans la descente de gradient        
        """
        modele = SGDClassifier(loss="perceptron",penalty="l2",alpha=self.lamb,learning_rate="constant",eta0=0.001,max_iter=1000)
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