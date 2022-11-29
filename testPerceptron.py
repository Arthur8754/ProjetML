"""
Dans cette classe, on applique le modèle du perceptron sur les données d'entraînement et de test, et on modifie plusieurs paramètres
et on effectue des traitements sur les données.
"""

import perceptron
import lireDonnees
import numpy as np
import matplotlib.pyplot as plt

class testPerceptron:

    def __init__(self):
        """
        Fixe les paramètres à utiliser dans l'algorithme avec le perceptron.
        """
        pass

    def test(self,x_train, t_train, reference, lamb, recherche_hyper_parametres, loss, penalty, learning_rate, eta, k):
        """
        Crée un instance de la classe perceptron, et détermine l'erreur d'entraînement et de validation.
        """
        modele = perceptron.perceptron(lamb)
        erreur_train, erreur_valid = modele.erreur_train_and_valid(x_train, t_train, reference, k, recherche_hyper_parametres, loss, penalty, learning_rate, eta)
        return erreur_train, erreur_valid
