"""
Tests pour les réseaux de neurones multicouches
"""

import reseauNeurones

class testReseauNeurones:

    def __init__(self):
        pass

    def test(self,x_train, t_train, lamb):
        modele = reseauNeurones.reseauNeurones(lamb)
        erreur_train, erreur_valid = modele.erreur_train_and_valid(x_train, t_train, k=10)
        return erreur_train, erreur_valid