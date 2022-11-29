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
        

# def compare_combinaisons():
#     # Récupération des données non normalisées :
#     reader = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
#     x_train,t_train = reader.extract_train_data()
#     x_test = reader.extract_test_data()
#     reference = reader.create_reference()

#     # Génération des données normalisées des données :
#     x_train_normalized, x_test_normalized = reader.normalize_data(1*x_train, 1*x_test)

#     # # 1. COMPARAISON ENTRE LES DONNÉES NORMALISÉES ET NON NORMALISÉES :
#     # non_normalized = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_non_normalized = non_normalized.test_combinaison(x_train, t_train, reference)

#     # normalized = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_normalized = normalized.test_combinaison(x_train_normalized, t_train, reference)

#     # plt.figure(0)
#     # plt.bar(["Données non normalisées","Données normalisées"],[erreur_train_non_normalized, erreur_train_normalized])
#     # plt.title("Comparaison des erreurs d'entraînement entre les données non normalisées et normalisées")
#     # plt.xlabel("Non normalisé / normalisé")
#     # plt.ylabel("Erreur d'entraînement (%)")
#     # plt.ylim(0,100)
    
#     # # 2. COMPARAISON ENTRE LA LOSS PERCEPTRON ET LA HINGE LOSS
#     # loss_perceptron = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_loss_perceptron = loss_perceptron.test_combinaison(x_train_normalized, t_train, reference)

#     # hinge_loss = testsPerceptron(lamb=0.01, eta=0.001, loss="hinge", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_hinge_loss = hinge_loss.test_combinaison(x_train_normalized, t_train, reference)

#     # plt.figure(1)
#     # plt.bar(["Perceptron","Hinge"],[erreur_train_loss_perceptron,erreur_train_hinge_loss])
#     # plt.title("Comparaison des erreurs d'entraînement entre les loss")
#     # plt.xlabel("Loss function")
#     # plt.ylabel("Erreur d'entraînement (%)")
#     # plt.ylim(0,100)

#     # # 3. COMPARAISON ENTRE LES TERMES DE RÉGULARISATIONS :
#     # l1_reg = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l1", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_l1_reg = l1_reg.test_combinaison(x_train_normalized, t_train, reference)

#     # l2_reg = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_l2_reg = l2_reg.test_combinaison(x_train_normalized, t_train, reference)

#     # elastic_reg = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="elasticnet", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_elastic_reg = elastic_reg.test_combinaison(x_train_normalized, t_train, reference)

#     # plt.figure(2)
#     # plt.bar(["l1","l2","elasticnet"],[erreur_train_l1_reg,erreur_train_l2_reg,erreur_train_elastic_reg])
#     # plt.title("Comparaison des erreurs d'entraînement entre les termes de régularisation")
#     # plt.xlabel("Terme de régularisation")
#     # plt.ylabel("Erreur d'entraînement (%)")
#     # plt.ylim(0,100)

#     # # 4. COMPARAISON ENTRE LES LEARNING_RATES :
#     # constant = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
#     # erreur_train_constant = constant.test_combinaison(x_train_normalized, t_train, reference)

#     # optimal = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="optimal", recherche_hyper_parametres=False)
#     # erreur_train_optimal = optimal.test_combinaison(x_train_normalized, t_train, reference)

#     # invscaling = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="invscaling", recherche_hyper_parametres=False)
#     # erreur_train_invscaling = invscaling.test_combinaison(x_train_normalized, t_train, reference)

#     # plt.figure(3)
#     # plt.bar(["constant","optimal","invscaling"],[erreur_train_constant,erreur_train_optimal,erreur_train_invscaling])
#     # plt.title("Comparaison des erreurs d'entraînement entre les learning rates")
#     # plt.xlabel("Modification de eta")
#     # plt.ylabel("Erreur d'entraînement (%)")
#     # plt.ylim(0,100)

#     # plt.show()

#     # Utilisation des paramètres optimaux : 
# #     test = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
# #     erreur_train = test.test_combinaison(x_train_normalized, t_train, reference)
# #     print(f"Erreur d'entraînement : {erreur_train} %")

# # if __name__=="__main__":
# #     reader = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
# #     x_train,t_train = reader.extract_train_data()
# #     x_test = reader.extract_test_data()
# #     reference = reader.create_reference()

# #     # Génération des données normalisées des données :
# #     x_train_normalized, x_test_normalized = reader.normalize_data(1*x_train, 1*x_test)

# #     test = testsPerceptron(lamb=0.01, eta=0.001, loss="perceptron", penalty="l2", learning_rate="constant", recherche_hyper_parametres=False)
# #     erreur_train, erreur_valid = test.cross_validation(x_train, t_train, reference)   
# #     print(erreur_train)
# #     print(erreur_valid)
