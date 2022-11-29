import lireDonnees
import perceptron
import visualiserDonnees
import numpy as np
import testPerceptron
import matplotlib.pyplot as plt

def main():
    """
    Point d'entrée du programme.
    """
    # 1. EXTRACTION ET VISUALISATION DES DONNÉES D'ENTRAÎNEMENT :

    # Extraction des données d'entraînement et de test
    rd = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
    x_train,t_train = rd.extract_train_data()
    x_test = rd.extract_test_data()
    reference = rd.create_reference() #liste des différentes espèces possibles
    x_train_normalized,x_test_normalized = rd.normalize_data(1*x_train, 1*x_test)

    # Visualisation des données d'entraînement non normalisées
    vd = visualiserDonnees.visualiserDonnees(x_train,x_test,t_train,reference)
    vd.visualiserEntrainement(False)

    # Visualisation des données d'entraînement normalisées :
    vd2 = visualiserDonnees.visualiserDonnees(x_train_normalized,x_test_normalized,t_train,reference)
    vd2.visualiserEntrainement(True)

    # 2. TESTS SUR LE MODÈLE DU PERCEPTRON :
    print("-------------------------------")
    print("TESTS MODÈLES DU PERCEPTRON : ")

    testPerc = testPerceptron.testPerceptron()

    # Test 1 : données non normalisées, sans recherche d'hyper-paramètres :
    err_train1, err_valid1 = testPerc.test(x_train, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, loss="perceptron", penalty="l2", learning_rate="constant", k=10, eta=0.001)
    print("# Tests sur des données non normalisées, sans recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train1} %")
    print(f"Erreur de validation : {err_valid1} %")
    print("")

    # Test 2 : données normalisées, sans recherche d'hyper-paramètres :
    err_train2, err_valid2 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, loss="perceptron", penalty="l2", learning_rate="constant", k=10, eta=0.001)
    print("# Tests sur des données normalisées, sans recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train2} %")
    print(f"Erreur de validation : {err_valid2} %")
    print("-------------------------------")

    # Comparaison données non normalisées et normalisées :
    plt.figure(0)
    plt.bar(["Non normalisé, entraînement","Non normalisé, validation","Normalisé, entraînement","Normalisé, validation"],[err_train1, err_valid1, err_train2, err_valid2])
    plt.title("Comparaison erreurs d'entraînement et de validation entre données non normalisées et normalisées")
    plt.xlabel("Non normalisé / normalisé")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)
    plt.show()

if __name__=="__main__":
    main()
