import lireDonnees
import visualiserDonnees
import testPerceptron
import testSvm
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
    print("TESTS MODÈLE DU PERCEPTRON : ")

    testPerc = testPerceptron.testPerceptron()

    # Test 1 : comparaison données non normalisées et normalisées :
    err_train1, err_valid1 = testPerc.test(x_train, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    err_train2, err_valid2 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    print("# Données non normalisées - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train1} %")
    print(f"Erreur de validation : {err_valid1} %")
    print("")

    print("# Données normalisées - pas de recherche d'hyper-paramètres :  : ")
    print(f"Erreur d'entraînement : {err_train2} %")
    print(f"Erreur de validation : {err_valid2} %")
    print("")

    plt.figure(0)
    plt.bar(["Non normalisé, entraînement","Non normalisé, validation","Normalisé, entraînement","Normalisé, validation"],[err_train1, err_valid1, err_train2, err_valid2])
    plt.title("PERCEPTRON - comparaison erreurs d'entraînement et de validation entre données non normalisées et normalisées")
    plt.xlabel("Non normalisé / normalisé")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)

    # Test 2 : comparaison des résultats selon le terme de régularisation :
    err_train3, err_valid3 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l1", learning_rate="constant", k=10, eta=0.001)
    err_train4, err_valid4 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    err_train5, err_valid5 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="elasticnet", learning_rate="constant", k=10, eta=0.001)
    
    print("# Données normalisées, penalty l1 - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train3} %")
    print(f"Erreur de validation : {err_valid3} %")
    print("")

    print("# Données normalisées, penalty l2 - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train4} %")
    print(f"Erreur de validation : {err_valid4} %")
    print("")

    print("# Données normalisées, penalty elasticnet - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train5} %")
    print(f"Erreur de validation : {err_valid5} %")
    print("")

    plt.figure(1)
    plt.bar(["l1 entraînement","l1 validation","l2 entraînement","l2 validation", "elasticnet entraînement","elasticnet validation"],[err_train3, err_valid3, err_train4, err_valid4, err_train5, err_valid5])
    plt.title("PERCEPTRON - Comparaison erreurs d'entraînement et de validation entre constant, optimal et invscaling learning rate")
    plt.xlabel("Penalty")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)


    # Test 3 : comparaison des résultats selon le learning_rate :
    err_train6, err_valid6 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    err_train7, err_valid7 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="optimal", k=10, eta=0.001)
    err_train8, err_valid8 = testPerc.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="invscaling", k=10, eta=0.001)
    
    print("# Données normalisées, penalty l2, learning_rate constant - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train6} %")
    print(f"Erreur de validation : {err_valid6} %")
    print("")

    print("# Données normalisées, penalty l2, learning_rate optimal - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train7} %")
    print(f"Erreur de validation : {err_valid7} %")
    print("")

    print("# Données normalisées, penalty l2, learning_rate invscaling - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train8} %")
    print(f"Erreur de validation : {err_valid8} %")
    print("-------------------------------")

    plt.figure(2)
    plt.bar(["constant entraînement","constant validation","optimal entraînement","optimal validation", "invscaling entraînement","invscaling validation"],[err_train6, err_valid6, err_train7, err_valid7, err_train8, err_valid8])
    plt.title("PERCEPTRON - Comparaison erreurs d'entraînement et de validation entre constant, optimal et invscaling learning rate")
    plt.xlabel("Learning rate")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)

    plt.show()

    # 2. TESTS SUR LE MODÈLE SVM :
    print("-------------------------------")
    print("TESTS SVM : ")

    testS = testSvm.testSvm()

    # Test 1 : comparaison données non normalisées et normalisées :
    err_train1, err_valid1 = testS.test(x_train, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    err_train2, err_valid2 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    print("# Données non normalisées - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train1} %")
    print(f"Erreur de validation : {err_valid1} %")
    print("")

    print("# Données normalisées - pas de recherche d'hyper-paramètres :  : ")
    print(f"Erreur d'entraînement : {err_train2} %")
    print(f"Erreur de validation : {err_valid2} %")
    print("")

    plt.figure(0)
    plt.bar(["Non normalisé, entraînement","Non normalisé, validation","Normalisé, entraînement","Normalisé, validation"],[err_train1, err_valid1, err_train2, err_valid2])
    plt.title("SVM - Comparaison erreurs d'entraînement et de validation entre données non normalisées et normalisées")
    plt.xlabel("Non normalisé / normalisé")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)

    # Test 2 : comparaison des résultats selon le terme de régularisation :
    err_train3, err_valid3 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l1", learning_rate="constant", k=10, eta=0.001)
    err_train4, err_valid4 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    err_train5, err_valid5 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="elasticnet", learning_rate="constant", k=10, eta=0.001)
    
    print("# Données normalisées, penalty l1 - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train3} %")
    print(f"Erreur de validation : {err_valid3} %")
    print("")

    print("# Données normalisées, penalty l2 - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train4} %")
    print(f"Erreur de validation : {err_valid4} %")
    print("")

    print("# Données normalisées, penalty elasticnet - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train5} %")
    print(f"Erreur de validation : {err_valid5} %")
    print("")

    plt.figure(1)
    plt.bar(["l1 entraînement","l1 validation","l2 entraînement","l2 validation", "elasticnet entraînement","elasticnet validation"],[err_train3, err_valid3, err_train4, err_valid4, err_train5, err_valid5])
    plt.title("SVM - Comparaison erreurs d'entraînement et de validation entre constant, optimal et invscaling learning rate")
    plt.xlabel("Penalty")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)


    # Test 3 : comparaison des résultats selon le learning_rate :
    err_train6, err_valid6 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="constant", k=10, eta=0.001)
    err_train7, err_valid7 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="optimal", k=10, eta=0.001)
    err_train8, err_valid8 = testS.test(x_train_normalized, t_train, reference, lamb=0.01, recherche_hyper_parametres=False, penalty="l2", learning_rate="invscaling", k=10, eta=0.001)
    
    print("# Données normalisées, penalty l2, learning_rate constant - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train6} %")
    print(f"Erreur de validation : {err_valid6} %")
    print("")

    print("# Données normalisées, penalty l2, learning_rate optimal - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train7} %")
    print(f"Erreur de validation : {err_valid7} %")
    print("")

    print("# Données normalisées, penalty l2, learning_rate invscaling - pas de recherche d'hyper-paramètres : ")
    print(f"Erreur d'entraînement : {err_train8} %")
    print(f"Erreur de validation : {err_valid8} %")
    print("-------------------------------")

    plt.figure(2)
    plt.bar(["constant entraînement","constant validation","optimal entraînement","optimal validation", "invscaling entraînement","invscaling validation"],[err_train6, err_valid6, err_train7, err_valid7, err_train8, err_valid8])
    plt.title("SVM - Comparaison erreurs d'entraînement et de validation entre constant, optimal et invscaling learning rate")
    plt.xlabel("Learning rate")
    plt.ylabel("Erreur (%)")
    plt.ylim(0,100)

    plt.show()

if __name__=="__main__":
    main()
