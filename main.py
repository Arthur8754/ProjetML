import readData
import perceptron
import numpy as np

def evaluate_perceptron(x_train,t_train,x_test,lamb,reference):
    print("------------------------------")
    print("Perceptron :")
    print("Pas de recherche d'hyper-paramètres")
    
    perc = perceptron.perceptron(lamb=lamb)

    # Entraînement du perceptron :
    perc.entrainement(x_train,t_train)

    # Prédiction (données d'entraînement):
    pred_train = np.array([perc.prediction(x,reference) for x in x_train])

    # Erreur d'entraînement :
    erreur_train_array = np.array([perc.erreur(t,pred) for t,pred in zip(t_train,pred_train)])
    erreur_train = 100*np.count_nonzero(erreur_train_array)/len(erreur_train_array)
    print(f"Erreur d'entraînement : {erreur_train} %")
    print("------------------------------")
    perc.afficher_donnees(x_train)

def main():
    # Extraction des données d'entraînement et de test
    rd = readData.readData("data/train.csv","data/test.csv","data/sample_submission.csv")
    x_train,t_train = rd.extract_train_data()
    x_test = rd.extract_test_data()
    reference = rd.create_reference()
    #print("x_train : ")
    #print(x_train)
    #print("t_train : ")
    #print(t_train)
    #print("x_test : ")
    #print(x_test)
    #print("Références :")
    #print(reference)

    # Modèle du perceptron :
    evaluate_perceptron(x_train,t_train,x_test,0.01,reference)

if __name__=="__main__":
    main()
