import lireDonnees
import perceptron
import visualiserDonnees
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
    erreur_train = round(100*np.count_nonzero(erreur_train_array)/len(erreur_train_array),1)
    print(f"Erreur d'entraînement : {erreur_train} %")
    print("------------------------------")

    # Prédiction (données de test) :
    pred_test = np.array([perc.prediction(x,reference) for x in x_test])
    #print(pred_test)

def main():
    # Extraction des données d'entraînement et de test
    rd = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
    x_train,t_train = rd.extract_train_data()
    x_test = rd.extract_test_data()
    reference = rd.create_reference() #liste des différentes espèces possibles

    # Visualisation des données d'entraînement
    vd = visualiserDonnees.visualiserDonnees(x_train,x_test,t_train,reference)
    vd.visualiserEntrainement()

    # Modèle du perceptron :
    evaluate_perceptron(x_train,t_train,x_test,0.01,reference)

if __name__=="__main__":
    main()
