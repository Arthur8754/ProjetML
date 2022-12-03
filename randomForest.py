"""
Implémentation de l'arbre de décision.
"""

from sklearn.ensemble import RandomForestClassifier
import lireDonnees

class randomForest:

    def __init__(self):
        self.clf = RandomForestClassifier()

    def entrainement(self, x_train, t_train):
        self.clf.fit(x_train, t_train)

    def prediction(self, x_tab):
        classes = self.clf.predict(x_tab)
        return classes

def main():
    # Lecture des données :
    rd = lireDonnees.lireDonnees("data/train.csv","data/test.csv","data/sample_submission.csv")
    x_train,t_train = rd.extract_train_data()
    
    # Modèle :
    modele = randomForest()
    modele.entrainement(x_train, t_train)
    classes = modele.prediction(x_train)
    print("Classes prédites :")
    print(classes)

if __name__=="__main__":
    main()