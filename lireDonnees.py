import pandas as pd
import numpy as np

class lireDonnees:
    def __init__(self,train_filename, test_filename, reference_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.reference_filename = reference_filename

    def extract_train_data(self):
        df_train = pd.read_csv(self.train_filename)
        x_train = df_train.iloc[:,2:].to_numpy()
        t_train = df_train.iloc[:,1].to_numpy()
        return x_train, t_train

    def extract_test_data(self):
        df_test = pd.read_csv(self.test_filename)
        x_test = df_test.iloc[:,1:].to_numpy()
        return x_test

    def normalize_data(self, x_train, x_test):
        transpose_train = np.transpose(x_train)
        transpose_test = np.transpose(x_test)
        for i in range(transpose_train.shape[0]): #pour chaque colonne de x_train, ie chaque ligne de transpose, on va calculer la moyenne de la composante
            mean_train = np.mean(transpose_train[i])
            std_train = np.std(transpose_train[i])
            transpose_train[i] = (transpose_train[i]-mean_train)/std_train

            mean_test = np.mean(transpose_test[i])
            std_test = np.std(transpose_test[i])
            transpose_test[i] = (transpose_test[i]-mean_test)/std_test

        return np.transpose(transpose_train),np.transpose(transpose_test)
        
    def create_reference(self):
        """
        Crée la liste des références : tableau des espèces possibles, avec leurs indices
        """
        df_reference = pd.read_csv(self.reference_filename)
        reference = df_reference.columns.to_numpy()
        return reference[1:] #on ne prend pas id