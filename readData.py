import pandas as pd

class readData:
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

    def create_reference(self):
        """
        Crée la liste des références : tableau des espèces possibles, avec leurs indices
        """
        df_reference = pd.read_csv(self.reference_filename)
        reference = df_reference.columns.to_numpy()
        return reference[1:] #on ne prend pas id