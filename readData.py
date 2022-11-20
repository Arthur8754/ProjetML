import pandas as pd

class readData:
    def __init__(self,train_filename, test_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename

    def extract_train_data(self):
        df_train = pd.read_csv(self.train_filename)
        x_train = df_train.iloc[:,2:]
        t_train = df_train.iloc[:,[1]]
        return x_train, t_train

    def extract_test_data(self):
        df_test = pd.read_csv(self.test_filename)
        x_test = df_test.iloc[:,1:]
        return x_test