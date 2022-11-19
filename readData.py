import pandas as pd

def read_train_data(filename):
    df_train = pd.read_csv(filename)
    x_train = df_train.iloc[:,2:]
    t_train = df_train.iloc[:,[1]] #species
    return x_train,t_train

def read_test_data(filename):
    df_test = pd.read_csv(filename)
    x_test = df_test.iloc[:,1:]
    return x_test

x_train,t_train = read_train_data("data/train.csv")
x_test = read_test_data("data/test.csv")

print(x_train)
print("")
print(t_train)
print("")
print(x_test)