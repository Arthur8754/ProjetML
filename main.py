import readData

def main():
    # Extraction des données d'entraînement et de test
    rd = readData.readData("data/train.csv","data/test.csv")
    x_train,t_train = rd.extract_train_data()
    x_test = rd.extract_test_data()
    print("x_train : ")
    print(x_train)
    print("t_train : ")
    print(t_train)
    print("x_test : ")
    print(x_test)

if __name__=="__main__":
    main()
