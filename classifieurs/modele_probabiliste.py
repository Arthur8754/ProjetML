from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# define the GlobalModelClassifier class
class GlobalModelClassifier:
    # initialize the class with a scikit-learn classifier and a dictionary of hyperparameters to search over
    def __init__(self, classifier, parameters):
        self.classifier = classifier
        self.parameters = parameters

    # define a method to train the classifier on training data
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    # define a method to make predictions on new data
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    # define a method to set the hyperparameters of the classifier
    def setParameter(self, parameter):
        self.classifier.set_params(**parameter)

    # define a method to search for the best hyperparameters for the classifier
    def hyperparameter_search(self, X, y, k, graphic=False):

        clf = GridSearchCV(estimator=self.classifier,param_grid=self.parameters,cv=k,verbose=1, scoring='accuracy')
        clf.fit(X, y)
        
        if graphic:
            
            cvres = clf.cv_results_
            list_param = []

            for i in range(0,len(cvres["params"])):
                list_param.append(list(cvres["params"][i].values())[0])

            ymax = np.ones(len(cvres["mean_test_score"]))*max(cvres["mean_test_score"])
            
            C = list(self.parameters.values())[0]
            plt.figure(figsize=(10,10))
            plt.plot(C, cvres["mean_test_score"])
            plt.plot(C, ymax, label='Best value is : ' + '{:1.3f}'.format(max(cvres["mean_test_score"])) + ' for var = ' + '{:1.5f}'.format(list(clf.best_params_.values())[0]))
            plt.legend()
            plt.xscale('log')
            plt.xlabel('Value of variance')
            plt.ylabel('Accuracy')
            plt.show()

        return clf.best_params_

    # define a method to calculate different metrics of the classifier on data
    def error(self, y_pred, y, average="micro"):        
        f1 = f1_score(y, y_pred,average=average)
        return {'f1-score': f1}

# define the GaussianNB_Classifier class that inherits from the GlobalModelClassifier class
class GaussianNB_Classifier(GlobalModelClassifier):
    # initialize the class with a dictionary of hyperparameters to search over
    def __init__(self, parameters=1e-9):
        # call the __init__ method of the parent GlobalModelClassifier class, passing the GaussianNB classifier and the hyperparameters
        super().__init__(GaussianNB(), parameters)

class MultinomialNB_Classifier(GlobalModelClassifier):
    def __init__(self, parameters=1e-9):
        super().__init__(MultinomialNB(), parameters)

class BernoulliNB_Classifier(GlobalModelClassifier):
    def __init__(self, parameters=1e-9):
        super().__init__(BernoulliNB(), parameters)

class LogisticRegression_Classifier(GlobalModelClassifier):
    def __init__(self, parameters={"C":1.0}):
        super().__init__(LogisticRegression(),parameters)

class SVM_Classifier(GlobalModelClassifier):
    def __init__(self, parameters={'alpha':0.0001}):
        super().__init__(SGDClassifier(loss='hinge'),parameters)

class Perceptron_Classifier(GlobalModelClassifier):
    def __init__(self, parameters={'alpha':0.0001}):
        super().__init__(SGDClassifier(loss='perceptron'),parameters)

class RandomForest_Classifier(GlobalModelClassifier):
    def __init__(self,parameters={'n_estimators':100}):
        super().__init__(RandomForestClassifier(),parameters)

class KNN_Classifier(GlobalModelClassifier):
    def __init__(self,parameters={'n_neighbors':5}):
        super().__init__(KNeighborsClassifier(),parameters)