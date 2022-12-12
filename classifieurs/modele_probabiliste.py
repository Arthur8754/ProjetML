#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, \
    recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


# define the GlobalModelClassifier class

class GlobalModelClassifier:

    # initialize the class with a scikit-learn classifier and a dictionary of 
    # hyperparameters to search over

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

    def hyperparameter_search(
        self,
        X,
        y,
        k,
        graphic=False,
        ):

        clf = GridSearchCV(estimator=self.classifier,
                           param_grid=self.parameters, cv=k, verbose=1,
                           scoring='accuracy')
        clf.fit(X, y)

        if graphic:
            if len(self.parameters)>2 : 
                print("No plot if there is more than 3 parameters")
            else :     
                def plot_grid_search(cv_results, grid_param_1, name_param_1, grid_param_2=None, name_param_2=None):
                    # Plot Grid search scores   
                    plt.figure(figsize=(10,10))

                    # Get Test Scores Mean
                    scores_mean = cv_results['mean_test_score']
                    
                    if(grid_param_2!=None):
                        scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

                        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
                        for idx, val in enumerate(grid_param_2):
                            plt.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

                    else : 
                        plt.plot(grid_param_1, scores_mean, '-o')

                    plt.title("Grid Search Scores", fontsize=20, fontweight='bold')
                    plt.xlabel(name_param_1, fontsize=16)
                    plt.ylabel('CV Average Score', fontsize=16)
                    plt.legend(loc="best", fontsize=15)
                    plt.grid('on')

                if len(self.parameters)>1: 
                    para1_name = list(self.parameters.keys())[0]
                    para1 = self.parameters[para1_name]
                    para2_name = list(self.parameters.keys())[1]
                    para2 = self.parameters[para2_name]

                else :
                    para1_name = list(self.parameters.keys())[0]
                    para1 = self.parameters[para1_name]
                    para2_name = None
                    para2 = None

                plot_grid_search(clf.cv_results_, para1 , para1_name, para2, para2_name)

        return clf.best_params_

    # define a method to calculate different metrics of the classifier on data

    def error(
        self,
        y_pred,
        y,
        average='micro',
        ):
        f1 = f1_score(y, y_pred, average=average)
        return {'f1-score': f1}


# define the GaussianNB_Classifier class that inherits from the 
# GlobalModelClassifier class

class GaussianNB_Classifier(GlobalModelClassifier):

    # initialize the class with a dictionary of hyperparameters to search over

    def __init__(self, parameters=1e-9):

        # call the __init__ method of the parent GlobalModelClassifier class, 
        # passing the GaussianNB classifier and the hyperparameters

        super().__init__(GaussianNB(), parameters)


class MultinomialNB_Classifier(GlobalModelClassifier):

    def __init__(self, parameters=1e-9):
        super().__init__(MultinomialNB(), parameters)


class BernoulliNB_Classifier(GlobalModelClassifier):

    def __init__(self, parameters=1e-9):
        super().__init__(BernoulliNB(), parameters)


class LogisticRegression_Classifier(GlobalModelClassifier):

    def __init__(self, parameters={'C': 1.0}):
        super().__init__(LogisticRegression(), parameters)


class SVM_Classifier(GlobalModelClassifier):

    def __init__(self, parameters={'alpha': 0.0001}):
        super().__init__(SGDClassifier(loss='hinge'), parameters)


class Perceptron_Classifier(GlobalModelClassifier):

    def __init__(self, parameters={'alpha': 0.0001}):
        super().__init__(SGDClassifier(loss='perceptron'), parameters)


class RandomForest_Classifier(GlobalModelClassifier):

    def __init__(self, parameters={'n_estimators': 100}):
        super().__init__(RandomForestClassifier(), parameters)


class KNN_Classifier(GlobalModelClassifier):

    def __init__(self, parameters={'n_neighbors': 5}):
        super().__init__(KNeighborsClassifier(), parameters)
