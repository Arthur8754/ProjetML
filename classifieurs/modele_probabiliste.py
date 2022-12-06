from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

class ProbabilistModelClassifier:
    def __init__(self, classifier, parameters):
        self.classifier = classifier
        self.parameters = parameters
    
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
    def setParameter(self, parameter):
        self.classifier.set_params(**parameter)

    def hyperparameter_search(self, X, y, k):
        if self.parameters==None :
            return None
        
        clf = GridSearchCV(estimator=self.classifier,param_grid=self.parameters,cv=k,verbose=1, scoring='accuracy')
        clf.fit(X, y)
        return clf.best_params_
    
    def error(self, y_pred, y, average="micro"):        
        # calculate the precision, accuracy, recall, and F1-score
        precision = precision_score(y, y_pred,average=average)
        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred,average=average)
        f1 = f1_score(y, y_pred,average=average)
        
        # return a dictionary with the scores
        return {'precision': precision, 'accuracy': accuracy, 'recall': recall, 'f1': f1}

class GaussianNBClassifier(ProbabilistModelClassifier):
    def __init__(self, parameters=1e-9):
        super().__init__(GaussianNB(), parameters)

class MultinomialNBClassifier(ProbabilistModelClassifier):
    def __init__(self, parameters=1e-9):
        super().__init__(MultinomialNB(), parameters)

class BernoulliNBClassifier(ProbabilistModelClassifier):
    def __init__(self, parameters=1e-9):
        super().__init__(BernoulliNB(), parameters)

class LogisticRegressionClassifier(ProbabilistModelClassifier):
    def __init__(self, parameters={"C":1.0}):
        super().__init__(LogisticRegression(),parameters)