from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score,f1_score, precision_recall_fscore_support, confusion_matrix,classification_report
import pandas as pd
from sklearn.model_selection import StratifiedKFold




class ClfSwitcher(BaseEstimator):
    
    def __init__(self, estimator=RandomForestClassifier()):
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)


def score(y_true, y_pred, index):    
    metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    performance = {'precision': metrics[0], 'recall': metrics[1], 'f1': metrics[2]}
    return pd.DataFrame(performance, index=[index])


# Returns Clf
def run_random_forest(dataloader):
    stop_words = set(stopwords.words('english'))
    dataloader.train_test_validate_split()
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words) 
    tfidf_train_vectors = tfidf_vectorizer.fit_transform(dataloader.x_train)
    tfidf_test_vectors = tfidf_vectorizer.transform(dataloader.x_test)
    clf = ClfSwitcher()
    clf.fit(tfidf_train_vectors, dataloader.y_train)
    y_pred = clf.predict(tfidf_test_vectors)
    print(classification_report(dataloader.y_test,y_pred))
    return clf

def run_multioutput_clf(dataloader, clf=RandomForestClassifier()):
    stop_words = set(stopwords.words('english'))
    dataloader.train_test_validate_split()
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words) 
    tfidf_train_vectors = tfidf_vectorizer.fit_transform(dataloader.x_train)
    tfidf_test_vectors = tfidf_vectorizer.transform(dataloader.x_test)
    model = MultiOutputClassifier(estimator=clf)
    model.fit(tfidf_train_vectors, dataloader.y_train)
    y_pred = model.predict(tfidf_test_vectors)
    print(classification_report(dataloader.y_test,y_pred))
    return model

def run_logistic_regression(dataloader):    
    stop_words = set(stopwords.words('english'))
    dataloader.train_test_validate_split()
    # BinaryRelevance / ClassifierChain
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words)), ('clf', ClassifierChain(LogisticRegression(solver='sag')))])
    pipeline.fit(dataloader.x_train, dataloader.y_train)
    predictions = pipeline.predict(dataloader.x_test)
    print('Macro F1 score of Logistic Regression is is ',f1_score(dataloader.y_test, predictions, average="macro"))
    return pipeline

if __name__ == '__main__':
    print('Machine Learning Algos Util')