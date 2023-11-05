import os
import gensim
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DataPreprocessing import preprocess_text
from sklearn import metrics
from sklearn.svm import SVC

current_file = __file__
f = os.path.dirname(os.path.abspath(current_file))


def word2Vec(text):
    if type(text) is list:
        tokens = text
    else:
        tokens = text.split(" ")
    model = gensim.models.KeyedVectors.load_word2vec_format(f + '/Word2Vec/baomoi.model.bin', binary=True)
    vectors = [model[token] for token in tokens if token in model]
    vector = np.mean(vectors, axis=0)
    return vector.reshape(1, -1) if type(text) is list else vector.reshape(-1)


class SVM_Model:
    def __init__(self):
        self._vectorizer = word2Vec
        self.clf = SVC(kernel='linear', C=1.0)
        self.data = pd.read_csv(f + "/datasets/result.csv", encoding="utf-8")

    def trainModel(self):
        X = self.data["Features"].apply(self._vectorizer)
        X = np.stack(X)
        y = self.data['Nhãn']

        self.clf.fit(X, y)
        joblib.dump(self.clf, f + "/modelData/SVC.model")

    def loadModel(self):
        self.clf = joblib.load(f + "/modelData/SVC.model")

    def evaluateModel(self):
        X = self.data['Features'].apply(self._vectorizer)
        X = np.stack(X)
        y = self.data['Nhãn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = SVC(kernel='linear', C=1.0)
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        print(accuracy_score(predicted, y_test) * 100)
        print(f"Classification report for classifier {model}: \n"
              f"{metrics.classification_report(y_test, predicted)}")

        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()

    def recommend(self, text):
        processed_text = preprocess_text(text.lower())
        X_test = self._vectorizer(processed_text)
        prediction = self.clf.predict(X_test)
        return prediction[0]
