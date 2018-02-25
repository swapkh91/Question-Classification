from sklearn import linear_model
import string
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from create_vectors import create_vector

categories = ['when', 'what', 'who', 'affirmation', 'unknown']

training_data_path = "data/train.txt"

def load_data(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            question, label = line.split(",,,", 1)
            res.append((question.strip(), label.strip()))
    return res


train_data = load_data(training_data_path)
question_vectors = np.asarray([create_vector(line[0]) for line in train_data])
encoder = LabelEncoder()
encoder.fit(categories)
train_labels = encoder.transform([line[1] for line in train_data])

clf = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf.fit(question_vectors, train_labels)

joblib.dump(clf, 'model/trained_model.pkl') 

print("Saved model to disk")