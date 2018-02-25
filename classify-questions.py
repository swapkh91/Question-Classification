from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from create_vectors import create_vector

# def precision(y_true, y_pred, strategy='weighted'):
#     return metrics.precision_score(y_true, y_pred, average=strategy)

# def recall(y_true, y_pred, strategy='weighted'):
#     return metrics.recall_score(y_true, y_pred, average=strategy)

# def f1_score(y_true, y_pred, strategy='weighted'):
#     return metrics.f1_score(y_true, y_pred, average=strategy)

# def training_error(y_true, y_pred):
#     prec = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     return prec, rec, f1

def predict_question_category(encoder, clf):
    print "Type exit to exit"
    print "Enter question: "
    question = raw_input()
    while (question != 'exit'):
        predicted_cat = encoder.inverse_transform(clf.predict([create_vector(question.lower())]))
        print "Predicted Category:", predicted_cat[0]
        print "Enter question: "
        question = raw_input()

if __name__ == '__main__':
    categories = ['when', 'what', 'who', 'affirmation', 'unknown']
    encoder = LabelEncoder()
    encoder.fit(categories)

    clf = joblib.load('model/trained_model.pkl')

    predict_question_category(encoder, clf)