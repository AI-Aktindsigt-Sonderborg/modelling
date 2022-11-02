import os.path
import pickle

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from local_constants import PREP_DATA_DIR, MODEL_DIR
from utils.helpers import read_jsonlines


if __name__ == '__main__':

    model_filename = 'classifiers/svm_02.sav'

    # handle labels
    label2id = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhvervsudvikling': 2,
                'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
                'Sundhed og ældre': 6, 'Økonomi og administration': 7, 'Økonomi og budget': 8}
    label_list = list(label2id)
    id2label = {v: k for k, v in label2id.items()}

    train_json = read_jsonlines(input_dir=PREP_DATA_DIR, filename='train_classified')
    test_json = read_jsonlines(input_dir=PREP_DATA_DIR, filename='test_classified')


    train_sentences = [x['text'] for x in train_json]
    train_labels = [x['label'] for x in train_json]

    test_sentences = [x['text'] for x in test_json]
    test_labels = [x['label'] for x in test_json]

    vectorizer = TfidfVectorizer(max_features=919)
    X_train = vectorizer.fit_transform(train_sentences)
    y_train = [label2id[x] for x in train_labels]

    X_test = vectorizer.fit_transform(test_sentences)
    y_test = [label2id[x] for x in test_labels]

    classifier = svm.SVC(kernel='rbf', C=1)
    classifier.fit(X=X_train, y=y_train)

    pickle.dump(classifier, open(model_filename, 'wb'))

    loaded_model = pickle.load(open(model_filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print("score:" + result)

    predictions = loaded_model.predict(X_test)
    print(predictions)

    # scores = cross_val_score(clf, X, y, cv=10)
    # print(scores)

    print()