from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from local_constants import PREP_DATA_DIR
from utils.helpers import read_jsonlines

data = read_jsonlines(input_dir=PREP_DATA_DIR, filename='test_classified')

vectorizer = TfidfVectorizer()

sentences = [x['text'] for x in data]


labels = [x['label'] for x in data]

label2id = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhvervsudvikling': 2,
            'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
            'Sundhed og ældre': 6, 'Økonomi og administration': 7, 'Økonomi og budget': 8}

label_list = list(label2id)

id2label = {v: k for k, v in label2id.items()}

X = vectorizer.fit_transform(sentences)
y = [label2id[x] for x in labels]


clf = svm.SVC(kernel='linear', C=1)

scores = cross_val_score(clf, X, y, cv=10)
print(scores)

print()