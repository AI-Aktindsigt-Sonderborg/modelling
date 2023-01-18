# pylint: skip-file
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyinputplus import inputInt, inputChoice
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_utils.custom_dataclasses import CosineSimilarity
from modelling_utils.input_args import SequenceModellingArgParser
from modelling_utils.sequence_classification import SequenceClassification
from utils.helpers import TimeCode, bcolors


warnings.filterwarnings("ignore")

sc_parser = SequenceModellingArgParser()

code_timer = TimeCode()
label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
              'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

top_n = 5

LABELS = list(label_dict)

args = sc_parser.parser.parse_args()

# args.model_name = 'sarnikowski/convbert-small-da-cased'
args.model_name = 'last_model-2022-12-21_10-53-25'
args.labels = LABELS
args.evaluate_during_training = False
args.load_alvenir_pretrained = True
# args.device = 'cpu'
# args.test_data = 'test_local.json'
# ToDo: Figure out how to deal with max_length
args.max_length = 64

modelling = SequenceClassification(args)

modelling.load_data(train=False, test=True)

# test_data_wrapped = modelling.tokenize_and_wrap_data(data=modelling.test_data)
# test_loader = DataLoader(dataset=test_data_wrapped,
#                          # collate_fn=modelling.data_collator,
#                          batch_size=1,
#                          shuffle=False)

model = modelling.get_model()

# embedding_outputs = modelling.create_embeddings_windowed(
#     model=model, save_dict=True)

with open("data/test_data/test_embeddings", "rb") as fp:
    embedding_outputs = pickle.load(fp)

# ToDo: experiment with pca stuff?
X = np.array([x.embedding for x in embedding_outputs]).astype(float)
y_true = np.array([int(modelling.label2id[x.label]) for x in embedding_outputs])
y_pred = np.array([int(modelling.label2id[x.prediction]) for x in embedding_outputs])

from utils.projpursuit import projpursuit
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from umap import UMAP

sc = StandardScaler(with_mean=False)
sc.fit(X)
X_std = sc.transform(X)

# pca = PCA(n_components=240)
# iso = Isomap(n_components=2)
# # tsne = TSNE(n_components=2, n_iter=5000, init='pca', random_state=1)
# lle = LocallyLinearEmbedding(n_components=10, method='modified', n_neighbors=40)

# best so far: metric='manhattan', target_n_neighbors=29, n_components=10
umap = UMAP(metric='manhattan', target_n_neighbors=29, n_components=10,
            random_state=0)

X_transformed = umap.fit_transform(X)
# X_pca = pca.fit_transform(X_std)
# X_iso = iso.fit_transform(X_std)
# X_tsne_std = iso.fit_transform(X_std)
# X_tsne = iso.fit_transform(X)
# # X_tsne_pca = iso.fit_transform(X_std)
# X_lle = lle.fit_transform(X_std)
#
# X_ppa = projpursuit(X_pca)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],
#            c=y_true, cmap='rainbow')
# plt.show()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for i, label in enumerate(LABELS):
    plt.scatter(X_transformed[y_true == int(modelling.label2id[label]), 0],
                X_transformed[y_true == int(modelling.label2id[label]), 1],
                label=label, marker='.', c = colors[i], s = 20)
    plt.scatter(X_transformed[y_pred == int(modelling.label2id[label]), 0],
                X_transformed[y_pred == int(modelling.label2id[label]), 1],
                marker='o', edgecolors=colors[i],
                s = 40, facecolors='none', linewidths=1)

plt.title('Test sæt visualiseret.\n'
          '"." illustrerer den sande kategori og "o" markerer modellens prediktion')
plt.legend()
plt.legend()
plt.show()