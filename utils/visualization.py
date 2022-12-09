import json
import os
import time
from os.path import exists
from typing import List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math
import seaborn as sn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from wordcloud import WordCloud
from nltk.corpus import stopwords
from utils.helpers import TimeCode, read_jsonlines
from local_constants import PREP_DATA_DIR

from wordcloud import WordCloud, STOPWORDS

stopwords_union = set.union(set(stopwords.words('danish')), STOPWORDS)

# danish_stopwords = set(stopwords.words('danish'))
stopwords_union.update(
    ["ved", "kan", "samt", "så", "se", "får", "få", "f eks", "f", "eks", "Stk", "stk", "må", "der",
     "for", "fx", "bl", ""])



class DataVisualisation:

    def __init__(self, corpus: List[str], pca: PCA = None, tsne: TSNE = None, kmeans: KMeans = None):
        self.tsne_results = None
        self.vectorizer = TfidfVectorizer()
        self.pca = pca
        self.kmeans = kmeans
        self.corpus = corpus
        self.pca_result = None
        corpus_chunks = np.array_split(corpus, 10)
        self.X_chunks = []
        for chunk in corpus_chunks:
            X = self.vectorizer.fit_transform(chunk)
            self.X_chunks.append(X)

        self.names = self.vectorizer.get_feature_names_out()
        self.tsne = tsne

    def compute_pca(self):
        pca_chunks = []
        for chunk in self.X_chunks:
            pca_result = self.pca.fit_transform(chunk.toarray())
            pca_chunks.extend(pca_result)

        self.pca_result = np.array(pca_chunks)
        print(f'Cumulative explained variation for N principal components: '
              f'{np.sum(self.pca.explained_variance_ratio_)}')


    def compute_tsne(self, on_pca: bool = False):
        if on_pca:
            self.tsne_results = self.tsne.fit_transform(self.pca_result)
        else:
            self.tsne_results = self.tsne.fit_transform(self.X)

    def compute_tsne_on_pca(self):
        time_start = time.time()
        if self.pca_result is None:
            self.compute_pca()
        self.compute_tsne(on_pca=True)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    def k_means_viz(self, d_type: str='tsne', dim: int = 2, save_plot: bool = True,
                    labels: List[int] = None):

        if d_type == 'tsne':
            preds = self.kmeans.fit_predict(self.tsne_results)

            title = f'{dim}d_type-{d_type}_n_pca-{self.pca.n_components}_n_tsne-{self.tsne.n_components}_' \
                    f'n_clusters-{self.kmeans.n_clusters}'
            save_path = f'plots/{title}.png'
            if not exists(save_path):
                if dim == 3:
                    self.plot3d(title, save_path, preds, labels, save_plot, d_type)
                else:
                    self.plot2d(title, save_path, preds, labels, save_plot, d_type)

        elif d_type == 'pca':
            preds = self.kmeans.fit_predict(self.pca_result)

            title = f'{dim}d_type-{d_type}_n_pca-{self.pca.n_components}_' \
                    f'n_clusters-{self.kmeans.n_clusters}'
            save_path = f'plots/{title}.png'
            if not exists(save_path):
                if dim == 3:
                    self.plot3d(title, save_path, preds, labels, save_plot, d_type)

                else:
                    self.plot2d(title, save_path, preds, labels, save_plot, d_type)

    def plot3d(self, title, save_path, pred, u_labels, save_plot, d_type):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if d_type == 'pca':
            for i in u_labels:
                ax.scatter(self.pca_result[pred == i, 0], self.pca_result[pred == i, 1],
                           self.pca_result[pred == i, 2], label=i)
        elif d_type == 'tsne':
            for i in u_labels:
                ax.scatter(self.tsne_results[pred == i, 0], self.tsne_results[pred == i, 1],
                           self.tsne_results[pred == i, 2], label=i)

        if save_plot:

            ax.set_title(title)

            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot2d(self, title, save_path, preds, labels, save_plot, d_type):
        u_labels = np.unique(labels)
        for i in u_labels:
            plt.scatter(labels[preds == i, 0], self.pca_result[preds == i, 1],
                        label=i, )
        if d_type == 'pca':
            for i in u_labels:
                plt.scatter(self.pca_result[preds == i, 0], self.pca_result[preds == i, 1],
                            label=i)
        elif d_type == 'tsne':
            for i in u_labels:
                plt.scatter(self.tsne_results[preds == i, 0], self.tsne_results[preds == i, 1],
                            label=i)
        plt.legend()
        if save_plot:

            plt.title(title)

            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        plt.show()

def plot_running_results(output_dir: str, epochs: int, lrs, accs, loss, epsilon: str = "",
                         delta: str = ""):
    file_path = os.path.join(output_dir, 'results')
    title = os.path.join(f'Epochs: {epochs}, Epsilon: {epsilon},  Delta: {delta}')
    plt.ioff()
    learning_rates_steps = [int(x['step']) for x in lrs]
    learning_rates = [x['lr'] for x in lrs]

    accuracies_steps = [int(x['step']) for x in accs]
    accuracies = [x['acc'] for x in accs]

    losses_steps = [int(x['step']) for x in loss]
    losses = [x['loss'] for x in loss]

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(title, fontsize=14)

    ax1.plot(learning_rates_steps, learning_rates)
    ax1.set(ylabel='learning rate')

    ax2.plot(accuracies_steps, accuracies, 'orange')
    ax2.set(ylabel='accuracy')

    ax3.plot(losses_steps, losses, 'green')
    ax3.set(ylabel='loss', xlabel='step')
    plt.savefig(file_path)

def make_plots(corpus: List[str], clusters: List[int] = [3], dims: List[int] = [2],
               pca_components: List[int] = [3], tsne_components: List[int] = [2]):
    for pca_component in pca_components:
        pca = PCA(n_components=pca_component)
        for tsne_component in tsne_components:
            tsne = TSNE(n_components=tsne_component, verbose=0, perplexity=40, n_iter=1000)
            for n_clusters in clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                data_viz = DataVisualisation(corpus=corpus, tsne=tsne, pca=pca, kmeans=kmeans)
                data_viz.compute_pca()
                data_viz.compute_tsne_on_pca()
                for dim in dims:
                    data_viz.k_means_viz(d_type='pca', dim=dim)
                    if dim == 3 and tsne.n_components == 2:
                        print('cant plot tsne 3d with two components')
                    else:
                        data_viz.k_means_viz(d_type='tsne', dim=dim)


def wordclouds_classified(data: List[dict] = None, max_words: int = 75,
                      labels: List[str] = None):
    if labels:
        for label in labels:
            label_data = [x for x in data if x['klassifikation'] == label]
            all_text = ""
            for i, line in enumerate(label_data):
                all_text += line['text'] + " "


            wordcloud = WordCloud(stopwords=stopwords_union, max_words=max_words,
                                  background_color="white").generate(all_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Mest benyttede ord i {label}')
            plt.axis("off")
            plt.savefig(f'plots/class_wordclouds/word_cloud-{label}.png', bbox_inches='tight')
    else:
        all_text = ""
        for i, line in enumerate(data):
            all_text += line['text'] + " "

        wordcloud = WordCloud(stopwords=stopwords_union, max_words=max_words,
                              background_color="white").generate(all_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Mest benyttede ord totalt')
        plt.axis("off")
        plt.savefig(f'plots/class_wordclouds/word_cloud-total.png', bbox_inches='tight')



def barplot_muni_counts(data: List[dict] = None):

    muni_list = np.unique([x['kommune'] for x in data])

    splitted = [[y for y in data if y['kommune'] == x] for x in muni_list]

    labels = [x[0]['kommune'] for x in splitted]
    counts = [len(x) for x in splitted]

    fig = plt.figure(figsize=(7, 3))
    plt.bar(labels, counts)
    plt.xticks(rotation=90)
    plt.savefig(f'plots/class_wordclouds/class_sizes.png', bbox_inches='tight')
    plt.close()

def simple_barplot(labels: List[str], data: List[float]):

    fig = plt.figure(figsize=(7, 3))
    plt.bar(labels, data)
    plt.xticks(rotation=90)
    # plt.savefig(f'plots/labelled/class_sizes.png', bbox_inches='tight')
    plt.show()


def calc_f1_score(y_list, prediction_list, labels, conf_plot: bool = False):


    if conf_plot:
        conf_matrix = confusion_matrix(y_list, preds, labels=labels,
                                       normalize='true')
        df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g')
        plt.show()


    return precision_recall_fscore_support(y_list, preds, labels=labels, average='micro'), \
           f1_score(y_true=y_list, y_pred=preds, labels=labels, average=None)



if __name__ == '__main__':

    data = read_jsonlines(input_dir=PREP_DATA_DIR, filename='test_classified')
    # data = [x for x in data if not (isinstance(x['text'], float) and math.isnan(x['text'])) and not x['text_len'] > 3000]


    # barplot_muni_counts(data=data)



    sentences = [x['text'] for x in data]

    labels = [x['label'] for x in data]

    label2id = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhvervsudvikling': 2,
                'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
                'Sundhed og ældre': 6, 'Økonomi og administration': 7, 'Økonomi og budget': 8}

    label_list = list(label2id)

    id2label = {v: k for k, v in label2id.items()}
    label_ids = [label2id[x] for x in labels]

    # create word clouds
    # create_wordclouds_classified(data=data, labels=label_list)
    # wordclouds_classified(data=data)
    pca = PCA(n_components=2)
    tsne = TSNE()
    kmeans = KMeans(n_clusters=len(label2id), random_state=0)
    data_viz = DataVisualisation(corpus=sentences, pca=pca, kmeans=kmeans)
    # data_viz.compute_pca()
    # preds = data_viz.kmeans.fit_predict(data_viz.pca_result)
    # data_viz.k_means_viz(d_type='pca', dim=2, labels=label_ids)

    f_score, f1 = calc_f1_score(y_list=label_ids, prediction_list=preds, labels=label_list, conf_plot=True)



    print()


    # code_timer = TimeCode()
    # make_plots(corpus=['a', 'b'], dims=[2, 3], pca_components=[10, 20, 30, 40],
    #            tsne_components=[2, 3], clusters=[3, 4, 5, 6])

