import os
import time
from os.path import exists
from typing import List

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from utils.helpers import TimeCode


class DataVisualisation:

    def __init__(self, corpus: List[str], pca: PCA = None, tsne: TSNE = None, kmeans: KMeans = None):
        self.tsne_results = None
        self.vectorizer = TfidfVectorizer()
        self.pca = pca
        self.kmeans = kmeans
        self.corpus = corpus
        self.pca_result = None
        self.X = self.vectorizer.fit_transform(corpus)
        self.names = self.vectorizer.get_feature_names_out()
        self.tsne = tsne

    def compute_pca(self):
        self.pca_result = self.pca.fit_transform(self.X.toarray())
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

    def k_means_viz(self, d_type: str='tsne', dim: int = 2, save_plot: bool = True):

        if d_type == 'tsne':
            label = self.kmeans.fit_predict(self.tsne_results)
            u_labels = np.unique(label)

            title = f'{dim}d_type-{d_type}_n_pca-{self.pca.n_components}_n_tsne-{self.tsne.n_components}_' \
                    f'n_clusters-{self.kmeans.n_clusters}'
            save_path = f'plots/{title}.png'
            if not exists(save_path):
                if dim == 3:
                    self.plot3d(title, save_path, label, u_labels, save_plot, d_type)
                else:
                    self.plot2d(title, save_path, label, u_labels, save_plot, d_type)

        elif d_type == 'pca':
            label = self.kmeans.fit_predict(self.pca_result)
            u_labels = np.unique(label)
            title = f'{dim}d_type-{d_type}_n_pca-{self.pca.n_components}_' \
                    f'n_clusters-{self.kmeans.n_clusters}'
            save_path = f'plots/{title}.png'
            if not exists(save_path):
                if dim == 3:
                    self.plot3d(title, save_path, label, u_labels, save_plot, d_type)

                else:
                    self.plot2d(title, save_path, label, u_labels, save_plot, d_type)

    def plot3d(self, title, save_path, label, u_labels, save_plot, d_type):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if d_type == 'pca':
            for i in u_labels:
                ax.scatter(self.pca_result[label == i, 0], self.pca_result[label == i, 1],
                           self.pca_result[label == i, 2], label=i)
        elif d_type == 'tsne':
            for i in u_labels:
                ax.scatter(self.tsne_results[label == i, 0], self.tsne_results[label == i, 1],
                           self.tsne_results[label == i, 2], label=i)

        if save_plot:

            ax.set_title(title)

            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot2d(self, title, save_path, label, u_labels, save_plot, d_type):
        for i in u_labels:
            plt.scatter(self.pca_result[label == i, 0], self.pca_result[label == i, 1],
                        label=i)
        if d_type == 'pca':
            for i in u_labels:
                plt.scatter(self.pca_result[label == i, 0], self.pca_result[label == i, 1],
                            label=i)
        elif d_type == 'tsne':
            for i in u_labels:
                plt.scatter(self.tsne_results[label == i, 0], self.tsne_results[label == i, 1],
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

if __name__ == '__main__':


    code_timer = TimeCode()
    make_plots(corpus=['a', 'b'], dims=[2, 3], pca_components=[10, 20, 30, 40],
               tsne_components=[2, 3], clusters=[3, 4, 5, 6])

